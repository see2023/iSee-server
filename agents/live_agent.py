import asyncio
import json
import logging
import os
import sys
import time
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import config
from agents.tts_manager import TTSManager

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG if config.agents.log_debug else logging.INFO,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    logging.getLogger("websockets").setLevel(logging.ERROR)
setup_logging()
import numpy as np
import fast_whisper_stt
import xf_stt
import sense_voice
import chat_ext
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, REDIS_PREFIX, write_chat_to_redis, write_mq_to_redis, write_detected_names_to_redis

from dotenv import load_dotenv
load_dotenv(override=True)
from livekit import agents, rtc
from livekit.plugins import openai 
from detect.yolov8 import YoloV8Detector
from vad import VAD
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image
import matplotlib
from llm.llm_base import LLMBase, VisualLLMBase, KEEP_SILENT_RESPONSE
from control.simple_control import SimpleControl, Command
from agents.speaker import Speaker

import base64
from io import BytesIO
from llm.llm_base import Message, MessageRole
from llm.qwen2_vl_http_server_cli import send_message_to_server
from video_scene_monitor import VideoSceneMonitor

font_path = "/Library/Fonts/Arial Unicode.ttf" if sys.platform == "darwin" else "arial.ttf"
font_size = 40
font = ImageFont.truetype(font_path, font_size)

logger = logging.getLogger()


class ModelManager: # singleton
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._prompt = "你好！你吃饭了吗？吃过了，谢谢。"
        if config.agents.speaker_distance_threshold> 0:
            self.speaker_detector = Speaker(speaker_distance_threshold=config.agents.speaker_distance_threshold, min_chunk_duration=config.agents.min_chunk_duration)
        else:
            self.speaker_detector = None
        stt_type  = config.agents.stt_type
        if self._instance is not None:
            raise Exception('This is a singleton class, use get_instance() instead')
        if stt_type == "whisper_api":
            # api
            self.stt_model = openai.STT(language="zh", detect_language=False, model="whisper-1")
        elif stt_type == "local_whisper_original":
            # local original whisper
            self.stt_model = fast_whisper_stt.STT( use_fast_whisper=False, language="zh", model_size_or_path="medium", device="cuda", 
                                                            compute_type="default", beam_size=5, initial_prompt=self._prompt)
        elif stt_type == "local_whisper_fast":
            # local faster whisper
            self.stt_model = fast_whisper_stt.STT( use_fast_whisper=True, language="zh", model_size_or_path="large-v3", device="cuda", 
                                                            compute_type="default", beam_size=5, initial_prompt=self._prompt)
        elif stt_type == "xf_api":
            self.stt_model = xf_stt.STT(speaker_detector=self.speaker_detector)
        elif stt_type == "sense_voice_small":
            self.stt_model = sense_voice.SenseVoiceSTT(speaker_detector=self.speaker_detector)
        else:
            raise Exception("invalid whisper type: {}".format(stt_type))
        self.detector = YoloV8Detector()
        self.vad = VAD()



def get_model():
    model_manager = ModelManager.get_instance()
    return model_manager.stt_model, model_manager.detector, model_manager.vad, model_manager.speaker_detector

class LiveAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = LiveAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins

        self.stt_model, self.detector, self.vad, self.speaker_detector = get_model() 
        self.catch_follow_pos = False
        self.motion_control = SimpleControl(self.detector)

        logging.info("Live agent created")

        self.ctx = ctx
        self.tts_manager = TTSManager()  # 新增 TTSManager 实例
        self.chat = chat_ext.ChatExtManager(ctx.room)
        self.tts_manager.set_chat_manager(self.chat)  # 设置 chat manager
        self.redis = get_redis_client()
        self.remoteId: str = None
        self.stream_key: str = None
        self.currrent_img: Image = None
        self.currrent_vl_img: Image = None
        self.currrent_vl_time: float = 0
        self.current_move_time: float = 0
        self.current_detected_names = []
        self.detecting: bool = False
        self.start_catch_pic = False
        self.results = []
        self.video_streaming = False
        self.audio_streaming = False

        # 捕捉说话期间的图片
        self.image_list_for_speak = []
        self.max_images_for_speak = config.agents.max_images_for_speak  # 设置最大长度限制
        self.image_capture_interval_for_speak = 1  # 设置图像捕获间隔（秒）
        self.capture_images_for_speak = False
        self.last_capture_time_for_speak = 0
        self.caption_interval_for_speak = 60
        self.last_caption_time_for_speak = 0
        self.current_caption_for_speak = ""
        self.last_question_from_user = ""
        self.last_question_from_user_time = 0

        self.show_detected_results = config.agents.show_detected_results
        if self.show_detected_results and config.agents.enable_video:
            plt.ion()
            self.fig = plt.figure("Video")
            self.im =   plt.imshow(np.zeros((10, 10, 3)))

        if config.agents.scene_check_interval > 0:
            self.video_scene_monitor = VideoSceneMonitor(
                interval=config.agents.scene_check_interval,
                send_message_callback=self.send_to_app,  # 使用 send_to_app 方法
                send_to_server_callback=send_message_to_server
            )
        else:
            self.video_scene_monitor = None 
        self.last_scene_check_time = 0

        # setup callbacks
        def subscribe_cb(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.audio_streaming = True
                if config.agents.enable_audio:
                    self.ctx.create_task(self.audio_track_worker(track))
                if not self.remoteId:
                    self.remoteId = participant.identity
                    self.tts_manager.set_user_sid(participant.sid)
                    async def set_user_info(user_id: str):
                        self.stream_key = REDIS_CHAT_KEY + user_id
                        logging.info("================================== redis xtrim key: %s", self.stream_key)
                        await self.redis.xtrim(self.stream_key, 10000)
                        # set last user
                        await self.redis.set(REDIS_PREFIX+ "last_user", user_id)
                    loop = asyncio.get_event_loop()
                    loop.create_task(set_user_info(self.remoteId))
            elif track.kind == rtc.TrackKind.KIND_VIDEO:
                self.video_streaming = True
                if config.agents.enable_video:
                    self.ctx.create_task(self.video_track_worker(track))
                    self.ctx.create_task(self.video_detect_worker())
                    self.ctx.create_task(self.vision_lang_worker())

        async def process_chat(msg: chat_ext.ChatExtMessage):
            logging.info("received chat message: %s", msg.message)
            if msg.message == config.llm.vl_cmd_catch_pic:
                # 等待提示音
                await asyncio.sleep(3)
                self.start_catch_pic = True
                logging.info("start catching picture")
        
        def unsubscribe_cb(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if participant.identity == self.remoteId:
                logging.info("remote participant left, cleaning up")
                self.remoteId = None
                self.stream_key = None
                if self.video_scene_monitor:
                    self.video_scene_monitor.stop_monitoring()
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self.video_streaming = False
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self.audio_streaming = False

        self.ctx.room.on("track_subscribed", subscribe_cb)
        self.ctx.room.on("track_unsubscribed", unsubscribe_cb)
        def on_message(msg: chat_ext.ChatExtMessage):
            asyncio.ensure_future(process_chat(msg))
        self.chat.on("message_received", on_message)

    async def send_to_app(self, msg: str):
        if msg == KEEP_SILENT_RESPONSE:
            return
        logging.info("sending message to app: %s", msg)
        chat_msg = await self.chat.send_message(
            message=msg, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
        )
        await self.tts_manager.enqueue_message(chat_msg)  # 将消息加入 TTS 队列

    async def start(self):
        # 给一点时间让用户完全连接，以免错过欢迎消息
        await asyncio.sleep(1)

        # 创建任务以发送欢迎消息
        self.ctx.create_task(
            self.chat.send_message(
                message="Welcome to the live agent! Just speak to me."
            )
        )

        self.update_agent_state("listening")

        # 启动 TTS 管理器
        self.ctx.create_task(self.tts_manager.run_tts_msg())

        # 启动场景监控
        if self.video_scene_monitor:
            self.ctx.create_task(self.video_scene_monitor.start_monitoring())

    def update_agent_state(self, state: str):
        metadata = json.dumps(
            {
                "agent_state": state,
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))

    # 保存图片至本地磁盘，等待视觉语言模型处理
    async def vision_lang_worker(self):
        while self.video_streaming:
            try:
                await asyncio.sleep(0.01)
                if self.currrent_vl_img is None:
                    continue
                detecting_img: Image = self.currrent_vl_img
                self.currrent_vl_img = None
                self.currrent_vl_time = time.time()
                local_file = self.save_img(detecting_img)
                logging.info("vision_lang_worker save img to %s", local_file)
            except Exception as e:
                logging.error("error processing vision lang: %s", e)
                await asyncio.sleep(1)
        logging.info("vision lang worker done")

    # 如果超过了检测间隔，或者检测目标发生了变化，则需要将图片保存起来，下一步进行视觉语言处理
    async def check_vision_lang(self, detected_names, detecting_img: Image):
        if detecting_img.width < 300 or detecting_img.height < 300:
            return
        if self.start_catch_pic:
            self.start_catch_pic = False
            self.currrent_vl_img = detecting_img
            logging.info("catching picture ok")
            return
        now = time.time()
        if now - self.currrent_vl_time < config.agents.vision_lang_interval or config.agents.vision_lang_interval <= 0:
            return
        if len(detected_names) == 0:
            return
        if detected_names == self.current_detected_names and self.current_move_time < now - config.agents.vision_lang_interval:
            return
        logging.info("sending to vision lang model")
        self.currrent_vl_img = detecting_img


    # 在图像中检测跟随目标，并向app发送移动命令
    async def video_detect_worker(self):
        while self.video_streaming:
            try:
                await asyncio.sleep(0.001)
                if self.currrent_img is None:
                    continue
                self.detecting = True
                detecting_img = self.currrent_img
                self.currrent_img = None
                loop = asyncio.get_event_loop()
                self.results = await loop.run_in_executor(None, self.detector.detect, detecting_img)

                if self.catch_follow_pos:
                    start_rt, rt_msg = self.motion_control.start_follow(self.results)
                    logging.info("received motion command, starting motion control rt_msg: %s", rt_msg)
                    await self.chat.send_message(
                        message=rt_msg, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                    )
                    if start_rt:
                        self.catch_follow_pos = False
                else:
                    cmd = await self.motion_control.loop(self.results)
                    if cmd:
                        logging.info("received motion command, sending message to app: %s", cmd)
                        await self.chat.send_move_cmd(
                            cmd, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                        )
                        if cmd in [Command.FORWARD, Command.RIGHT_FRONT, Command.RIGHT, Command.RIGHT_BACK, Command.BACK, Command.LEFT_BACK, Command.LEFT, Command.LEFT_FRONT]:
                            self.current_move_time = time.time()
                        if cmd in [Command.RIGHT_FRONT, Command.LEFT_FRONT]:
                            logging.info("Forward turn ----------")
                            await asyncio.sleep(0.2)
                            await self.chat.send_move_cmd(
                                Command.FORWARD, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                            )
                            await asyncio.sleep(0.3)
                        elif cmd in [Command.LEFT_BACK, Command.RIGHT_BACK]:
                            logging.info("Back turn ----------")
                            await asyncio.sleep(0.2)
                            await self.chat.send_move_cmd(
                                Command.BACK, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                            )
                            await asyncio.sleep(0.3)
                        else:
                            await asyncio.sleep(0.5)

                        await self.chat.send_move_cmd(
                            Command.STOP, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                        )

                detected_names = [self.detector.names[int(i)] for i in self.results[0].boxes.cls]
                await self.check_vision_lang(detected_names, detecting_img)
                self.current_detected_names = detected_names
                if len(detected_names) > 0:
                    detected_names_str = ", ".join(detected_names)
                    logging.info("detected objects: %s", detected_names)
                    await write_detected_names_to_redis(detected_names_str)
                    self.draw_results(detecting_img)

                self.detecting = False
            except Exception as e:
                logging.error("error processing video frame: %s", e)
                self.detecting = False
                self.currrent_img = None
        logging.info("video detect worker done")

    # 从视频流中获取单帧图像,放入self.currrent_img(PIL.Image)
    async def video_track_worker(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        last_catch_time = 0
        frame_interval_normal = config.agents.yolo_frame_interval

        async for event in video_stream:
            if not self.video_streaming:
                logging.info("streaming stopped, closing video track worker")
                await video_stream.aclose()
            frame: rtc.VideoFrame = event.frame
            current_time = time.time()
            if (current_time - last_catch_time) < frame_interval_normal or self.detecting or config.agents.vision_lang_interval<=0:
                should_catch_for_detect = False
            else:
                should_catch_for_detect = True
            if self.capture_images_for_speak and (current_time - self.last_capture_time_for_speak) >= self.image_capture_interval_for_speak \
                    and len(self.image_list_for_speak) < self.max_images_for_speak:
                should_catch_for_speak = True
            else:
                should_catch_for_speak = False
            if current_time - self.last_scene_check_time < config.agents.scene_check_interval:
                should_catch_for_scene = False
            else:
                should_catch_for_scene = True
            if not should_catch_for_detect and not should_catch_for_speak and not should_catch_for_scene:
                continue
            logging.debug("received video frame: %d x %d, catch for detect: %s, catch for speak: %s", frame.width, frame.height, should_catch_for_detect, should_catch_for_speak)
            try:
                img = YoloV8Detector.VidioFrame_to_Image(frame)
                if not img:
                    continue
                if should_catch_for_detect:
                    last_catch_time = current_time
                    self.currrent_img = img
                if should_catch_for_speak:
                    self.image_list_for_speak.append(img)
                    self.last_capture_time_for_speak = current_time
                    if len(self.image_list_for_speak) >= self.max_images_for_speak:
                        logging.info("start sending images to server")
                        await self.send_images_to_server()

                # Pass the current frame to the scene monitor
                if should_catch_for_scene:
                    self.last_scene_check_time = current_time
                    if self.video_scene_monitor:
                        self.video_scene_monitor.set_current_frame(img)

            except Exception as e:
                logging.error("error catching video frame: %s", e)

        logging.info("video track worker done")
    
    def save_img(self, img: Image) -> str:
        # ./tmp/2022-03-16-15-30-45-123456.jpg
        try:
            os.makedirs("tmp", exist_ok=True)
            file_path = os.path.join("tmp", datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + ".jpg")
            img.save(file_path)
            # return absolute path for upload: file:///Users/boli/...
            absolute_path = "file://" + os.path.abspath(file_path)
            logging.info("saved img to %s, local path: %s", file_path, absolute_path)
            return absolute_path
        except Exception as e:
            logging.error("error saving img: %s", e)
            return None
    
    def draw_results(self, img):
        if self.show_detected_results and len(self.results) > 0:
            draw = ImageDraw.Draw(img)
            ids = self.results[0].boxes.cls
            for i in range(len(ids)):
                id = int(ids[i])
                name = self.detector.names[id]
                obj = self.results[0].boxes.xyxy[i]
                draw.rectangle([obj[0], obj[1], obj[2], obj[3]], outline='red')
                draw.text([obj[0], obj[1]], name, fill='red', font=font)
            self.im.set_data(img)
            plt.draw()
            plt.pause(0.001)

    async def send_images_to_server(self) -> str:
        if not self.image_list_for_speak:
            logging.warning("No images to send to server")
            return ""
        if time.time() - self.last_caption_time_for_speak < self.caption_interval_for_speak:
            logging.info("skip sending images to server, last caption time: %s, current time: %s", self.last_caption_time_for_speak, time.time())
            return

        # 将图像列表转换为base64编码的字符串列表
        image_contents = []
        for i, img in enumerate(self.image_list_for_speak):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

        if self.last_question_from_user and time.time() - self.last_question_from_user_time < self.caption_interval_for_speak:
            prompts = "请根据实时视频和说话内容，简要描述一下正在说话的人的特征、当前的场景。"
            prompts += f"他上次的问题是：{self.last_question_from_user}"
        else:
            prompts = "请根据实时视频简要的描述一下正在说话的人的特征和当前的场景。"
        # 添加文本内容
        image_contents.append({
            "type": "text",
            "text": prompts
        })

        # 创建 Message 对象
        message = Message(
            role=MessageRole.user,
            content=image_contents
        )
        response = ""
        try:
            # 发送消息到服务器并获取响应
            response = await send_message_to_server([message],model=config.llm.openai_custom_mm_model)
            logging.info(f"Server response: {response}, images sent: {len(self.image_list_for_speak)}, prompts: {prompts},\
                         model: {config.llm.openai_custom_mm_model}")
            self.last_caption_time_for_speak = time.time()
            self.current_caption_for_speak = response

        except Exception as e:
            logging.error(f"Error sending images to server: {e}")

        # 清空图像列表
        self.image_list_for_speak.clear()
        return response


    async def speakStatusChanged(self, event: agents.vad.VADEventType) -> None:
        logging.info("speech status changed: %s", event)
        if event == agents.vad.VADEventType.START_SPEAKING:
            self.capture_images_for_speak = True
            self.image_list_for_speak.clear()
        elif event == agents.vad.VADEventType.END_SPEAKING:
            self.capture_images_for_speak = False
            # await self.send_images_to_server()
        else:
            return
        cmd : str = None
        if event == agents.vad.VADEventType.START_SPEAKING:
            cmd = Command.START_SPEAK
        elif event == agents.vad.VADEventType.END_SPEAKING:
            cmd = Command.STOP_SPEAK
        await self.chat.send_move_cmd(
            cmd, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
        )
        logging.info("sent move(speak) command: %s", cmd)


    async def audio_track_worker(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        max_buffered_speech = config.agents.max_buffered_speech
        vad_stream = self.vad.stream(min_silence_duration=config.agents.min_silence_duration, min_speaking_duration=config.agents.min_speaking_duration, 
                                     max_buffered_speech=max_buffered_speech, threshold=config.agents.vad_threshold, newEventCallback=self.speakStatusChanged)
        stt = agents.stt.StreamAdapter(self.stt_model, vad_stream)
        stt_stream = stt.stream()
        self.ctx.create_task(self.stt_worker(stt_stream))

        logging.info("starting audio track worker")
        frame_count = 0
        async for event in audio_stream:
            if not self.audio_streaming:
                logging.info("streaming stopped, closing audio track worker")
                await audio_stream.aclose()
            frame = event.frame
            if frame.sample_rate != 16000:
                frame = frame.remix_and_resample(16000, 1)
            stt_stream.push_frame(frame)
            frame_count += 1
            if frame_count == 1:
                logger.info("audio track worker receive first frame")
        await stt_stream.flush()
        logging.info("audio track worker done")


    async def stt_worker(self, stt_stream: agents.stt.SpeechStream):
        async for event in stt_stream:
            # we only want to act when result is final
            if not event.is_final:
                logger.info("received interim result: %s", event.alternatives[0].text)
                continue
            if not event.alternatives[0].text:
                logging.info("received empty result, skipping")
                continue
            speech: agents.stt.SpeechData = event.alternatives[0]
            duration = speech.end_time - speech.start_time
            speaker_id = getattr(speech, 'speaker_id', None)
            logging.info(f"received speech from stt_stream: {speech.text}, duration: {duration}, speaker_id: {speaker_id}")
            cmd_str = LLMBase.remove_last_punctuation(speech.text).lower()
            if cmd_str ==  config.common.motion_mode_start_cmd:
                self.catch_follow_pos = True
                speech.text = speech.text.lower()
                logging.info("received start motion command, starting motion control")
            elif cmd_str == config.common.motion_mode_stop_cmd:
                try:
                    rt_msg = self.motion_control.stop_follow()
                    logging.info("received stop motion command, sending message to app: %s", rt_msg)
                    await self.chat.send_message(
                        message=speech.text, srcname=chat_ext.CHAT_MEMBER_APP, timestamp=speech.start_time,
                        duration=duration, language=speech.language
                    )
                    await self.chat.send_message(
                        message=rt_msg, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                    )
                except Exception as e:
                    logging.error("error stopping motion control: %s", e)
                self.catch_follow_pos = False
                continue

            if len(speech.text) > 5:
                self.last_question_from_user = speech.text
                self.last_question_from_user_time = speech.start_time
            if speaker_id is not None and speaker_id > 0:
                speech.text = f"[speaker_{speaker_id}] {speech.text}"
            if config.llm.openai_custom_mm_model and time.time() - self.last_caption_time_for_speak<self.caption_interval_for_speak and self.current_caption_for_speak:
                # asyncio.create_task(self.send_images_to_server())
                speech.text = f"[当前会话背景:{self.current_caption_for_speak}] {speech.text}"
                self.current_caption_for_speak = None

            await write_chat_to_redis(self.stream_key, text=speech.text, timestamp=speech.start_time, duration=duration, language=speech.language, srcname=chat_ext.CHAT_MEMBER_APP)
            await self.chat.send_message(
                message=speech.text, srcname=chat_ext.CHAT_MEMBER_APP, timestamp=speech.start_time,
                duration=duration, language=speech.language
            )

            await asyncio.sleep(0.001)
        await stt_stream.aclose()



if __name__ == "__main__":

    logger.info("starting live agent")

    async def job_request_cb(job_request: agents.JobRequest):
        logger.info("Accepting job for live")
        await job_request.accept(
            LiveAgent.create,
            identity="api_agent",
            name="api_agent",
            # subscribe to all audio tracks automatically
            auto_subscribe=agents.AutoSubscribe.SUBSCRIBE_ALL,
            # disconnect when the last participant leaves
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )
        logger.info("agent accepted job")

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
