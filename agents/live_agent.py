import asyncio
import json
import logging
import os
import numpy as np
import fast_whisper_stt
import xf_stt
import chat_ext
import sys
import time
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, REDIS_PREFIX, write_chat_to_redis, write_mq_to_redis, write_detected_names_to_redis

from dotenv import load_dotenv
load_dotenv()
from common.config import config

from livekit import agents, rtc
from livekit.plugins import openai 
from detect.yolov8 import YoloV8Detector
from vad import VAD
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image
import matplotlib
from llm.llm_base import LLMBase, VisualLLMBase
from control.simple_control import SimpleControl, Command
font_path = "/Library/Fonts/Arial Unicode.ttf" if sys.platform == "darwin" else "arial.ttf"
font_size = 40
font = ImageFont.truetype(font_path, font_size)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
)
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
            self.stt_model = xf_stt.STT()
        else:
            raise Exception("invalid whisper type: {}".format(stt_type))
        self.detector = YoloV8Detector()


def get_model():
    model_manager = ModelManager.get_instance()
    return model_manager.stt_model, model_manager.detector

class LiveAgent:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        agent = LiveAgent(ctx)
        await agent.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins

        self.stt_model, self.detector = get_model() 
        self.catch_follow_pos = False
        self.motion_control = SimpleControl(self.detector)

        self.vad = VAD()
        logging.info("Live agent created")

        self.ctx = ctx
        self.chat = chat_ext.ChatExtManager(ctx.room)
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

        self.show_detected_results = config.agents.show_detected_results
        if self.show_detected_results and config.agents.enable_video:
            plt.ion()
            self.fig = plt.figure("Video")
            self.im =   plt.imshow(np.zeros((10, 10, 3)))

        # setup callbacks
        def subscribe_cb(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                if config.agents.enable_audio:
                    self.ctx.create_task(self.audio_track_worker(track))
                if not self.remoteId:
                    self.remoteId = participant.identity
                    async def set_user_info(user_id: str):
                        self.stream_key = REDIS_CHAT_KEY + user_id
                        logging.info("================================== redis xtrim key: %s", self.stream_key)
                        await self.redis.xtrim(self.stream_key, 10000)
                        # set last user
                        await self.redis.set(REDIS_PREFIX+ "last_user", user_id)
                    loop = asyncio.get_event_loop()
                    loop.create_task(set_user_info(self.remoteId))
            elif track.kind == rtc.TrackKind.KIND_VIDEO:
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

        self.ctx.room.on("track_subscribed", subscribe_cb)
        self.ctx.room.on("track_unsubscribed", lambda *args: logging.info("unsubscribed")) # todo, handle member disconnect
        def on_message(msg: chat_ext.ChatExtMessage):
            asyncio.ensure_future(process_chat(msg))
        self.chat.on("message_received", on_message)

    async def send_to_app(self, msg: str):
        logging.info("sending message to app: %s", msg)
        await self.chat.send_message(
            message=msg, srcname=chat_ext.CHAT_MEMBER_ASSITANT, timestamp=time.time(),
        )

    async def start(self):
        # give a bit of time for the user to fully connect so they don't miss
        # the welcome message
        await asyncio.sleep(1)

        # create_task is used to run coroutines in the background
        self.ctx.create_task(
            self.chat.send_message(
                message="Welcome to the live agent! Just peak to me."
            )
        )

        self.update_agent_state("listening")

    def update_agent_state(self, state: str):
        metadata = json.dumps(
            {
                "agent_state": state,
            }
        )
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))

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

    async def video_track_worker(self, track: rtc.VideoTrack):
        video_stream = rtc.VideoStream(track)
        last_catch_time = 0
        frame_interval_normal = config.agents.yolo_frame_interval
        self.video_streaming = True

        async for event in video_stream:
            frame: rtc.VideoFrame = event.frame
            current_time = time.time()
            if (current_time - last_catch_time) < frame_interval_normal or self.detecting:
                logging.debug("video frame skipped ______ ")
                continue
            last_catch_time = current_time
            logging.debug("received video frame: %d x %d", frame.width, frame.height)
            try:
                img = YoloV8Detector.VidioFrame_to_Image(frame)
                if not img:
                    continue
                self.currrent_img = img
            except Exception as e:
                logging.error("error catching video frame: %s", e)

        self.video_streaming = False

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


    async def speakStatusChanged(self, event: agents.vad.VADEventType) -> None:
        logging.info("speech status changed: %s", event)
        cmd : str = None
        if event == agents.vad.VADEventType.START_SPEAKING:
            cmd = Command.START_SPEAK
        elif event == agents.vad.VADEventType.END_SPEAKING:
            cmd = Command.STOP_SPEAK
        else:
            return
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
            frame = event.frame
            if frame.sample_rate != 16000:
                frame = frame.remix_and_resample(16000, 1)
            stt_stream.push_frame(frame)
            frame_count += 1
            if frame_count % 500 == 0:
                logger.info("audio track worker received frame")
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
            logging.info("received speech from stt_stream: %s, duration: %.3fs", speech.text, duration)
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

            await self.chat.send_message(
                message=speech.text, srcname=chat_ext.CHAT_MEMBER_APP, timestamp=speech.start_time,
                duration=duration, language=speech.language
            )
            await write_chat_to_redis(self.stream_key, text=speech.text, timestamp=speech.start_time, duration=duration, language=speech.language, srcname=chat_ext.CHAT_MEMBER_APP)
            await write_mq_to_redis(speech.text)

            await asyncio.sleep(0.001)
        await stt_stream.aclose()



if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)

    logger.info("starting live agent")
    logger.info('_____________ %s', Command.ACCELERATE)

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
