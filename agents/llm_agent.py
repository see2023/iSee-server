import asyncio
import logging
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)
import redis
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, REDIS_PREFIX, REDIS_MQ_KEY
from agents.chat_ext import CHAT_MEMBER_APP, CHAT_MEMBER_ASSITANT,ChatExtManager, ChatExtMessage
from llm import get_llm, get_vl_LLM, LLMBase, VisualLLMBase
from llm.llm_base import KEEP_SILENT_RESPONSE, MessageRole
from llm.llm_tools import Tools, ToolNames, ToolActions
from typing import Callable
from livekit import rtc
from common.http_post import get_token
from common.tts import text_to_speech_and_visemes
from control.simple_control import Command
from agents.tts_manager import TTSManager
from agents.advanced_analysis import AdvancedAnalysis
from agents.llm_common import LLMMessage, MessageType, VISUAL_ANALYSIS_REQUEST

from common.config import config

logging.basicConfig(
    level=logging.DEBUG if config.agents.log_debug else logging.INFO,
    format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
)
logging.getLogger("websockets").setLevel(logging.ERROR)

quit_flag: bool = False

class PendingTextMessage():
    def __init__(self, text: str, callback: Callable):
        self.text = text

class MessageConsumer():
    def __init__(self, user: str = None, callback: Callable = None ):
        super().__init__()
        self._user = user
        self._user_sid = None
        self._callback = callback
        self._redis_client: redis.asyncio.Redis = get_redis_client()
        self._room: rtc.Room = rtc.Room()
        self._rtc_api_url = os.getenv("LIVEKIT_API_URL")
        self._rtc_url = os.getenv("LIVEKIT_URL")
        self._chat = None
        self._speeking = False
        self.llm_engine: LLMBase = get_llm()
        self.vlm: VisualLLMBase = get_vl_LLM()
        self._tts_msg_queue = asyncio.Queue[ChatExtMessage]()
        self._app_msg_queue = asyncio.Queue[str]()
        self.tts_manager = TTSManager()
        self._task_tts = None
        self.advanced_analysis = AdvancedAnalysis(
            self.llm_engine,
            send_to_app=self.send_to_app,
            trigger_visual_analysis=self.trigger_visual_analysis
        )
    
    def clear_tts_msg_queue(self):
        while not self._tts_msg_queue.empty():
            self._tts_msg_queue.get_nowait()
    
    def update(self, user: str = None, callback: Callable = None):
        if user:
            self._user = user
        if callback:
            self._callback = callback
    
    async def join_room(self) ->bool:
        if not self._user:
            self._user = await self._redis_client.get(REDIS_PREFIX+ "last_user")
            if self._user:
                logging.info("Got last user from redis: " + self._user)
        token = await get_token(os.getenv("API_USER"))
        logging.info("starting connection to livekit %s, got token from: %s", self._rtc_url, self._rtc_api_url)
        def is_app_user(user_id: str) -> bool:
            if not user_id:
                return False
            if user_id.find("api") >= 0 or user_id.find("agent")>=0:
                return False
            return True

        if token:
            try:

                # on_participant_connected and on_participant_disconnected not working, to be fixed
                def on_participant_connected(participant: rtc.RemoteParticipant):
                    logging.info("participant connected: %s %s", participant.sid, participant.identity)
                    if is_app_user(participant.identity):
                        self._user = participant.identity
                        self._user_sid = participant.sid
                        self.tts_manager.set_user_sid(self._user_sid)
                        logging.info("got user: " + self._user)
                    else:
                        logging.info("got api message: " + participant.identity)
                self._room.on("participant_connected", on_participant_connected)
                
                def on_participant_disconnected(participant: rtc.RemoteParticipant):
                    logging.info("participant disconnected: %s %s", participant.sid, participant.identity)
                    if participant.identity == self._user:
                        self._user = None
                        self._user_sid = None
                        logging.info("user left: " + participant.identity)
                self._room.on("participant_disconnected", on_participant_disconnected)

                # connect timeout 10s
                await self._room.connect(self._rtc_url, token)
                self._chat = ChatExtManager(self._room)
                self.tts_manager.set_chat_manager(self._chat)
                await asyncio.sleep(0.5)

                def process_chat(msg: ChatExtMessage):
                    logging.info("received chat message: %s", msg.message)
                    if msg.srcname == CHAT_MEMBER_APP and msg.message is not None and len(msg.message) > 0:
                        self._app_msg_queue.put_nowait(msg.message)
                        # self.llm_engine.interrupt(msg.message)

                def process_move_cmd(msg: ChatExtMessage):
                    if msg.srcname == CHAT_MEMBER_ASSITANT:  # voice from app, but vad from assistant
                        if msg.cmd == Command.START_SPEAK:
                            self._speeking = True
                            logging.info("Assistant start speaking")
                        elif msg.cmd == Command.STOP_SPEAK:
                            self._speeking = False
                            logging.info("Assistant stop speaking")

                self._chat.on("message_received", process_chat)
                self._chat.on("move_received", process_move_cmd)

                # find app user
                for participant in self._room.participants.values():
                    if is_app_user(participant.identity):
                        self._user = participant.identity
                        self._user_sid = participant.sid
                        self.tts_manager.set_user_sid(self._user_sid)
                        logging.info("Found app user: " + self._user)
                        break


            except Exception as e:
                logging.error(f"Failed to connect to livekit: {e}")
                return False
            logging.info("Connected to livekit")
            return True
        else:
            logging.error("Failed to get token")
            return False

    async def send_to_app(self, msg: str, message_type: MessageType = MessageType.TEXT, save_to_redis: bool = False):
        if not self._chat:
            logging.error("No chat manager found, cannot send message to app")
            return
        if msg == KEEP_SILENT_RESPONSE:
            return
        llm_message = LLMMessage(type=message_type, content=msg)
        if message_type == MessageType.TEXT:
            str_msg = msg
        else:
            str_msg = str(llm_message)
        chat_msg:ChatExtMessage = await self._chat.send_message(
            message=str_msg, srcname=CHAT_MEMBER_ASSITANT, timestamp=time.time(),
        )
        logging.info(f"sending message to app: {str_msg}, id: {chat_msg.id}")
        if message_type == MessageType.TEXT:
            await self.tts_manager.enqueue_message(chat_msg)
            if save_to_redis:
                await self.llm_engine.save_message_to_redis(self._user, str_msg, role=MessageRole.assistant)

    async def handle_action(self, cmd:str, args:list = None):
        if cmd == ToolActions.VLLM:
            logging.info("Received VLLM command, start checking picture...")
            await self.clear_picture_dir()
            await self.send_to_app(config.llm.vl_cmd_catch_pic)
            # 等待10秒，循环 check_new_picture
            finished = False
            for i in range(100):
                if await self.check_new_picture():
                    finished = True
                    break
                await asyncio.sleep(0.1)
            if not finished:
                logging.error("Failed to catch picture in 10 seconds, try again later")


    ## 订阅 redis stream: REDIS_CHAT_KEY，接收消息并处理
    async def run(self):
        global quit_flag
        rt = await self.join_room()
        if not rt:
            return
        
        self._task = asyncio.create_task(self.process_app_msg())
        self._task_tts = asyncio.create_task(self.tts_manager.run_tts_msg())
        while not quit_flag:
            await asyncio.sleep(1)
        self._task.cancel()
        self.tts_manager.stop()
        self._task_tts.cancel()
        await self._task
        await self._task_tts
        logging.info("Quit message consumer...")

    async def clear_picture_dir(self, dir_path: str = "tmp"):
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            for f in files:
                os.remove(os.path.join(dir_path, f))
                logging.info("Deleted old picture: " + f)

    async def check_new_picture(self, dir_path: str = "tmp") -> bool:
        if not os.path.exists(dir_path):
            return False
        # 找出最新的 .jpg 文件
        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".jpg")]
        if not files:
            return False
        latest_file = files[-1]
        # 删除其他文件
        for f in files:
            if f!= latest_file:
                os.remove(os.path.join(dir_path, f))
                logging.info("Deleted old picture: " + f)
        latest_file = os.path.join(dir_path, latest_file)
        absolute_path = "file://" + os.path.abspath(latest_file)
        logging.info("Found new picture: " + absolute_path)
        message = await self.vlm.call_mml_with_local_file(self._user, [absolute_path], model=config.llm.vl_model)
        if message:
            logging.info("received vision lang message: %s", message)
            if config.agents.send_vl_result:
                await self.send_to_app(message)
        else:
            logging.info("no vision lang message received")
            
        os.remove(latest_file)
        logging.info("Deleted picture file: " + latest_file)
        return True


    async def process_app_msg(self):
        logging.info("Start consuming app messages...")
        while not quit_flag:
            await asyncio.sleep(0.001)
            if not self._user:
                continue

            try:
                msg_text:str = await self._app_msg_queue.get()
                
                if self._speeking:
                    logging.info("Assistant is speaking, skip this message: " + msg_text)
                    continue
                if msg_text == config.common.motion_mode_start_cmd or msg_text == config.common.motion_mode_stop_cmd:
                    logging.info("Received motion mode command, skip this message:" + msg_text)
                    continue
                if len(msg_text) <= 2:
                    logging.info("Received short message, skip this message:" + msg_text)
                    continue
                logging.info("Received message: " + msg_text + ", start response...")


                if self.llm_engine.is_responsing():
                    self.llm_engine.interrupt(msg_text)
                    logging.info("LLM is responsing, skip this message: " + msg_text)
                    continue
                self.llm_engine.set_responsing_text(msg_text)
                self.llm_engine.clear_history()

                initial_response = ""
                try:
                    if self.llm_engine.stream_support():
                        got_response = False
                        async for output in self.llm_engine.generate_text_streamed(self._user, config.llm.model):
                            if output:
                                if not got_response:
                                    got_response = True
                                    self.tts_manager.clear_tts_msg_queue()
                                if config.llm.enable_custom_functions:
                                    output = self.llm_engine.parse_custom_function(output)
                                    if output is None or len(output) == 0:
                                        continue
                                initial_response += output
                                await self.send_to_app(output)
                                if self._callback:
                                    self._callback(output)
                    else:
                        initial_response = await self.llm_engine.generate_text(self._user, config.llm.model)
                        if initial_response:
                            self.tts_manager.clear_tts_msg_queue()
                            if config.llm.enable_custom_functions:
                                initial_response = self.llm_engine.parse_custom_function(initial_response)
                            await self.send_to_app(initial_response)
                            if self._callback:
                                self._callback(initial_response)

                    await self.llm_engine.save_message_to_redis(self._user, self.llm_engine.get_output())
                    if config.llm.enable_custom_functions:
                        await self.llm_engine.handle_custom_function_output(output_callback_func=self.send_to_app, cmd_callback_func=self.handle_action)
                    # 启动高级分析
                    asyncio.create_task(self.advanced_analysis.process_message(
                        self._user, msg_text, initial_response
                    ))
                except Exception as e:
                    logging.error(f"Error occurred while handling message: {e}")
            except asyncio.TimeoutError:
                logging.info("Timeout while consuming message, retry...")
            except Exception as e:
                logging.error(f"Error occurred while consuming message: {e}")

        logging.info("Quit consuming app messages...")

    async def trigger_visual_analysis(self, visual_analysis_prompt: str):
        # 发送视觉分析请求给live_agent
        await self.send_to_app(visual_analysis_prompt, MessageType.VISUAL_ANALYSIS_REQUEST)

    async def interrupt_response(self, new_response: str):
        # 中断当前响应并发送新的响应
        self.tts_manager.clear_tts_msg_queue()
        await self.send_to_app(new_response)


async def my_callback(output):
    print(f"Received output: {output}")

async def main():
    consumer = MessageConsumer()
    await consumer.run()

if __name__ == '__main__':
    import signal
    async def handle_quit(*args):
        global quit_flag
        logging.info("Quit signal received, exiting...")
        quit_flag = True
        await asyncio.sleep(3)
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda *args: asyncio.create_task(handle_quit(*args)))
    signal.signal(signal.SIGTERM, lambda *args: asyncio.create_task(handle_quit(*args)))

    asyncio.run(main())
