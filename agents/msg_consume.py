import asyncio
import json
import logging
import os
import sys
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import redis
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, REDIS_PREFIX, REDIS_MQ_KEY
from agents.chat_ext import CHAT_MEMBER_APP, CHAT_MEMBER_ASSITANT,ChatExtManager
from llm import get_llm, get_vl_LLM, LLMBase, VisualLLMBase
from typing import Callable
from livekit import rtc
from common.http_post import get_token
from dotenv import load_dotenv
load_dotenv()

from common.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
)

quit_flag: bool = False

class MessageConsumer():
    def __init__(self, user: str = None, callback: Callable = None ):
        super().__init__()
        self._user = user
        self._callback = callback
        self._redis_client: redis.Redis = get_redis_client()
        self._room: rtc.Room = rtc.Room()
        self._rtc_api_url = os.getenv("LIVEKIT_API_URL")
        self._rtc_url = os.getenv("LIVEKIT_URL")
        self._chat = None
        if not self._user:
            self._user = self._redis_client.get(REDIS_PREFIX+ "last_user")
            if self._user:
                logging.info("Got last user from redis: " + self._user)
        self.llm_engine: LLMBase = get_llm()
        self.vlm: VisualLLMBase = get_vl_LLM()
    
    def update(self, user: str = None, callback: Callable = None):
        if user:
            self._user = user
        if callback:
            self._callback = callback
    
    async def join_room(self) ->bool:
        logging.info("starting connection to livekit...")
        token = get_token(os.getenv("API_USER"))
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
                        logging.info("got user: " + self._user)
                    else:
                        logging.info("got api message: " + participant.identity)
                self._room.on("participant_connected", on_participant_connected)
                
                def on_participant_disconnected(participant: rtc.RemoteParticipant):
                    logging.info("participant disconnected: %s %s", participant.sid, participant.identity)
                    if participant.identity == self._user:
                        self._user = None
                        logging.info("user left: " + participant.identity)
                self._room.on("participant_disconnected", on_participant_disconnected)

                await self._room.connect(self._rtc_url, token)
                self._chat = ChatExtManager(self._room)
                await asyncio.sleep(0.5)

                def process_chat(msg: rtc.ChatMessage):
                    logging.info("received chat message: %s", msg.message)
                self._chat.on("message_received", process_chat)

                # find app user
                for participant in self._room.participants.values():
                    if is_app_user(participant.identity):
                        self._user = participant.identity
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

    async def send_to_app(self, msg: str):
        logging.info("sending message to app: %s", msg)
        if not self._chat:
            logging.error("No chat manager found, cannot send message to app")
            return
        await self._chat.send_message(
            message=msg, srcname=CHAT_MEMBER_ASSITANT, timestamp=time.time(),
        )


    ## 订阅 redis stream: REDIS_CHAT_KEY，接收消息并处理
    async def run(self):
        rt = await self.join_room()
        if not rt:
            return
        
        self._task = asyncio.create_task(self.process_app_msg())
        while not quit_flag:
            await asyncio.sleep(1)
        await self._task

    async def check_new_picture(self, dir_path: str = "tmp"):
        if not os.path.exists(dir_path):
            return
        # 找出最新的 .jpg 文件
        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".jpg")]
        if not files:
            return
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
                await self._chat.send_message(
                    message=message, srcname=CHAT_MEMBER_ASSITANT, timestamp=time.time(),
                )
        else:
            logging.info("no vision lang message received")
            
        os.remove(latest_file)
        logging.info("Deleted picture file: " + latest_file)


    async def process_app_msg(self):
        logging.info("Start consuming messages...")
        redis_client = get_redis_client()
        pubsub = redis_client.pubsub()
        pubsub.subscribe(REDIS_MQ_KEY)
        while not quit_flag:
            await asyncio.sleep(0.001)
            if not self._user:
                continue

            # redis_stream_key = REDIS_CHAT_KEY + self._user
            # messages = redis_client.xread({redis_stream_key: "$"}, count=1, block=1000)
            # if messages and len(messages) > 0:
            #     msg = messages[0][1][0][1]
            #     if msg['srcname'] != CHAT_MEMBER_APP:
            #         continue
            #     msg_text = str(msg['text'])
            #     logging.info("Received message from app: " + msg_text + ", start response...")
            # xread loss message sometimes, change to pubsub
            try:
                message = pubsub.get_message(timeout=1)
                if message and message['type'] =='message':
                    msg_text = message['data']
                    logging.info("Received message from MQ: " + msg_text + ", start response...")
                    if msg_text == config.common.motion_mode_start_cmd or msg_text == config.common.motion_mode_stop_cmd:
                        logging.info("Received motion mode command, skip this message")
                        continue
                else:
                    await self.check_new_picture()
                    continue

                if self.llm_engine.is_responsing():
                    self.llm_engine.interrupt()
                    logging.info("LLM is responsing, skip this message: " + msg_text)
                    continue
                try:
                    async for output in self.llm_engine.generate_text_streamed(self._user, config.llm.model):
                        if output:
                            await self.send_to_app(output)
                            if self._callback:
                                self._callback(output)
                except Exception as e:
                    logging.error(f"Error occurred while handling message: {e}")
            except asyncio.TimeoutError:
                logging.info("Timeout while consuming message, retry...")
                pass
            except Exception as e:
                logging.error(f"Error occurred while consuming message: {e}")

        logging.info("Quit consuming messages...")



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
