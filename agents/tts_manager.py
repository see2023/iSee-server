import asyncio
import logging
from typing import Optional
from agents.chat_ext import ChatExtManager, ChatExtMessage
from common.tts import text_to_speech_and_visemes

class TTSManager:
    def __init__(self):
        self._tts_msg_queue = asyncio.Queue[ChatExtMessage]()
        self._chat: Optional[ChatExtManager] = None
        self._user_sid: Optional[str] = None
        self._quit_flag: bool = False

    def set_chat_manager(self, chat: ChatExtManager):
        self._chat = chat

    def set_user_sid(self, user_sid: str):
        self._user_sid = user_sid

    def clear_tts_msg_queue(self):
        while not self._tts_msg_queue.empty():
            self._tts_msg_queue.get_nowait()

    async def enqueue_message(self, chat_msg: ChatExtMessage):
        await self._tts_msg_queue.put(chat_msg)

    async def run_tts_msg(self):
        logging.info("start tts task")
        while not self._quit_flag:
            try:
                chat_msg: ChatExtMessage = await self._tts_msg_queue.get()
            except asyncio.CancelledError:
                break
            if not chat_msg.message or len(chat_msg.message) < 2:
                logging.info("empty message, skip")
                continue
            logging.info("tts got chat message: %s", chat_msg.message)
            res, visemes_fps = await text_to_speech_and_visemes(chat_msg.message)
            if not res:
                logging.error("Failed to get tts result")
                await asyncio.sleep(1)
                continue
            logging.info("got tts result, len: %0.2f", len(res['visemes'])/visemes_fps)
            if self._chat and self._user_sid:
                await self._chat.send_audio_message(res['audio'], res['visemes'], chat_msg.id, visemes_fps, self._user_sid)
            else:
                logging.error("Chat manager or user_sid not set, cannot send audio message")
        logging.info("tts task finished")

    def stop(self):
        self._quit_flag = True