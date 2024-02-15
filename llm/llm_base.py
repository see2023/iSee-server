import asyncio
import json
import logging
import os
import sys
import time
import redis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, write_chat_to_redis, get_detected_names_from_redis
from agents.chat_ext import CHAT_MEMBER_APP, CHAT_MEMBER_ASSITANT
from enum import Enum
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from common.tools import get_lunar
from common.config import config

from dotenv import load_dotenv
load_dotenv()

DEFAULT_SYSTEM_PROMPTS = {
    "app": """你叫桔子，今年1岁，是一个聪明、友善的助手。你会在手机屏幕上以一个可爱的形象出现，手机绑在一个可移动小车上，一直朝着目标用户移动，并保持一定距离。
    用户用语音持续输入指令，你需要分析并理解用户的意图，给出简短、精准的回复，请言简意赅的说结论，最好不要超过30个字。
    请注意语音输入可能识别不准确，需要结合语境、上下文等来判断用户的意图，尤其要注意同音字误判的纠正。
    通过摄像头你现在可以看到这些目标：%s。 当前地点是:%s， 时间：%s。 """,
}

MessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])
@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.name, "content": self.content}
    
    def get_content_length(self) -> int:
        return len(self.content)
    @classmethod
    def get_system_prompt(self, prompt_type="app") -> str:
        if prompt_type in DEFAULT_SYSTEM_PROMPTS:
            detected_names = get_detected_names_from_redis()
            return DEFAULT_SYSTEM_PROMPTS[prompt_type] % (detected_names, config.llm.location,  get_lunar())
        else:
            return 'You are a helpful assistant.'

class LLMBase:
    def __init__(self, app_secret_env_var_name: str, app_id_env_var_name: str=None, message_capacity:int=6000, history_count: int=20):
        self._app_secret = os.getenv(app_secret_env_var_name)
        if app_id_env_var_name:
            self._app_id = os.getenv(app_id_env_var_name)
        self._redis_client: redis.Redis = get_redis_client()
        self._producing_response: bool = False
        self._needs_interrupt: bool = False
        self._message_capacity: int = int(message_capacity)
        self._complete_response: str = ""
        self._stream_timeout: float = 15
        self._response_timeout: float = 30
        self._history_count: int = history_count

    @classmethod
    async def anext_util(self, aiter):
        async for item in aiter:
            return item
        return None

    def is_responsing(self) -> bool:
        return self._producing_response

    def interrupt(self):
        if self._producing_response:
            self._needs_interrupt = True

    def get_complete_response(self) -> str:
        return self._complete_response
    
    @classmethod
    def last_sentence_end(cls, text: str, skip_comma: bool = True, min_length: int = 10) -> int:
        # 找出句子的最后一个逗号、句号或其他结束符, 要求至少10个字符，并且避免 2.3 这种数字
        if not text or len(text) < min_length:
            return -1
        total_length = len(text)
        end_chars = ['。', '！', '？', '；', 
                     '…', '.', '!', '?', ';']
        # 从后往前找
        for i in range(total_length-1, -1, -1):
            if i<min_length:
                break
            if text[i] in end_chars:
                if i<total_length-1 and text[i] == '.' and text[i-1].isdigit() and text[i+1].isdigit():
                    continue
                return i
        if not skip_comma:
            end_chars = ['，', ',']
            for i in range(total_length-1, -1, -1):
                if i<min_length:
                    return -1
                if text[i] in end_chars:
                    return i
        return -1
    
    @classmethod # 去除最后一个标点符号
    def remove_last_punctuation(cls, text: str) -> str:
        last_punctuation_index = cls.last_sentence_end(text, skip_comma=False, min_length=2)
        if last_punctuation_index == -1:
            return text
        return text[:last_punctuation_index]

    async def build_history_from_redis(self, user_id: str) -> List[Message]:
        history: List[dict] = []
        system_message = Message(MessageRole.system, Message.get_system_prompt())
        total_length = system_message.get_content_length()
        stream_key = REDIS_CHAT_KEY+user_id
        start_time = int(time.time() - 1800)
        messages = self._redis_client.xrevrange(stream_key, min='-', max='+', count=self._history_count)
        if messages:
            last_msg_type = ''
            first_msg_type = ''
            for message in messages:
                msg_data = message[1]
                #{"text": speech.text, "timestamp": speech.start_time, 
                #   "duration": duration, "language": speech.language, "srcname": chat_ext.CHAT_MEMBER_APP})
                if msg_data['srcname'] and msg_data['text'] and msg_data['timestamp']:
                    msg = Message(content=msg_data['text'], role=MessageRole.assistant)
                else:
                    continue
                if first_msg_type == '' and msg_data['srcname'] == CHAT_MEMBER_ASSITANT:
                    logging.debug(f"Skip first assistant message: {msg_data['text']}")
                    continue
                msg_time = int(msg_data['timestamp'])/1000
                if msg_time < start_time:
                    break
                msg_length = msg.get_content_length()
                if total_length + msg_length > self._message_capacity:
                    break
                if msg_data['srcname'] == CHAT_MEMBER_APP:
                    msg.role = MessageRole.user
                elif msg_data['srcname'] == CHAT_MEMBER_ASSITANT:
                    msg.role = MessageRole.assistant
                else:
                    continue
                if msg_data['srcname'] == last_msg_type:
                    logging.debug(f"combine duplicated message: {msg_data['text']}")
                    history[0]['content'] = msg_data['text'] + history[0]['content']
                    continue
                last_msg_type = msg_data['srcname']
                first_msg_type = msg_data['srcname']
                history.insert(0, msg.to_dict())
                total_length += msg_length
        # 如果第一个和最后一个的 role 都是 assistant，都需要删除
                
        if history and len(history) > 1 and history[0]['role'] == MessageRole.assistant.name:
            history.pop(0)
        if history and len(history) > 1 and history[-1]['role'] == MessageRole.assistant.name:
            history.pop(-1)
        if not history or len(history) == 0:
            return []
        history.insert(0, system_message.to_dict())
        return history
    async def generate_text_streamed(self, user_id: str, model: str = "", addtional_user_message: Message = None) -> AsyncIterable[str]:
        pass

    async def save_message_to_redis(self, user_id: str, content: str, role: MessageRole = MessageRole.assistant):
        stream_key = REDIS_CHAT_KEY+user_id
        srcname = CHAT_MEMBER_APP if role == MessageRole.user else CHAT_MEMBER_ASSITANT
        # try:
        write_chat_to_redis(stream_key, text=content, timestamp=time.time(), srcname=srcname)
        logging.debug(f"Saved message to redis: {content}")
        # except Exception as e:
        #     logging.error(f"Failed to save message to redis: {e}")  


class VisualLLMBase:
    def __init__(self, message_capacity:int=6000, history_count: int=20):
        self._redis_client: redis.Redis = get_redis_client()
        self._message_capacity: int = int(message_capacity)
        self._history_count: int = history_count
        self._system_prompt = Message.get_system_prompt()
    
    async def get_user_history(self, user_id: str) -> List[str]:
        try:
            stream_key = REDIS_CHAT_KEY+user_id
            start_time = int(time.time() - 1800)
            messages = self._redis_client.xrevrange(stream_key, min='-', max='+', count=self._history_count)
            questions: List[str] = []
            if messages:
                for message in messages:
                    msg_data = message[1]
                    #{"text": speech.text, "timestamp": speech.start_time, 
                    #   "duration": duration, "language": speech.language, "srcname": chat_ext.CHAT_MEMBER_APP})
                    if not msg_data['srcname'] or not msg_data['text'] or not msg_data['timestamp']:
                        continue
                    if msg_data['srcname'] != CHAT_MEMBER_APP:
                        continue
                    msg_time = int(msg_data['timestamp'])/1000
                    if msg_time < start_time:
                        break
                    questions.append(msg_data['text'])
            return questions
        except Exception as e:
            logging.error(f"Failed to get user history: {e}")  
            return []


    async def call_mml_with_local_file(self, user_id: str,  files: List[str], model: str = "qwen-vl-plus") -> str:
        pass
