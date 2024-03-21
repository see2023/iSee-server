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
from typing import AsyncIterable, List, Optional, Callable
from common.tools import get_lunar
from common.config import config
from llm.llm_tools import Tools, ToolNames, ToolActions

from dotenv import load_dotenv
load_dotenv()


MessageRole = Enum("MessageRole", ["system", "user", "assistant", "function"])
@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.name, "content": self.content}
    
    def get_content_length(self) -> int:
        return len(self.content)


SYSTEM_PROMPTS = {
    "default": """你叫桔子，今年1岁，是一个聪明、友善的助手。你会在手机屏幕上以一个可爱的形象出现，手机绑在一个可移动小车上，可通过摄像头观察环境，并一直朝着目标用户移动，保持一定距离。
    用户用语音持续输入指令，你需要分析并理解用户的意图，给出简短、精准的回复，请言简意赅的说结论，最好不要超过30个字。
    请注意语音输入可能识别不准确，需要结合语境、上下文等来判断用户的意图，尤其要注意同音字误判的纠正。
    当前地点是:%s， 时间：%s。 """,
    "detect": "通过摄像头你现在可以看到这些目标：%s。",
    "tools": """你可以使用以下工具：%s。 如果需要使用工具， 请务必按顺序分3行返回且仅返回以下字段：
    Reason:首先返回需要使用工具的原因，比如：我需要先搜索一下. 
    Tool:Search
    Args:xx天气预报。
    如果不需要使用工具，请直接回复内容，不需要Reasons, Tool和Args。 
    """
}

class LLMBase:
    def __init__(self, app_secret_env_var_name: str=None, app_id_env_var_name: str=None, message_capacity:int=6000, history_count: int = 0, stream_support: bool = True):
        if history_count <= 0:
            history_count = config.llm.chat_history_count
        if app_secret_env_var_name:
            self._app_secret = os.getenv(app_secret_env_var_name)
        if app_id_env_var_name:
            self._app_id = os.getenv(app_id_env_var_name)
        self._redis_client: redis.Redis = get_redis_client()
        self._producing_response: bool = False
        self._function_working: bool = True
        self._needs_interrupt: bool = False
        self._message_capacity: int = int(message_capacity)
        self._complete_response: str = ""
        self._stream_timeout: float = 15
        self._response_timeout: float = 30
        self._responsing_txt: str = ""
        self._history_count: int = history_count
        self._tools = Tools()
        self._fn_name = ""
        self._fn_args = ""
        self._fn_output = ""
        self._user = ""
        self._model = ""
        self._history: List[dict] = []
        self._interactions_count = 0
        self._max_interactions = 3
        self._stream_support = stream_support
        self._custom_tool_prompt:str = None
    
    def stream_support(self) -> bool:
        return self._stream_support
    
    def set_custom_tool_prompt(self, prompt: str):
        self._custom_tool_prompt = prompt
    
    def get_output(self) -> str:
        return self._fn_output

    def clean_output(self):
        self._fn_name = ""
        self._fn_args = ""
        self._fn_output = ""

    def clear_history(self):
        self._history = []
        self._function_working = True
        self._interactions_count = 0
        self.clean_output()
    
    async def prepare(self, user: str, model: str, addtional_user_message: Message = None, use_redis_history: bool = True) -> bool:
        if self._interactions_count >= self._max_interactions:
            logging.info("Max interactions reached for user_id: %s", user)
            return False
        self.clean_output()
        if user and len(user) > 0:
            self._user = user
        if model and len(model) > 0:
            self._model = model
        if len(self._history) < 1:
            if use_redis_history:
                self._history = await self.build_history_from_redis(self._user)
                if self._history is None:
                    logging.info("No valid history found for user_id: %s", self._user)
                    return False
            else:
                self._history = await self.build_system_message()
        if addtional_user_message is not None:
            self._history.append(addtional_user_message.to_dict())
        if not self._history or len(self._history) == 0:
            logging.info("No history found for user_id: %s", self._user)
            return False
        return True
    
    def set_function_working(self, working: bool):
        self._function_working = working

    def add_history(self, message: Message):
        # check history length
        if len(self._history) >= self._history_count and len(self._history) > 3:
            logging.info("History length exceeds capacity, drop oldest message: %s", self._history[0]['content'])
            self._history = [self._history[0]] + self._history[3:]
        # check message length
        if message.get_content_length() + sum(map(lambda x: len(x['content']), self._history)) > self._message_capacity:
            logging.info("Message length exceeds capacity, drop message: %s", message.content)
            self._history = [self._history[0]] + self._history[3:]
        self._history.append(message.to_dict())

    def get_history(self) -> List[dict]:
        return self._history
    
    async def get_system_prompt(self, prompt_type=None) -> str:
        if prompt_type is None:
            if config.llm.enable_custom_functions and self._function_working:
                prompt_type = "tools"
            else:
                prompt_type = "default"
        detected_names = await get_detected_names_from_redis()
        default_prompt = SYSTEM_PROMPTS['default'] % (config.llm.location,  get_lunar())
        if detected_names and len(detected_names) >= 1:
            default_prompt += SYSTEM_PROMPTS["detect"] % (detected_names)
        if prompt_type == "default":
            return default_prompt
        if self._custom_tool_prompt is not None:
            if config.llm.enable_custom_functions and self._function_working:
                return default_prompt +self._custom_tool_prompt
            else:
                return default_prompt
        if prompt_type in SYSTEM_PROMPTS:
            add_prompt = SYSTEM_PROMPTS[prompt_type]
            if prompt_type == "tools" and config.llm.enable_custom_functions and self._function_working:
                add_prompt = add_prompt % (self._tools.get_functions_simple_style())
                return default_prompt + add_prompt
        return default_prompt 

    @classmethod
    async def anext_util(cls, aiter):
        async for item in aiter:
            return item
        return None

    def is_responsing(self) -> bool:
        return self._producing_response

    def interrupt(self, current_text: str):
        if self._producing_response and current_text != self._responsing_txt:
            logging.info("Interrupting llm response =============================")
            self._needs_interrupt = True
    
    def set_responsing_text(self, text: str):
        self._responsing_txt = text

    def get_complete_response(self) -> str:
        return self._complete_response
    
    @classmethod
    def last_sentence_end(cls, text: str, skip_comma: bool = True, min_length: int = 10) -> int:
        # 找出句子的最后一个逗号、句号或其他结束符, 要求至少10个字符，并且避免 2.3 这种数字
        if not text or len(text) < min_length:
            return -1
        total_length = len(text)
        end_chars = ['。', '！', '？', '；', "\n",
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

    async def build_history_from_redis(self, user_id: str) -> List[dict]:
        history: List[dict] = []
        system_message = Message(MessageRole.system, await self.get_system_prompt())
        total_length = system_message.get_content_length()
        stream_key = REDIS_CHAT_KEY+user_id
        start_time = int(time.time() - config.llm.chat_history_time_limit)
        messages = await self._redis_client.xrevrange(stream_key, min='-', max='+', count=self._history_count)
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
                    history[0]['content'] = msg_data['text'] + "\n" + history[0]['content']
                    continue
                last_msg_type = msg_data['srcname']
                first_msg_type = msg_data['srcname']
                history.insert(0, msg.to_dict())
                total_length += msg_length
        # 如果第一个消息的role 是 assistant，需要删除
        if history and len(history) > 1 and history[0]['role'] == MessageRole.assistant.name:
            history.pop(0)
        if history and len(history) > 1 and history[-1]['role'] == MessageRole.assistant.name:
            # 最后一个消息是assistant，已经回复过了，不需要再回复
            return None
        history.insert(0, system_message.to_dict())
        return history
    
    async def build_system_message(self) -> List[dict]:
        history: List[dict] = []
        system_message = Message(MessageRole.system, await self.get_system_prompt())
        history.append(system_message.to_dict())
        return history
    
    async def generate_text_streamed(self, user_id: str, model: str = "", addtional_user_message: Message = None, use_redis_history: bool = True) -> AsyncIterable[str]:
        pass

    async def generate_text(self, user_id: str, model: str = "", addtional_user_message: Message = None, use_redis_history: bool = True) ->str:
        pass

    async def save_message_to_redis(self, user_id: str, content: str, role: MessageRole = MessageRole.assistant):
        stream_key = REDIS_CHAT_KEY+user_id
        srcname = CHAT_MEMBER_APP if role == MessageRole.user else CHAT_MEMBER_ASSITANT
        # try:
        await write_chat_to_redis(stream_key, text=content, timestamp=time.time(), srcname=srcname)
        logging.debug(f"Saved message to redis: {content}")
        # except Exception as e:
        #     logging.error(f"Failed to save message to redis: {e}")  
    
    def extract_json_part(self, text: str) -> str:
        json_start = text.find("{")
        if json_start == -1:
            return ""
        json_end = text.rfind("}")
        if json_end == -1:
            return ""
        return text[json_start:json_end+1]

    def parse_custom_function(self, text: str) -> str:
        if not config.llm.enable_custom_functions:
            return text
        if config.llm.custom_functios_output_use_json:
            try:
                text_json = self.extract_json_part(text)
                data = json.loads(text_json)
                if "Tool" in data and "Args" in data:
                    self._fn_name = data["Tool"]
                    self._fn_args = data["Args"]
                if "Text" in data:
                    self._fn_output = data["Text"]
                    return data["Text"]
                else:
                    self._fn_output = text
                    logging.warning(f"Invalid custom function output json text: {text}")
                    return text
            except Exception as e:
                logging.error(f"Failed to parse custom function json: {e}")  
                return text
        text = text.strip()
        lines = text.split("\n")
        new_lines = []
        for line in lines:
            # parse: Tool: tool_name
            if line.startswith("Reason:"):
                # remove Reason:
                line = line[7:].strip()
                new_lines.append(line)
            elif line.startswith("Tool:"):
                tool_name = line.split(":")[1].strip()
                tool_name = ''.join(filter(str.isalnum, tool_name))
                if tool_name in ToolNames.all():
                    self._fn_name = tool_name
                else:
                    logging.warning(f"Invalid tool name: {tool_name}")
            elif line.find(ToolNames.TAKE_PHOTO)>-1:
                logging.info("hacking take photo")
                self._fn_name = ToolNames.TAKE_PHOTO
                return ''
            elif text.startswith("Args:"):
                self._fn_args = text[5:].strip()
                break
            elif self._fn_name  == "":
                new_lines.append(line)
            else:
                pass
        new_lines_str = "\n".join(new_lines)
        new_lines_str = new_lines_str.strip()
        self._fn_output += new_lines_str
        return new_lines_str

    async def handle_custom_function_output(self, output_callback_func: Callable[[str], None] = None, cmd_callback_func: Callable[[str], None] = None) -> str:
        self.set_function_working(False)
        logging.info(f"llm return function call: %s with args: %s", self._fn_name, self._fn_args)
        if self._fn_name is not None and len(self._fn_name) > 0:
            logging.debug(f"Calling custom function: {self._fn_name}")
            fn_result = await self._tools.call_tool(self._fn_name, self._fn_args)
            logging.info(f"Custom function result: {fn_result}")

            next_step = self._tools.get_next_step(self._fn_name)
            if next_step == ToolActions.LLM:
                system_message = Message(MessageRole.system, await self.get_system_prompt())
                self._history[0] = system_message.to_dict()
                self.add_history(Message(content='通过外部工具：' + self._fn_name + " 得知： " + fn_result, role=MessageRole.assistant))
                self.add_history(Message(content="现在请根据外部工具的结果回答我的问题", role=MessageRole.user))
                output_all = ''
                if self.stream_support():
                    async for output in self.generate_text_streamed(self._user,  model=self._model, use_redis_history=False):
                        logging.info(f"after custom function, llm stream output: {output}")
                        output_all += output
                else:
                    output_all = await self.generate_text(self._user,  model=self._model, use_redis_history=False)
                    logging.info(f"after custom function, llm output: {output_all}")
                if config.llm.enable_custom_functions:
                    # 仍然用工具解析一下，反正过拟合输出json格式
                    output_all = self.parse_custom_function(output_all)
                if output_callback_func:
                    await output_callback_func(output_all)
                await self.save_message_to_redis(self._user, output_all)
            elif next_step == ToolActions.VLLM:
                if cmd_callback_func:
                    await cmd_callback_func(ToolActions.VLLM)
                pass
            elif next_step == ToolActions.PASSTHROUGH:
                if output_callback_func:
                    await output_callback_func(fn_result)





class VisualLLMBase(LLMBase):
    def __init__(self, message_capacity:int=6000, history_count: int=20):
        super().__init__(message_capacity=message_capacity, history_count=history_count)
    
    async def get_user_history(self, user_id: str, max_count: int = 3) -> List[str]:
        try:
            self._system_prompt = await self.get_system_prompt()
            stream_key = REDIS_CHAT_KEY+user_id
            start_time = int(time.time() - 1800)
            messages = await self._redis_client.xrevrange(stream_key, min='-', max='+', count=self._history_count)
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
                    if len(questions) >= max_count:
                        break
            return questions
        except Exception as e:
            logging.error(f"Failed to get user history: {e}")  
            return []


    async def call_mml_with_local_file(self, user_id: str,  files: List[str], model: str = "qwen-vl-plus") -> str:
        pass
