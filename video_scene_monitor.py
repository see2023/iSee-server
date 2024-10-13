import asyncio
import time
from typing import List
from PIL import Image
from io import BytesIO
import base64
from llm.qwen2_vl_http_server_cli import send_message_to_server
from llm.llm_base import Message, MessageRole, KEEP_SILENT_RESPONSE
import logging
from common.config import config

class VideoSceneMonitor:
    def __init__(self, interval, send_message_callback, send_to_server_callback, enable_self_reaction: bool = False, min_interval: float = 30):
        self.interval = interval
        self.min_interval = min_interval
        self.send_message_callback = send_message_callback
        self.send_to_server_callback = send_to_server_callback
        self.last_scene_description = ""
        self.last_check_time = 0
        self.monitoring = False
        self.imgs: List[Image] = [] # type: ignore
        self.max_imgs = 3
        self.has_new_img:bool = False
        self.last_sentences = []
        self.max_sentences = 6
        self.immediate_check_event = asyncio.Event()
        self.enalbe_self_reaction = enable_self_reaction
        self.last_add_question = ''
        self.last_response_str = ''

    def add_sentence(self, sentence: str, from_user: bool = True, max_len: int = 256):
        sentence = sentence[:max_len]
        if from_user:
            self.last_sentences.append(f"{sentence}")
        else:
            self.last_sentences.append(f"Assistant: {sentence}")
        if len(self.last_sentences) > self.max_sentences:
            self.last_sentences.pop(0)

    async def start_monitoring(self):
        self.monitoring = True
        logging.info("VideoSceneMonitor start monitoring")
        while self.monitoring:
            try:
                await asyncio.wait_for(self.immediate_check_event.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
            finally:
                self.immediate_check_event.clear()
                await self.check_scene()
        logging.info("VideoSceneMonitor monitoring stopped")

    def check_scene_immediately(self):
        if self.monitoring:
            self.immediate_check_event.set()

    def stop_monitoring(self):
        self.monitoring = False

    async def check_scene(self):
        current_time = time.time()
        if current_time - self.last_check_time < self.min_interval:
            logging.info(f"Scene check too frequent, skip, last check time: {self.last_check_time}, current time: {current_time}, min interval: {self.min_interval}")
            return
        self.last_check_time = current_time

        scene_description = await self.analyze_scene()
        if scene_description is None:
            logging.error("Scene description is None")
            return
        if KEEP_SILENT_RESPONSE in scene_description:
            logging.info("Silent response")
            return
        if scene_description == self.last_scene_description:
            logging.info("Same scene description")
            return
        self.last_scene_description = scene_description
        # 如果scene_description以talk:开头，则直接发送给用户
        if scene_description.startswith("talk:"):
            logging.info("Scene changed or anomaly detected, talk to user")
            await self.send_message_callback(f"{scene_description[5:]}", with_tts=True)
            return
        else:
            if scene_description.startswith("description:"):
                scene_description = scene_description[11:]
            logging.info("Scene changed or anomaly detected, describe to other assistant")
            await self.send_message_callback(f"[assistant看到场景的描述]:{{scene_description}}", with_tts=False, save_to_redis=True)

    async def analyze_scene(self, add_question: str = ''):
        if not self.has_new_img and add_question==self.last_add_question:
            return self.last_response_str
        if len(self.imgs) == 0:
            return ""
        self.last_add_question = add_question
        if self.enalbe_self_reaction:
            system_prompt = "假设你是一个智能助理(Assistant)，可以通过摄像头截图看到当前的场景，并用文字或语音的方式和用户、其他Assistant进行沟通。"
        else:
            system_prompt = "假设你是一个智能助理(Assistant)，可以通过摄像头截图看到当前的场景，请根据对话内容描述当前场景，以帮助其他Assistant理解当前的场景。"
        system_prompt += "用户可能有多个人，请以[user_n]来区分说话的人。"
        if self.enalbe_self_reaction:
            system_prompt += "首先考虑根据对话内容给其他Assistant描述场景，以帮助其他Assistant理解当前的场景。"
            system_prompt += "如果发现任何异常的、紧急的、有趣的、有疑问的情况，请用简短的口语告知或询问用户；否则请描述和对话相关的场景。"
            system_prompt += f"请注意回复格式必须为以下二种之一:\ntalk: 说给用户的话\ndescription: 说给其他Assistant的场景描述。"
        else:
            system_prompt += "请注意，你不需要直接回答问题，而是需要把你看到的和对话相关的内容描述给其他Assistant。"

        # system_prompt += "如果考虑后不应该打扰用户，或者不应该参与用户之间的对话，或者想说的内容与之前的雷同，仅回复:"+KEEP_SILENT_RESPONSE+"\n"

        system_message = Message(role=MessageRole.system, content=system_prompt)
        user_prompt = ""
        image_contents = [
        ]
        if len(self.last_sentences) > 0:
            for sentence in self.last_sentences:
                user_prompt += f"{sentence}\n"
        if add_question:
            user_prompt += f"{add_question}\n"
        for _, img in  enumerate(self.imgs):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        self.has_new_img = False
        if self.enalbe_self_reaction:
            user_prompt += "现在请以Assitance的身份，以talk或description开始回复："
        if user_prompt != "":
            image_contents.append({"type": "text", "text": user_prompt})
        logging.debug(f"Scene analysis start with system prompt: {system_prompt}, user prompt: {user_prompt}")
        user_message = Message(role=MessageRole.user, content=image_contents)
        response_str = await self.send_to_server_callback([system_message, user_message], model=config.llm.openai_custom_mm_model)
        logging.info(f"Scene analysis: {response_str}, message count: {len(image_contents)}")
        self.last_response_str = response_str
        return response_str

    def get_current_frame(self) -> Image:
        if len(self.imgs) == 0:
            return None
        return self.imgs[-1]
    
    def add_frame(self, img: Image):
        self.imgs.append(img)
        self.has_new_img = True
        if len(self.imgs) > self.max_imgs:
            self.imgs.pop(0)