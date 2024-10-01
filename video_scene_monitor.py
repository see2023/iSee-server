import asyncio
import time
from PIL import Image
from io import BytesIO
import base64
from llm.qwen2_vl_http_server_cli import send_message_to_server
from llm.llm_base import Message, MessageRole, KEEP_SILENT_RESPONSE
import logging
from common.config import config
class VideoSceneMonitor:
    def __init__(self, interval, send_message_callback, send_to_server_callback):
        self.interval = interval
        self.send_message_callback = send_message_callback
        self.send_to_server_callback = send_to_server_callback
        self.last_scene_description = ""
        self.last_check_time = 0
        self.monitoring = False
        self.cur_img: Image = None
        self.last_image_base64 = ""
        self.keywords = ["变化", "异常", "紧急", "危险", "警告"]

    async def start_monitoring(self):
        self.monitoring = True
        logging.info("VideoSceneMonitor start monitoring")
        while self.monitoring:
            await asyncio.sleep(self.interval)
            await self.check_scene()
        logging.info("VideoSceneMonitor monitoring stopped")

    async def stop_monitoring(self):
        self.monitoring = False

    async def check_scene(self):
        current_time = time.time()
        if current_time - self.last_check_time < self.interval:
            return
        self.last_check_time = current_time

        current_frame = self.get_current_frame()
        if current_frame is None:
            return

        scene_description = await self.analyze_scene(current_frame)
        
        if self.should_notify(scene_description):
            logging.info("Scene changed or anomaly detected")
            await self.send_message_callback(f"{scene_description}")
            self.last_scene_description = scene_description

    async def analyze_scene(self, img: Image):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        if self.last_image_base64 == img_base64:
            return KEEP_SILENT_RESPONSE
        prompt = "假设你是一个智能助理，通过摄像头截图看到当前的场景。 如果发现任何异常的、紧急的情况，或者发现有趣的事情需要分享看法，或者是需要询问场景里的人物，请用简短的口语化语句回复。"
        image_contents = [
        ]
        if self.last_image_base64 != "":
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.last_image_base64}"}})
            prompt += "如果发现两张图片里的人物有明显变化，可以主动询问对方，给出合理的口语表达的响应。"
        prompt += "如果场景合适，你也可以开个玩笑，或者简单问候一下。"
        prompt += "否则请保持沉默，回复:"+KEEP_SILENT_RESPONSE
        image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        image_contents.append({"type": "text", "text": prompt})
        self.last_image_base64 = img_base64
        message = Message(role=MessageRole.user, content=image_contents)
        response = await self.send_to_server_callback([message], model=config.llm.openai_custom_mm_model)
        logging.info(f"Scene analysis: {response}, message count: {len(image_contents)}")
        return response

    def should_notify(self, current_description: str) -> bool:
        if KEEP_SILENT_RESPONSE in current_description or not current_description or current_description==self.last_scene_description:
            return False
        else:
            return True
        # 现在我们简单地检查描述是否包含关键词
        # return any(keyword in current_description for keyword in self.keywords)

    def get_current_frame(self) -> Image:
        return self.cur_img
    
    def set_current_frame(self, img: Image):
        self.cur_img = img
