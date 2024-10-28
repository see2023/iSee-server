import requests
import base64
import json
import os
import time
import logging
from llm.llm_base import Message, MessageRole
from typing import List
import asyncio
from dotenv import load_dotenv
load_dotenv()
import openai
from common.config import config

# 读取图片文件并转换为base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 服务器地址
default_url = "http://192.168.123.46:8000/v1/chat/completions"

async def send_to_openai_api(messages: List[Message], model: str = "Qwen2-VL-2B-Instruct-AWQ", url: str = default_url, max_images: int = 2) -> str:
    url = url.replace("/chat/completions", "")
    image_count = 0
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get('type') == "image_url":
                    image_count += 1
    # 如果image_count超过max_images，则从前往后删除，直到满足条件
    if image_count > max_images:
        while image_count > max_images:
            for msg in messages:
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == "image_url":
                            msg.content.remove(item)
                            image_count -= 1
                            logging.debug(f"Remove image, current image_count: {image_count}")
                            break
                if image_count <= max_images:
                    break
    custom_url = config.llm.openai_custom_mm_url
    if custom_url:
        url = custom_url
    try:
        client = openai.AsyncOpenAI(api_key=os.getenv(config.llm.openai_custom_key_envname), base_url=url)
        response = await client.chat.completions.create(
            model=model,
            messages=[message.to_dict() for message in messages],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.info(f'send mm to {url} error: {e}')
        return ''



# 发送消息到服务器，返回响应Content
'''
    "model": "Qwen2-VL-2B-Instruct-AWQ",  qwen_api: qwen-vl-plus-0809 qwen-vl-max-0809
    "messages": []
'''
async def send_message_to_server(messages: List[Message], model: str = "Qwen2-VL-2B-Instruct-AWQ", use_openai_functions: bool = True, url: str = default_url) -> str:
    if use_openai_functions:
        return await send_to_openai_api(messages, model, url)
    # 准备请求数据
    data = {
        "model": model,
        "messages": [message.to_dict() for message in messages]
    }
    # logging.debug(f"Request data: {json.dumps(data, indent=2)}")

    # 记录开始时间
    start_time = time.time()

    # 发送POST请求
    headers = {"Content-Type": "application/json"}
    response = requests.post(default_url, headers=headers, data=json.dumps(data))

    # 计算耗时
    elapsed_time = time.time() - start_time

    # 打印响应和耗时
    status_code = response.status_code
    if status_code != 200:
        raise Exception(f"Failed to generate output: {status_code}, url: {default_url}")
    logging.debug(f"Status code: {status_code}, request time: {elapsed_time:.2f} seconds")

    content = response.json()['choices'][0]['message']['content']
    logging.debug("Content:")
    logging.debug(content)
    return content

async def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s')
    # 使用 os.path.expanduser() 展开 ~ 符号
    image_path3 = os.path.expanduser("~/Pictures/boy3.jpg")
    image_path2 = os.path.expanduser("~/Pictures/boy2.jpg")
    image_path1 = os.path.expanduser("~/Pictures/boy1.jpg")
    logging.info(f"Image path: {image_path3}")

    # 将图片转换为base64
    image_base64_3 = image_to_base64(image_path3)
    logging.debug(f"Image base64: {image_base64_3[:50]}...")  # 只打印前50个字符
    image_base64_2 = image_to_base64(image_path2)
    image_base64_1 = image_to_base64(image_path1)

    # 创建 Message 对象
    messages = [
        Message(
            role=MessageRole.system,
            content="你是一个儿童心理学家，请根据图片中的孩子回答问题。"
        ),
        Message(
            role=MessageRole.user,
            content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_3}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_2}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_1}"}},
            {"type": "text", "text": "图中的孩子们在干嘛？他们坐的端正吗？他们开心吗？"}
        ]
    )
    ]

    # 发送消息到服务器
    content = await send_message_to_server(messages, model="Qwen/Qwen2-VL-2B-Instruct-AWQ")
    logging.info(f"Content: {content}")

if __name__ == "__main__":
    asyncio.run(main())
