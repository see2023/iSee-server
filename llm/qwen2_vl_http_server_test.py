import requests
import base64
import json
import os
import time

'''
curl http://192.168.123.46:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen2-VL-2B-Instruct-AWQ",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'$(base64 -w 0 /Users/boli/Pictures/boy2.jpg)'"}},
            {"type": "text", "text": "Describe the content of the image."}
        ]}
    ]
}'
'''

# 读取图片文件并转换为base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 服务器地址
url = "http://192.168.123.46:8000/v1/chat/completions"

# 使用 os.path.expanduser() 展开 ~ 符号
image_path = os.path.expanduser("~/Pictures/boy2.jpg")
print(image_path)

# 将图片转换为base64
image_base64 = image_to_base64(image_path)

# 准备请求数据
data = {
    "model": "Qwen2-VL-2B-Instruct-AWQ",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": "请简要描述图片内容."}
        ]}
    ]
}

# 记录开始时间
start_time = time.time()

# 发送POST请求
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(data))

# 计算耗时
elapsed_time = time.time() - start_time

# 打印响应和耗时
print(f"Status code: {response.status_code}")
print(f"Request time: {elapsed_time:.2f} seconds")
print(json.dumps(response.json(), indent=2))

# 直接打印中文内容，不进行额外的解码
content = response.json()['choices'][0]['message']['content']
print("content:")
print(content)

