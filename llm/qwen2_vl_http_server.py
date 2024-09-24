from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
import requests
from io import BytesIO
import base64
from PIL import Image, UnidentifiedImageError
import logging
import time

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s')

app = Flask(__name__)

http_proxy = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = http_proxy
os.environ["HTTPS_PROXY"] = http_proxy

# default: Load the model on the available device(s), cache in E:\dp\cache
cache_dir = "E:\\dp\\cache" if os.path.exists("E:\\dp\\cache") else None
model_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True, cache_dir=cache_dir
)

# default processer
processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
'''
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "Qwen2-VL-2B-Instruct-AWQ",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "E:\\dp\\pics\\boy1.jpg"}},
            {"type": "text", "text": "Describe the content of the image."}
        ]}
    ]
}'
'''

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    start_time = time.time()
    data = request.json
    messages = data['messages']
    
    # 处理图片URL
    for message in messages:
        if 'content' in message and isinstance(message['content'], list):
            for item in message['content']:
                if item['type'] == 'image_url':
                    url = item['image_url']['url']
                    try:
                        if url.startswith('http://') or url.startswith('https://'):
                            # 处理HTTP链接
                            response = requests.get(url)
                            response.raise_for_status()
                            image = Image.open(BytesIO(response.content))
                        elif url.startswith('data:image'):
                            # 处理Base64编码的内容
                            header, base64_data = url.split(',', 1)
                            image = Image.open(BytesIO(base64.b64decode(base64_data)))
                        else:
                            # 处理本地路径
                            image = Image.open(url)
                        item['image'] = image  # 替换URL为图片数据
                    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as e:
                        logging.error(f"Failed to process image: {str(e)}")
                        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # 准备推理
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 推理：生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    except Exception as e:
        logging.error(f"Failed to generate output: {str(e)}")
        return jsonify({"error": f"Failed to generate output: {str(e)}"}), 500

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Request processed in {processing_time:.2f} seconds")

    return jsonify({
        "model": data['model'],
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": output_text[0]
                }
            }
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)