from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device name: {torch.cuda.get_device_name(0)}")

http_proxy = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = http_proxy
os.environ["HTTPS_PROXY"] = http_proxy

# default: Load the model on the available device(s), cache in E:\dp\cache
cache_dir = "E:\\dp\\cache" if os.path.exists("E:\\dp\\cache") else None
model_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
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

test_files = [
    "E:\\dp\\pics\\boy1.jpg",
    "E:\\dp\\pics\\boy2.jpg",
    "E:\\dp\\pics\\boy3.jpg",
    "E:\\dp\\pics\\en_q.jpg",
    "E:\\dp\\pics\\grade6_720p_30fps.mp4",
]



messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": test_files[0],
            },
            {"type": "text", "text": "这个孩子戴着眼镜吗？"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

print("start inference")
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)