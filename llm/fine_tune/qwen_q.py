model_path = "/content/drive/MyDrive/github/llm/Qwen1.5/examples/sft/output_qwen_1.8B"
new_path = "/content/drive/MyDrive/github/llm/Qwen1.5/examples/sft/output_qwen_1.8B_full"
quant_path = "/content/drive/MyDrive/github/llm/Qwen1.5/examples/sft/output_qwen_1_8_awq"
org_path = "Qwen/Qwen1.5-1.8B-Chat"

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(new_path)
model = AutoAWQForCausalLM.from_pretrained(new_path, device_map="auto", safetensors=True)
print("load model from " + new_path)

import json
data = []
jsonl_file =  '/content/drive/MyDrive/github/llm/fine_tune_2.jsonl'
with open(jsonl_file, 'r') as f:
    messages = [json.loads(line) for line in f]
for msg in messages:
    msgs = msg['messages']
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())

model.quantize(tokenizer, quant_config=quant_config, calib_data=data)


model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)
from transformers import AutoConfig
config = model.config
config.save_pretrained(quant_path)
