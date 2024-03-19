import os,json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

org_path = "Qwen/Qwen1.5-1.8B-Chat"
model_root = "E:\dp\llm\qwen"
cache_root = "E:\dp\cache"
model_path = os.path.join(model_root, "output_qwen_1.8B_full")
quant_path = os.path.join(model_root, "output_qwen_1.8B_full_quant")

# Specify paths and hyperparameters for quantization
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_root)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True, cache_dir=cache_root)
print("load model from " + model_path)

data = []
jsonl_file =  os.path.join(model_root, "fine_tune_2.jsonl")
# unicode read
with open(jsonl_file, 'r', encoding='utf-8') as f:
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
