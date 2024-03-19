from transformers import AutoModelForCausalLM, AutoTokenizer
import time,os
from data import system_prompt
device = "cuda" # the device to load the model onto
model_root = "E:\dp\llm\qwen"
cache_root = "E:\dp\cache"
model_path = os.path.join(model_root, "output_qwen_1.8B_full")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    cache_dir=cache_root
)
model = model.to(device)
# print model torch_dtype and model_path
print(model.config.torch_dtype)
print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


prompt = "我喜欢吃烤鱼，不辣的，怎么翻译成英语？"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

start_time = time.time()
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
end_time = time.time()
print("--------------- time used: " + str(end_time - start_time))
print(response)