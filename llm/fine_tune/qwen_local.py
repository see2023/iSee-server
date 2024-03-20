from transformers import AutoModelForCausalLM, AutoTokenizer
import time,os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.fine_tune.data import system_prompt, prompt_common
from llm.llm_base import LLMBase, Message
import logging
import asyncio
from common.config import config

class QwenLocal(LLMBase):
    def __init__(self, model_path, cache_root, device="cuda"):
        super().__init__(stream_support=False)
        self.model_path = model_path
        self.cache_root = cache_root
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            cache_dir=cache_root
        )
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("model loaded from %s", model_path)
        logging.info("model.config.torch_dtype: %s ", self.model.config.torch_dtype)
        self.set_custom_tool_prompt(prompt_common + "\n现在请按照上述格式简要回答用户的问题:")
    
    def generate(self, messages):
        logging.info("generate response")
        logging.debug("messages: %s", messages)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        start_time = time.time()
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.time()
        logging.info("generate response time: %f, response: %s", end_time - start_time, response)
        return response


    async def generate_async(self, messages):
        return await asyncio.get_event_loop().run_in_executor(None, self.generate, messages)

    async def generate_text(self, user_id: str, model: str = "", addtional_user_message: Message = None, use_redis_history: bool = True) ->str:
        if config.llm.enable_openai_functions:
            logging.error("OpenAI functions are not supported in QwenLocal")
            return ""
        if not await self.prepare(user_id, model, addtional_user_message, use_redis_history):
            return ""
        full_content: str = ''
        self._producing_response = True
        full_content = await self.generate_async(self._history)
        if use_redis_history:
            await self.save_message_to_redis(self._user, full_content)
        self._needs_interrupt = False
        self._producing_response = False
        self._interactions_count += 1
        return full_content



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    # FastAPI app to handle gennerate requests
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    device = "cuda" # the device to load the model onto
    model_root = "E:\dp\llm\qwen"
    cache_root = "E:\dp\cache"
    model_path = os.path.join(model_root, "output_qwen_1.8B_full")


    prompt = "我喜欢吃烤鱼，不辣的，怎么翻译成英语？"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    app = FastAPI()
    qwen_local = QwenLocal(model_path=model_path, cache_root=cache_root, device=device)

    @app.get("/generate")
    async def generate(text: str = '我想去骑车，但是数学作业还没写完，怎么办？'):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        response = await qwen_local.generate_async(messages)
        logging.info("response: %s", response)
        return JSONResponse({"response": response})
    
    uvicorn.run(app, host="0.0.0.0", port=8010)
