#     from livekit
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import asyncio
from dotenv import load_dotenv
load_dotenv()
import openai
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional
from llm.llm_base import LLMBase, Message, MessageRole
from common.config import config

class ChatGPT(LLMBase):
    """OpenAI ChatGPT Plugin"""

    def __init__(self, api_key: str, base_url: str, message_capacity: int = 6000):
        """
        Args:
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo-0125')
        """
        if not base_url: 
            base_url = None
        super().__init__("OPENAI_API_KEY", message_capacity=message_capacity)
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def generate_text(self, user_id: str, model: str = "", addtional_user_message: Message = None, use_redis_history: bool = True, strict_mode: bool = True, system_prompt: str = None) -> str:
        if not await self.prepare(user_id, model, addtional_user_message, use_redis_history, strict_mode, system_prompt):
            return ""
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=self._history,
                ),
                self._response_timeout,
            )
        except Exception as e:
            logging.error("Error in chatgpt: %s", e)
            return ""
        logging.debug("chatgpt got content: %s", response.choices[0].message.content)
        return response.choices[0].message.content


    async def generate_text_streamed(self, user_id: str, model: str = "gpt-4o-mini", addtional_user_message: Message = None, use_redis_history: bool = True, strict_mode: bool = True, system_prompt: str = None) -> AsyncIterable[str]:
        if not await self.prepare(user_id, model, addtional_user_message, use_redis_history, strict_mode, system_prompt):
            yield ""
            return
        try:
            chat_params = {
                "model": self._model,
                "n": 1,
                "stream": True,
                "messages": self._history,
            }
            if config.llm.enable_openai_functions:
                chat_params["functions"] = self._tools.get_functions_openai_style()
            
            chat_stream = await asyncio.wait_for(
                self._client.chat.completions.create(**chat_params),
                self._response_timeout,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        full_content: str = ""
        one_sentence: str = ''
        self._fn_name = ''
        self._fn_args = ''

        while True:
            try:
                chunk = await asyncio.wait_for(LLMBase.anext_util(chat_stream), self._stream_timeout)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break
            except Exception as e:
                logging.error("Error in chatgpt: %s", e)
                break

            if chunk is None:
                break
            delta = chunk.choices[0].delta
            content = ''
            if config.llm.enable_openai_functions and delta.content is None and delta.function_call is not None:
                if delta.function_call.name is not None:
                    content = delta.function_call.name
                    self._fn_name += content
                if delta.function_call.arguments is not None:
                    content += delta.function_call.arguments
                    self._fn_args += content
            else:
                content = delta.content
            # logging.debug("chatgpt got chunk: chunk.choices: %s", chunk.choices)

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT interrupted")
                break

            if content is not None:
                full_content += content
                one_sentence += content
                last_sentence_end_pos = LLMBase.last_sentence_end(one_sentence, config.llm.split_skip_comma, config.llm.split_min_length)
                if last_sentence_end_pos > 0:
                    sentence = one_sentence[:last_sentence_end_pos+1]
                    one_sentence = one_sentence[last_sentence_end_pos+1:]
                    logging.info("chatgpt got sentence: %s", sentence)
                    yield sentence

        if one_sentence != '':
            yield one_sentence
            logging.info("Last sentence from chatgpt: %s", one_sentence)

        logging.debug("chatgpt got full content: %s", full_content)
        if not config.llm.enable_custom_functions and not config.llm.enable_openai_functions:
            self._fn_output = full_content
        self._producing_response = False
        self._needs_interrupt = False
        self._interactions_count += 1

async def simple_test(model: str):
    client = openai.OpenAI(api_key=os.environ[config.llm.openai_custom_key_envname], base_url=config.llm.openai_custom_url)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "assistant",
                "content": "This is a test",
            },
            {
                "role": "user",
                "content": "Hello.",
            },
        ],
        model=model,
    )
    print(chat_completion)

async def main():
    chatgpt = ChatGPT(os.environ[config.llm.openai_custom_key_envname], config.llm.openai_custom_url)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    logging.info("Starting chatgpt")
    model = "gpt-4o"
    model = "gpt-4o-mini"
    model = config.llm.model
    query = "从2加到10等于几？"
    query = "请帮我写一首诗, 下雪了啊"
    query = "现在外面是几度？下雨了吗？"
    await simple_test(model)
    return
    streamed = True
    if not streamed:
        message = await chatgpt.generate_text("user1", model, addtional_user_message=Message(content=query, role=MessageRole.user), use_redis_history=False)
        print(message)
    else:
        async for message in chatgpt.generate_text_streamed("user1", model, addtional_user_message=Message(content=query, role=MessageRole.user), use_redis_history=False):
            print(message)

if __name__ == '__main__':
    asyncio.run(main())
