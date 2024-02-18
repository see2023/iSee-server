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

class ChatGPT(LLMBase):
    """OpenAI ChatGPT Plugin"""

    def __init__(self, message_capacity: int = 6000):
        """
        Args:
            message_capacity (int): Maximum number of messages to send to the chat
            model (str): Which model to use (i.e. 'gpt-3.5-turbo-0125')
        """
        super().__init__("OPENAI_API_KEY", message_capacity=message_capacity)
        self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def generate_text_streamed(self, user_id: str, model: str = "gpt-3.5-turbo-0125", addtional_user_message: Message = None) -> AsyncIterable[str]:
        history = await self.build_history_from_redis(user_id)
        if addtional_user_message is not None:
            history.append(addtional_user_message.to_dict())
        if not history or len(history) == 0:
            logging.info("No history found for user_id: %s", user_id)
            yield ""
            return
        try:
            chat_stream = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    n=1,
                    stream=True,
                    messages=history,
                ),
                self._response_timeout,
            )
        except TimeoutError:
            yield "Sorry, I'm taking too long to respond. Please try again later."
            return

        self._producing_response = True
        full_content: str = ""
        one_sentence: str = ''

        while True:
            try:
                chunk = await asyncio.wait_for(LLMBase.anext_util(chat_stream), self._stream_timeout)
            except TimeoutError:
                break
            except asyncio.CancelledError:
                self._producing_response = False
                self._needs_interrupt = False
                break

            if chunk is None:
                break
            content = chunk.choices[0].delta.content

            if self._needs_interrupt:
                self._needs_interrupt = False
                logging.info("ChatGPT interrupted")
                break

            if content is not None:
                full_content += content
                one_sentence += content
                last_sentence_end_pos = LLMBase.last_sentence_end(one_sentence)
                if last_sentence_end_pos > 0:
                    sentence = one_sentence[:last_sentence_end_pos+1]
                    one_sentence = one_sentence[last_sentence_end_pos+1:]
                    logging.info("chatgpt got sentence: %s", sentence)
                    yield sentence

        if one_sentence != '':
            yield one_sentence
            logging.info("Last sentence from chatgpt: %s", one_sentence)

        logging.debug("chatgpt got full content: %s", full_content)
        await self.save_message_to_redis(user_id, full_content)
        self._producing_response = False
        self._needs_interrupt = False

async def main():
    chatgpt = ChatGPT()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    logging.info("Starting chatgpt")
    model = "gpt-3.5-turbo-0125"
    # model = 'gpt-4-0125-preview'
    async for message in chatgpt.generate_text_streamed("user1", model, Message(content="Hello, how are you?", role=MessageRole.user)):
        print(message)

if __name__ == '__main__':
    asyncio.run(main())
