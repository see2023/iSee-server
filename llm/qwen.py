from typing import AsyncIterable, List, Optional
from http import HTTPStatus
import logging
import asyncio
import sys
import os
import time
from agents.chat_ext import CHAT_MEMBER_APP, CHAT_MEMBER_ASSITANT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from dashscope import Generation, MultiModalConversation
from llm.llm_base import LLMBase, Message, MessageRole,VisualLLMBase
from db.redis_cli import REDIS_CHAT_KEY, write_chat_to_redis
from common.config import config



class Qwen(LLMBase):
    def __init__(self):
        super().__init__("DASHSCOPE_API_KEY")

    async def generate_text_streamed(self, user_id: str, model: str = Generation.Models.qwen_max, addtional_user_message: Message = None, use_redis_history: bool = True) -> AsyncIterable[str]:
        if config.llm.enable_openai_functions:
            yield "Sorry, OpenAI functions are not supported for Qwen"
            logging.error("OpenAI functions are not supported for Qwen")
            return
        logging.debug("Start qwen generation for user_id: %s", user_id)
        if not await self.prepare(user_id, model, addtional_user_message, use_redis_history):
            yield ""
            return
        logging.debug("Qwen history: %s", self._history)
        responses = Generation.call(
            self._model,
            messages=self._history,
            result_format='message',  # set the result to be "message" format.
            stream=True,
            incremental_output=True  # get streaming output incrementally
        )
        full_content: str = ''  # with incrementally we need to merge output.
        one_sentence: str = ''
        self._producing_response = True

        # while True:
        for response in responses:
            try:
                # response = await asyncio.wait_for(LLMBase.anext_util(responses), self._stream_timeout)
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0]['message']['content']
                    if content is not None:
                        full_content += content
                        one_sentence += content
                        last_sentence_end_pos = LLMBase.last_sentence_end(one_sentence,  config.llm.split_skip_comma, config.llm.split_min_length)
                        if last_sentence_end_pos > 0:
                            sentence = one_sentence[:last_sentence_end_pos+1]
                            one_sentence = one_sentence[last_sentence_end_pos+1:]
                            logging.info("Qwen got sentence: %s, input tokens: %d, output tokens: %d", sentence, response.usage.input_tokens, response.usage.output_tokens)
                            yield sentence
                else:
                    logging.warning('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                if self._needs_interrupt:
                    responses.close()
                    logging.info("Interrupting qwen generation for user_id: %s", self._user)
                    break
            except TimeoutError:
                logging.info("qwen timeout for user_id: %s", self._user)
                break
            except asyncio.CancelledError:
                logging.info("qwen cancelled for user_id: %s", self._user)
                break
            except Exception as e:
                logging.error(f"Error in qwen generation: {e}")
                break

        if one_sentence != '':
            yield one_sentence
            logging.info("Last sentence from Qwen: %s", one_sentence)

        logging.debug("qwen got full content: %s", full_content)
        if use_redis_history:
            await self.save_message_to_redis(self._user, full_content)
        self._needs_interrupt = False
        self._producing_response = False
        self._interactions_count += 1


class QwenVisualLLM(VisualLLMBase):
    def get_user_message(sefl, user_questions_str:str, files: List[str]):
        """{
        'role':
        'user',
        'content': [
                {
                    'image': local_file_path1
                },
                {
                    'image': local_file_path2
                },
                {
                    'text': '图片里有什么东西?'
                },
            ]
        }"""
        # 遍历files，将图片路径加入content，生成上述结构
        question: str = ""
        if user_questions_str is None:
            question += "请描述你在图片里看到的内容。"
        else:
            question += user_questions_str
        msg = {
            'role': 'user',
            'content': []
        }
        for file in files:
            msg['content'].append({
                'image': file
            })
        msg['content'].append({
            'text': question
        })
        return msg

    async def call_mml_with_local_file(self, user_id: str,  files: List[str], model: str = "qwen-vl-plus") -> str:
        try:
            stream_key = REDIS_CHAT_KEY+user_id
            user_questions: List[str] = await self.get_user_history(user_id)
            if user_questions is None or len(user_questions) == 0:
                user_questions_str = ""
            else:
                user_questions_str = " ".join(user_questions)
            user_message = self.get_user_message(user_questions_str, files)
            messages = [
                {
                    'role': 'system',
                    'content': [{
                        'text': self._system_prompt + """ 请不要以第三人称的口吻描述图片，而是用第一人称的口吻描述你看到的场景，或者回答用户的问题。
                        比如：我喜欢这件衣服，蓝白相间的格子简单又大气。"""
                    }]
                }, 
                user_message]

            response = MultiModalConversation.call(model=model, messages=messages)
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                await write_chat_to_redis(stream_key, text=content, timestamp=time.time(), srcname=CHAT_MEMBER_ASSITANT)
                logging.debug(f"Saved  vl message to redis: {content}")
                return content
            else:
                logging.warning('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                return None
        except Exception as e:
            logging.error(f"Error in calling mml with local file: {e}")
            return None


async def main():
    qwen = Qwen()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    logging.info("Start testing qwen generation")
    query = "请帮我写一首诗, 下雪了啊"
    query = "现在外面是几度？下雨了吗？"
    query = "你看看我画的好看吗？"
    query = "从2加到10等于几？"
    user = "user1"
    model = Generation.Models.qwen_max
    async for output in qwen.generate_text_streamed(user,  model=model,
                                                    addtional_user_message=Message(content=query, role=MessageRole.user), use_redis_history=False):
        output_new =qwen.parse_custom_function(output)
        if output_new is not None and output_new!= "":
            logging.info(f"Qwen generated output after parse_custom_function: {output_new}")
    await qwen.handle_custom_function_output()



if __name__ == '__main__':
    asyncio.run(main())
