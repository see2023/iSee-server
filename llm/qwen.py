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
from db.redis_cli import get_redis_client, REDIS_CHAT_KEY, write_chat_to_redis, get_detected_names_from_redis



class Qwen(LLMBase):
    def __init__(self):
        super().__init__("DASHSCOPE_API_KEY")

    async def generate_text_streamed(self, user_id: str, model: str = Generation.Models.qwen_max, addtional_user_message: Message = None) -> AsyncIterable[str]:
        history = await self.build_history_from_redis(user_id)
        if addtional_user_message is not None:
            history.append(addtional_user_message.to_dict())
        if not history or len(history) == 0:
            logging.info("No history found for user_id: %s", user_id)
            yield ""
            return
        responses = Generation.call(
            model,
            messages=history,
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
                        last_sentence_end_pos = LLMBase.last_sentence_end(one_sentence)
                        if last_sentence_end_pos > 0:
                            sentence = one_sentence[:last_sentence_end_pos+1]
                            one_sentence = one_sentence[last_sentence_end_pos+1:]
                            logging.info("Qwen got sentence: %s", sentence)
                            yield sentence
                else:
                    logging.warning('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                        response.request_id, response.status_code,
                        response.code, response.message
                    ))
                if self._needs_interrupt:
                    responses.close()
                    logging.info("Interrupting qwen generation for user_id: %s", user_id)
                    break
            except TimeoutError:
                logging.info("qwen timeout for user_id: %s", user_id)
                break
            except asyncio.CancelledError:
                logging.info("qwen cancelled for user_id: %s", user_id)
                break

        if one_sentence != '':
            yield one_sentence
            logging.info("Last sentence from Qwen: %s", one_sentence)

        logging.debug("qwen got full content: %s", full_content)
        await self.save_message_to_redis(user_id, full_content)
        self._needs_interrupt = False
        self._producing_response = False


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
        if user_questions_str is not None:
            question = "之前我的问题：["+user_questions_str+"]. 你在描述图片时，可以参考。 "
        # question += "你看到了什么？ 跟我聊聊吧！还是你要问我问题呢？"
        question += "请描述你在图片里看到的内容。"
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
                        'text': self._system_prompt + """ 请不要以第三人称的口吻描述图片，而是用第一人称的口吻描述你看到的场景。
                        描述的方式：'这件衣服好漂亮，蓝色的格子我也喜欢。' '"""
                    }]
                }, 
                user_message]

            response = MultiModalConversation.call(model=model, messages=messages)
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                write_chat_to_redis(stream_key, text=content, timestamp=time.time(), srcname=CHAT_MEMBER_ASSITANT)
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
    async for output in qwen.generate_text_streamed("user1",  model=Generation.Models.qwen_max,
                                                    addtional_user_message=Message(content="你好，今天好开心啊！", role=MessageRole.user)):
        print(output)


if __name__ == '__main__':
    asyncio.run(main())
