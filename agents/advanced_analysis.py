import asyncio
import logging
from typing import Callable, List, Dict
from llm import get_llm, LLMBase
from common.config import config
from llm.llm_base import Message, MessageRole
import difflib
from db.redis_cli import write_summary_and_state_to_redis, get_summary_and_state_from_redis
from common.search import search
from agents.task_manager import TaskManager


class AdvancedAnalysis:
    def __init__(self, llm_engine: LLMBase, send_to_app: Callable, trigger_visual_analysis: Callable):
        self.llm_engine = llm_engine
        self.analysis_llm = get_llm()  # 单独的 LLM 客户端
        self._analysis_in_progress = False
        self._user = ''
        self.conversation_summary = ""
        self.user_state = ""
        self.send_to_app = send_to_app
        self.trigger_visual_analysis = trigger_visual_analysis
        
        # 初始化 TaskManager
        self.task_manager = TaskManager(
            self.llm_engine,
            send_to_app=self.send_to_app,
            trigger_visual_analysis=self.trigger_visual_analysis,
            is_analysis_in_progress=lambda: self._analysis_in_progress
        )
        
        # 启动 TaskManager 的周期执行任务
        if config.agents.task_check_interval > 0:
            asyncio.create_task(self.task_manager.task_check_loop())

    def is_response_similar_by_difflib(self, response1: str, response2: str, threshold: float = 0.8) -> bool:
        """
        使用 difflib 判断两个回复是否相似。
        
        :param response1: 第一个回复
        :param response2: 第二个回复
        :param threshold: 相似度阈值，默认为0.8
        :return: 如果相似度高于阈值，返回True；否则返回False
        """
        similarity = difflib.SequenceMatcher(None, response1, response2).ratio()
        return similarity > threshold

    async def is_response_similar_by_llm(self, response1: str, response2: str) -> bool:
        """
        使用 LLM 判断两个回复是否相似。
        
        :param response1: 第一个回复
        :param response2: 第二个回复
        :return: 如果 LLM 认为回复相似，返回True；否则返回False
        """
        result = await self._llm_analysis("response_similarity", response1=response1, response2=response2)
        return result.get('is_similar', False)

    async def _llm_analysis(self, analysis_type: str, **kwargs) -> Dict:
        """
        使用 LLM 进行各种分析的底层函数。
        
        :param analysis_type: 分析类型
        :param kwargs: 其他参数
        :return: 分析结果字典
        """
        system_prompts = {
            "response_similarity": f"""
            You are an AI assistant tasked with evaluating the similarity of two responses. 
            
            Previous conversation summary: {self.conversation_summary}

            Analyze the given responses and determine if they are similar in content and intent.
            Note that sometimes the responses are similar, for example, the temperature is 20 degrees and 2 degrees, but they are not the same, you should return No.
            Provide your analysis in the following format:
            
            is_similar: [Yes/No]
            reason: [Your explanation here]
            """,
            "followup_necessity": f"""
            You are an AI assistant tasked with evaluating the necessity of a follow-up question.
            
            Previous conversation summary: {self.conversation_summary}

            Analyze the given conversation and proposed follow-up question, and determine if the follow-up is necessary.
            Provide your analysis in the following format:
            
            is_necessary: [Yes/No]
            reason: [Your explanation here]
            """
            # Add more analysis types as needed
        }

        self.analysis_llm.clear_history()
        self.analysis_llm.add_history(Message(role=MessageRole.system, content=system_prompts[analysis_type]))

        if analysis_type == "response_similarity":
            self.analysis_llm.add_history(Message(role=MessageRole.user, content=f"Response 1: {kwargs['response1']}\nResponse 2: {kwargs['response2']}\nAre these responses similar? Please provide your analysis."))
        elif analysis_type == "followup_necessity":
            self.analysis_llm.add_history(Message(role=MessageRole.user, content=f"User message: {kwargs['user_message']}"))
            self.analysis_llm.add_history(Message(role=MessageRole.assistant, content=f"Assistant response: {kwargs['assistant_response']}"))
            self.analysis_llm.add_history(Message(role=MessageRole.user, content=f"Proposed follow-up question: {kwargs['followup_question']}\nIs this follow-up question necessary? Please provide your analysis."))

        analysis = await self.analysis_llm.generate_text(self._user, model=config.llm.model, strict_mode=False, use_redis_history=False, system_prompt=system_prompts[analysis_type])
        return self._parse_llm_analysis(analysis, analysis_type)

    def _parse_llm_analysis(self, analysis: str, analysis_type: str) -> Dict:
        result = {}
        lines = analysis.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if analysis_type == "response_similarity" and key == 'is_similar':
                    result['is_similar'] = self.string_means_yes(value)
                elif analysis_type == "followup_necessity" and key == 'is_necessary':
                    result['is_necessary'] = self.string_means_yes(value)
                elif key == 'reason':
                    result['reason'] = value

        return result

    async def is_followup_necessary(self, user_message: str, assistant_response: str, followup_question: str) -> bool:
        """
        判断是否有必要发送跟进问题。

        :param user_message: 用户的原始问题
        :param assistant_response: 助手的回复
        :param followup_question: 生成的跟进问题
        :return: 如果跟进问题有必要，返回 True；否则返回 False
        """
        result = await self._llm_analysis("followup_necessity", 
                                          user_message=user_message, 
                                          assistant_response=assistant_response, 
                                          followup_question=followup_question)
        return result.get('is_necessary', False)

    async def process_message(self, user: str, user_message: str, assistant_response: str):
        logging.info("AdvancedAnalysis processing message: " + user_message + ", assistant_response: " + assistant_response)
        await self.set_user(user)
        if self.is_analysis_in_progress():
            logging.info("AdvancedAnalysis analysis in progress, skip")
            return
        self._analysis_in_progress = True
        self.llm_engine.add_history(Message(role=MessageRole.assistant, content=assistant_response))
        self.llm_engine.add_history(Message(role=MessageRole.user, content='Please analyze the conversation and provide insights in the format required by the system prompt'))
        try:
            analysis_result = await self.analyze_conversation()
            self.llm_engine.clear_history()
            
            search_results = None
            if analysis_result['needs_search']:
                search_results = await search(analysis_result['search_keywords'])
                logging.info(f"AdvancedAnalysis needs_search: {analysis_result['search_keywords']}, search_results: {search_results}")
            
            if analysis_result['needs_improvement']:
                logging.info("AdvancedAnalysis needs_improvement: " + analysis_result['improvement_reason'])
                improved_response = await self.improve_response(user, analysis_result, search_results)
                
                # 使用 difflib 判断改进的回复是否与原始回复相似
                if not self.is_response_similar_by_difflib(assistant_response, improved_response):
                    # 使用 LLM 进行二次确认
                    if not await self.is_response_similar_by_llm(assistant_response, improved_response):
                        await self.send_to_app(improved_response, save_to_redis=True)
                        self.llm_engine.add_history(Message(role=MessageRole.assistant, content=improved_response))
                    else:
                        logging.info(f"LLM determined the improved response is too similar, skipping: {improved_response}")
                else:
                    logging.info(f"Improved response is too similar to the original (difflib), skipping: {improved_response}")
            
            if analysis_result['needs_visual_analysis']:
                logging.info("AdvancedAnalysis needs_visual_analysis: " + analysis_result['visual_analysis_reason'])
                visual_analysis_prompt = analysis_result['visual_analysis_reason']
                await self.trigger_visual_analysis(visual_analysis_prompt)
            
            if analysis_result['needs_followup']:
                logging.info("AdvancedAnalysis needs_followup: " + analysis_result['followup_topic'])
                followup_question = await self.generate_followup_question(user, analysis_result['followup_topic'])
                
                # 使用 LLM 判断是否有必要发送跟进问题
                if await self.is_followup_necessary(user_message, assistant_response, followup_question):
                    await self.send_to_app(followup_question, save_to_redis=True)
                    self.llm_engine.add_history(Message(role=MessageRole.assistant, content=followup_question))
                else:
                    logging.info(f"Follow-up question is not necessary, skipping: {followup_question}")

            # 触发任务检查
            if config.agents.task_check_interval > 0:
                self.task_manager.trigger_task_check()

        finally:
            self._analysis_in_progress = False

    async def analyze_conversation(self) -> Dict:
        system_prompt = f"""
        {self.llm_engine.get_time_and_location()}
        You are an AI assistant. Analyze the following conversation and provide insights:

        Previous conversation summary: [ {self.conversation_summary}]
        Previous user state: [ {self.user_state}]

        Analyze the conversation based on the following criteria:
        1. Provide a brief, objective summary of the current conversation.
        2. Does the last response need improvement? [Yes/No] If so, why?
        3. Is visual analysis needed for better understanding or response? [Yes/No] If so, what kind?
        4. Are there any topics or questions that need follow-up? [Yes/No]
        5. What is the user's current emotional state or intent?
        6. Is additional information from an internet search needed? [Yes/No] If so, provide search keywords.

        Provide your analysis in a structured, single-line format for each point. Be detailed, well-reasoned, and comprehensive in your analysis. Example format:

        Example 1:
        summary: The user asked about dinner time, and the assistant provided ...
        needs_improvement: Yes
        improvement_reason: The response was off-topic and didn't address the user's question about dinner time.
        needs_visual_analysis: No
        visual_analysis_reason: N/A
        needs_followup: Yes
        followup_topic: Specific dinner time
        user_state: Seeking information about dinner time
        needs_search: No
        search_keywords: N/A

        Example 2:
        summary: The user inquired about weather data, and the assistant ...
        needs_improvement: No
        improvement_reason: N/A
        needs_visual_analysis: Yes
        visual_analysis_reason: We can see what the user is doing and feeling to help answering the question if needed.
        needs_followup: No
        followup_topic: N/A
        user_state: Curious about weather data
        needs_search: Yes
        search_keywords: reasonable and short keywords for improving the response if needed.

        Now, analyze the conversation and provide your insights in the same format. Your latest response must strictly follow the above format.
        """  

        prompt_message = Message(role=MessageRole.user, content='Please analyze the conversation and provide insights in the format required by the system prompt')
        analysis = await self.llm_engine.generate_text(self._user, config.llm.model, strict_mode=False, use_redis_history=True, system_prompt=system_prompt, addtional_user_message=prompt_message)
        logging.info("AdvancedAnalysis analysis: " + analysis)
        parsed_analysis = self.parse_analysis(analysis)
        
        # 更新 LLMBase 中的对话摘要
        self.llm_engine.set_conversation_summary(parsed_analysis['summary'])
        
        # Store summary and user_state in Redis
        await write_summary_and_state_to_redis(self._user, parsed_analysis['summary'], parsed_analysis['user_state'])
        
        return parsed_analysis
    
    def string_means_yes(self, string: str) -> bool:
        return string.lower() in ['true', 'yes', '是']

    def parse_analysis(self, analysis: str) -> Dict:
        result = {
            "summary": "",
            "needs_improvement": False,
            "improvement_reason": "",
            "needs_visual_analysis": False,
            "visual_analysis_reason": "",
            "needs_followup": False,
            "followup_topic": "",
            "user_state": "",
            "needs_search": False,
            "search_keywords": ""
        }
        
        lines = analysis.strip().split('\n')
        for line in lines:
            if not line:
                continue
            if ':' not in line:
                logging.warning(f"Skipping invalid line in analysis: {line}")
                continue
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'summary':
                result['summary'] = value
            elif key == 'needs_improvement':
                result['needs_improvement'] = self.string_means_yes(value)
            elif key == 'improvement_reason':
                result['improvement_reason'] = value if value.lower() != 'n/a' else ""
            elif key == 'needs_visual_analysis':
                result['needs_visual_analysis'] = self.string_means_yes(value)
            elif key == 'visual_analysis_reason':
                result['visual_analysis_reason'] = value if value.lower() != 'n/a' else ""
            elif key == 'needs_followup':
                result['needs_followup'] = self.string_means_yes(value)
            elif key == 'followup_topic':
                result['followup_topic'] = value if value.lower() != 'n/a' else ""
            elif key == 'user_state':
                result['user_state'] = value
            elif key == 'needs_search':
                result['needs_search'] = self.string_means_yes(value)
            elif key == 'search_keywords':
                result['search_keywords'] = value if value.lower() != 'n/a' else ""
        
        # 更新对话摘要
        self.conversation_summary = result['summary']
        
        return result

    async def improve_response(self, user: str, analysis_result: Dict, search_results: str = None) -> str:
        prompt = f"""Based on the following analysis, generate an improved response:
        Analysis: {analysis_result['improvement_reason']}
        """

        if search_results:
            prompt += f"\nSearch Results: {search_results}\n"
            prompt += "Incorporate relevant information from the search results if available."

        prompt += "\nProvide an improved response that addresses the identified issues. Don't repeat what you've already said."

        prompt_message = Message(role=MessageRole.user, content=prompt)
        improved_response = await self.llm_engine.generate_text(user, config.llm.model, addtional_user_message=prompt_message, strict_mode=False)
        logging.debug("AdvancedAnalysis improved_response: " + improved_response)
        return improved_response

    async def generate_followup_question(self, user: str, followup_topic: str) -> str:
        prompt = f"""Based on the following analysis, generate a follow-up question:
        Analysis: {followup_topic}
        Provide a follow-up question that deepens the conversation or clarifies any ambiguities.
        """

        prompt_message = Message(role=MessageRole.user, content=prompt)
        followup_question = await self.llm_engine.generate_text(user, config.llm.model, addtional_user_message=prompt_message, strict_mode=False)
        logging.debug("AdvancedAnalysis followup_question: " + followup_question)
        return followup_question

    def is_analysis_in_progress(self) -> bool:
        return self._analysis_in_progress

    async def set_user(self, user: str):
        if self._user != user:
            self._user = user
            if config.agents.task_check_interval > 0:
                await self.task_manager.set_user(user)
            # Read summary and user_state from Redis
            self.conversation_summary, self.user_state = await get_summary_and_state_from_redis(user)

