import asyncio
import logging
from typing import Callable, List, Dict
from llm import get_llm, LLMBase
from common.config import config
from llm.llm_base import Message, MessageRole
import difflib

class AnalysisNode:
    def __init__(self, content: str, children: List['AnalysisNode'] = None):
        self.content = content
        self.children = children or []

class AnalysisTree:
    def __init__(self, root: AnalysisNode):
        self.root = root

    def add_child(self, parent: AnalysisNode, child: AnalysisNode):
        parent.children.append(child)

class AdvancedAnalysis:
    def __init__(self, llm_engine: LLMBase):
        self.llm_engine = llm_engine
        self.analysis_llm = get_llm()  # 单独的 LLM 客户端
        self._analysis_in_progress = False
        self.current_analysis_tree: AnalysisTree = None
        self._user = ''

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
            "response_similarity": """
            You are an AI assistant tasked with evaluating the similarity of two responses. 
            Analyze the given responses and determine if they are similar in content and intent.
            Provide your analysis in the following format:
            
            is_similar: [Yes/No]
            reason: [Your explanation here]
            """,
            "followup_necessity": """
            You are an AI assistant tasked with evaluating the necessity of a follow-up question.
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

        analysis = await self.analysis_llm.generate_text(self._user, model=config.llm.model, strict_mode=False)
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
                    result['is_similar'] = value.lower() == 'yes'
                elif analysis_type == "followup_necessity" and key == 'is_necessary':
                    result['is_necessary'] = value.lower() == 'yes'
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

    async def process_message(self, user: str, user_message: str, assistant_response: str, 
                              send_to_app: Callable, 
                              trigger_visual_analysis: Callable):
        logging.info("AdvancedAnalysis processing message: " + user_message + ", assistant_response: " + assistant_response)
        if self.is_analysis_in_progress():
            logging.info("AdvancedAnalysis analysis in progress, skip")
            return
        self._analysis_in_progress = True
        self._user = user
        self.llm_engine.add_history(Message(role=MessageRole.assistant, content=assistant_response))
        self.llm_engine.add_history(Message(role=MessageRole.user, content='Please analyze the conversation and provide insights in the format required by the system prompt'))
        try:
            analysis_result = await self.analyze_conversation(user)
            self.llm_engine.clear_history()
            
            if analysis_result['needs_improvement']:
                logging.info("AdvancedAnalysis needs_improvement: " + analysis_result['improvement_reason'])
                improved_response = await self.improve_response(user, user_message, assistant_response, analysis_result)
                
                # 使用 difflib 判断改进的回复是否与原始回复相似
                if not self.is_response_similar_by_difflib(assistant_response, improved_response):
                    # 使用 LLM 进行二次确认
                    if not await self.is_response_similar_by_llm(assistant_response, improved_response):
                        await send_to_app(improved_response, save_to_redis=True)
                        self.llm_engine.add_history(Message(role=MessageRole.assistant, content=improved_response))
                    else:
                        logging.info("LLM determined the improved response is too similar, skipping")
                else:
                    logging.info("Improved response is too similar to the original (difflib), skipping")
            
            if analysis_result['needs_visual_analysis']:
                logging.info("AdvancedAnalysis needs_visual_analysis: " + analysis_result['visual_analysis_reason'])
                visual_analysis_prompt = analysis_result['visual_analysis_reason']
                await trigger_visual_analysis(visual_analysis_prompt)
            
            if analysis_result['needs_followup']:
                logging.info("AdvancedAnalysis needs_followup: " + analysis_result['followup_topic'])
                followup_question = await self.generate_followup_question(user, analysis_result['followup_topic'])
                
                # 使用 LLM 判断是否有必要发送跟进问题
                if await self.is_followup_necessary(user_message, assistant_response, followup_question):
                    await send_to_app(followup_question, save_to_redis=True)
                    self.llm_engine.add_history(Message(role=MessageRole.assistant, content=followup_question))
                else:
                    logging.info("Follow-up question is not necessary, skipping")

        finally:
            self._analysis_in_progress = False

    async def analyze_conversation(self, user: str) -> Dict:
        system_prompt = f"""
        You are an AI assistant. Analyze the following conversation and provide insights:

        1. Does the last response need improvement? If so, why?
        2. Is visual analysis needed for better understanding or response? If so, what kind?
        3. Are there any topics or questions that need follow-up?
        4. What is the user's current emotional state or intent?

        Provide your analysis in a structured format like the examples below:

        Example 1:
        needs_improvement: Yes
        improvement_reason: The response was off-topic and didn't address the user's question about dinner time.
        needs_visual_analysis: No
        visual_analysis_reason: N/A
        needs_followup: Yes
        followup_topic: Specific dinner time
        user_state: Seeking information about dinner time

        Example 2:
        needs_improvement: No
        improvement_reason: N/A
        needs_visual_analysis: Yes
        visual_analysis_reason: We can see what the user is doing and feeling to help answering the question.
        needs_followup: No
        followup_topic: N/A
        user_state: Curious about weather data

        Now, analyze the conversation and provide your insights in the same format, please note that the history conversation is not required to follow the format, but your latest response must strictly follow the above format.
        """  

        analysis = await self.llm_engine.generate_text(user, config.llm.model, strict_mode=False, system_prompt=system_prompt)
        return self.parse_analysis(analysis)

    def parse_analysis(self, analysis: str) -> Dict:
        result = {
            "needs_improvement": False,
            "improvement_reason": "",
            "needs_visual_analysis": False,
            "visual_analysis_reason": "",
            "needs_followup": False,
            "followup_topic": "",
            "user_state": ""
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
            
            if key == 'needs_improvement':
                result['needs_improvement'] = value.lower() == 'yes'
            elif key == 'improvement_reason':
                result['improvement_reason'] = value if value.lower() != 'n/a' else ""
            elif key == 'needs_visual_analysis':
                result['needs_visual_analysis'] = value.lower() == 'yes'
            elif key == 'visual_analysis_reason':
                result['visual_analysis_reason'] = value if value.lower() != 'n/a' else ""
            elif key == 'needs_followup':
                result['needs_followup'] = value.lower() == 'yes'
            elif key == 'followup_topic':
                result['followup_topic'] = value if value.lower() != 'n/a' else ""
            elif key == 'user_state':
                result['user_state'] = value
        
        return result

    async def improve_response(self, user: str, user_message: str, assistant_response: str, analysis_result: Dict) -> str:
        prompt = f"""Based on the following analysis, generate an improved response:
        Analysis: {analysis_result['improvement_reason']}
        Provide an improved response that addresses the identified issues.
        """

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

    # Future methods for implementing thought tree or LangGraph processing
    async def build_thought_tree(self, root_thought: str):
        # Implement thought tree building logic
        pass

    async def process_lang_graph(self):
        # Implement LangGraph processing logic
        pass
