import asyncio
import logging
from typing import Callable, List, Dict
from llm import get_llm, LLMBase
from common.config import config
from llm.llm_base import Message, MessageRole  # Add this import

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
        self._analysis_in_progress = False
        self.current_analysis_tree: AnalysisTree = None

    async def process_message(self, user: str, user_message: str, assistant_response: str, 
                              send_to_app: Callable, 
                              trigger_visual_analysis: Callable):
        logging.info("AdvancedAnalysis processing message: " + user_message + ", assistant_response: " + assistant_response)
        if self.is_analysis_in_progress():
            logging.info("AdvancedAnalysis analysis in progress, skip")
            return
        self._analysis_in_progress = True
        self.llm_engine.add_history(Message(role=MessageRole.assistant, content=assistant_response))
        self.llm_engine.add_history(Message(role=MessageRole.user, content='Please analyze the conversation and provide insights in the format required by the system prompt'))
        try:
            analysis_result = await self.analyze_conversation(user)
            self.llm_engine.clear_history()
            
            if analysis_result['needs_improvement']:
                logging.info("AdvancedAnalysis needs_improvement: " + analysis_result['improvement_reason'])
                improved_response = await self.improve_response(user, user_message, assistant_response, analysis_result)
                await send_to_app(improved_response, save_to_redis=True)
                self.llm_engine.add_history(Message(role=MessageRole.assistant, content=improved_response))
            
            if analysis_result['needs_visual_analysis']:
                logging.info("AdvancedAnalysis needs_visual_analysis: " + analysis_result['visual_analysis_reason'])
                visual_analysis_prompt = analysis_result['visual_analysis_reason']
                await trigger_visual_analysis(visual_analysis_prompt)
            
            if analysis_result['needs_followup']:
                logging.info("AdvancedAnalysis needs_followup: " + analysis_result['followup_topic'])
                followup_question = await self.generate_followup_question(user, analysis_result['followup_topic'])
                await send_to_app(followup_question, save_to_redis=True)
                self.llm_engine.add_history(Message(role=MessageRole.assistant, content=followup_question))

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
