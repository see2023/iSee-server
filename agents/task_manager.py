import asyncio
import logging
import json
import time
from typing import Callable, List, Dict
from llm import get_llm, LLMBase
from common.config import config
from llm.llm_base import Message, MessageRole
from common.search import search
from db.redis_cli import get_redis_client, REDIS_PREFIX, REDIS_CHAT_KEY

class Task:
    def __init__(self, name: str, description: str, key_points: List[str], due_time: str = None):
        self.name = name
        self.description = description
        self.key_points = key_points
        self.due_time = due_time
        self.completed_points = []
        self.last_update = ""

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "key_points": self.key_points,
            "due_time": self.due_time,
            "completed_points": self.completed_points,
            "last_update": self.last_update
        }

    @classmethod
    def from_dict(cls, data):
        task = cls(data["name"], data["description"], data["key_points"], data["due_time"])
        task.completed_points = data["completed_points"]
        task.last_update = data["last_update"]
        return task

class TaskManager:
    def __init__(self, llm_engine: LLMBase, send_to_app: Callable, trigger_visual_analysis: Callable, is_analysis_in_progress: Callable):
        self.llm_engine = llm_engine
        self.task_llm = get_llm()  # 单独的 LLM 客户端用于任务管理
        self._task_in_progress = False
        self._user = ''
        self.send_to_app = send_to_app
        self.trigger_visual_analysis = trigger_visual_analysis
        self.is_analysis_in_progress = is_analysis_in_progress
        self.tasks: Dict[str, Task] = {}
        self._redis_client = get_redis_client()
        self.task_check_event = asyncio.Event()
        self.task_check_interval = config.agents.task_check_interval
        self.task_check_min_interval = config.agents.task_check_min_interval
        self.last_check_time = 0

    async def set_user(self, user: str):
        self._user = user
        await self.load_tasks_from_redis()

    async def load_tasks_from_redis(self):
        tasks_json = await self._redis_client.get(f"{REDIS_PREFIX}tasks:{self._user}")
        if tasks_json:
            tasks_data = json.loads(tasks_json)
            self.tasks = {name: Task.from_dict(task_data) for name, task_data in tasks_data.items()}

    async def save_tasks_to_redis(self):
        tasks_data = {name: task.to_dict() for name, task in self.tasks.items()}
        await self._redis_client.set(f"{REDIS_PREFIX}tasks:{self._user}", json.dumps(tasks_data))

    async def add_task(self, name: str, description: str, key_points: List[str], due_time: str = None):
        self.tasks[name] = Task(name, description, key_points, due_time)
        await self.save_tasks_to_redis()
        await self.send_to_app(f"新任务 '{name}' 已添加。关键点：{', '.join(key_points)}", save_to_redis=True)

    async def get_task_summary(self):
        if not self.tasks:
            await self.send_to_app("当前没有进行中的任务。")
            return

        summary = "当前任务概况：\n"
        for task in self.tasks.values():
            completed = len(task.completed_points)
            total = len(task.key_points)
            summary += f"- {task.name}: 完成 {completed}/{total} 个关键点, 截止日期: {task.due_time or '未设置'}\n"
            if task.last_update:
                summary += f"  最新进展: {task.last_update}\n"
        await self.send_to_app(summary)

    async def initialize_tasks(self):
        if not self.tasks:
            tasks = await self.generate_tasks_from_history()
            if tasks:
                for task in tasks:
                    await self.add_task(task['name'], task['description'], task['key_points'], task['due_time'])
            else:
                await self.prompt_user_for_tasks()

    async def generate_tasks_from_history(self) -> List[Dict]:
        system_prompt = """
        You are an AI assistant tasked with identifying potential tasks from a conversation history. 
        Analyze the conversation and identify any tasks, goals, or objectives mentioned by the user.
        Consider the following:
        
        1. Avoid generating duplicate tasks
        2. Carefully check if a task has already been completed
        3. Only include future tasks

        For each task you identify, provide the following information:
        
        task_name: [Task name]
        description: [Brief description of the task]
        key_points: [Comma-separated list of key points or steps to complete the task]
        due_time: [Due time if mentioned, otherwise "Not specified"]

        If no tasks are identified, respond with "No tasks identified."
        """

        conversation_history = await self.get_conversation_history()
        
        self.task_llm.clear_history()
        additional_user_message = Message(role=MessageRole.user, content="Please respond in the format required by the system prompt: task_name: , description: , key_points: , due_time: .")
        response = await self.task_llm.generate_text(self._user, model=config.llm.model, strict_mode=False, use_redis_history=True, system_prompt=system_prompt, addtional_user_message=additional_user_message)
        
        return self._parse_task_generation_response(response)

    def _parse_task_generation_response(self, response: str) -> List[Dict]:
        tasks = []
        current_task = {}
        
        for line in response.strip().split('\n'):
            if line.startswith("task_name:"):
                if current_task:
                    tasks.append(current_task)
                current_task = {"name": line.split(":", 1)[1].strip()}
            elif line.startswith("description:"):
                current_task["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("key_points:"):
                current_task["key_points"] = [point.strip() for point in line.split(":", 1)[1].split(",")]
            elif line.startswith("due_time:"):
                current_task["due_time"] = line.split(":", 1)[1].strip()
        
        if current_task:
            tasks.append(current_task)
        
        return tasks

    async def get_conversation_history(self) -> str:
        # Retrieve the last 50 messages from Redis
        messages = await self._redis_client.xrevrange(f"{REDIS_CHAT_KEY}{self._user}", count=50)
        conversation = []
        for msg in reversed(messages):
            msg_data = msg[1]
            role = "User" if msg_data['srcname'] == 'app' else "Assistant"
            conversation.append(f"{role}: {msg_data['text']}")
        return "\n".join(conversation)

    async def prompt_user_for_tasks(self):
        system_prompt = """
        You are an AI assistant tasked with generating a prompt to ask the user about their tasks or plans.
        Based on the recent conversation history, suggest possible tasks or activities the user might be interested in.
        Create a friendly and engaging prompt that encourages the user to share their plans or tasks.
        Your response should be a single paragraph, directly addressing the user.
        """

        conversation_history = await self.get_conversation_history()
        
        additional_user_message = Message(
            role=MessageRole.user,
            content=f"Recent conversation:\n{conversation_history}\n\nGenerate a prompt to ask about the user's tasks or plans."
        )

        generated_prompt = await self.task_llm.generate_text(
            self._user,
            model=config.llm.model,
            strict_mode=False,
            use_redis_history=True,
            system_prompt=system_prompt,
            addtional_user_message=additional_user_message
        )

        await self.send_to_app(generated_prompt, save_to_redis=True)

    async def task_check_loop(self):
        while True:
            try:
                await asyncio.wait_for(self.task_check_event.wait(), timeout=self.task_check_interval)
            except asyncio.TimeoutError:
                pass
            finally:
                self.task_check_event.clear()
                if not self._user:
                    logging.debug("Task check loop skipped due to no user")
                    continue
                if not self.tasks:
                    logging.info("Task check loop skipped due to no tasks")
                    await self.initialize_tasks()
                elif self._task_in_progress:
                    logging.info("Task check loop skipped due to task in progress")
                else:
                    current_time = time.time()
                    if current_time - self.last_check_time >= self.task_check_min_interval:
                        self.last_check_time = current_time
                        logging.info("Task check loop, check tasks")
                        await self.check_all_tasks()
                    else:
                        logging.info("Task check loop skipped due to short interval")

    def trigger_task_check(self):
        self.task_check_event.set()

    async def check_all_tasks(self):
        try:
            if not self.tasks:
                return
            for task_name, task in self.tasks.items():
                await self.check_task(task)
        except Exception as e:
            logging.error(f"Error in check_all_tasks: {e}")

    async def check_task(self, task: Task):
        system_prompt = f"""
        You are an AI assistant tasked with checking the progress of a task and providing proactive suggestions. 
        Current task: {task.name}
        Description: {task.description}
        Key points: {', '.join(task.key_points)}
        Completed points: {', '.join(task.completed_points)}
        Due time: {task.due_time}
        Last update: {task.last_update}

        Based on the task information, please provide:
        1. A progress assessment
        2. Whether the task is completed
        3. Whether the user wants to delete the task
        4. Suggestions for next steps (if not completed)
        5. Any information that needs to be searched to help with the task
        6. Whether a visual analysis might be helpful

        Format your response as follows:
        progress_assessment: [brief assessment of the task progress]
        is_completed: [Yes/No]
        delete_task: [Yes/No]
        suggestions: [list of suggestions for next steps, or "N/A" if completed]
        search_queries: [list of search queries that could help with the task, or "N/A" if completed]
        need_visual_analysis: [Yes/No]
        visual_analysis_reason: [reason for visual analysis, if needed, or "N/A" if not needed]
        """

        self.task_llm.clear_history()
        addtional_user_message = Message(role=MessageRole.user, content=f"Reply in the format required by the system prompt: progress_assessment: , is_completed: , delete_task: , suggestions: , search_queries: , need_visual_analysis: , visual_analysis_reason: .")
        analysis = await self.task_llm.generate_text(self._user, model=config.llm.model, strict_mode=False, use_redis_history=True, system_prompt=system_prompt, addtional_user_message=addtional_user_message)
        result = self._parse_task_check_result(analysis)
        
        if result['is_completed'] or result['delete_task']:
            await self.complete_task(task)
        else:
            await self.process_task_check_result(task, result)

    def _parse_task_check_result(self, analysis: str) -> Dict:
        result = {
            "progress_assessment": "",
            "is_completed": False,
            "delete_task": False,
            "suggestions": [],
            "search_queries": [],
            "need_visual_analysis": False,
            "visual_analysis_reason": ""
        }
        
        current_key = ""
        for line in analysis.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                if key in result:
                    current_key = key
                    if key in ["suggestions", "search_queries"]:
                        result[key] = [item.strip() for item in value.strip().strip('[]').split(',') if item.strip()]
                    elif key in ["is_completed", "need_visual_analysis", "delete_task"]:
                        result[key] = value.strip().lower() == "yes"
                    else:
                        result[key] = value.strip()
            elif current_key in ["suggestions", "search_queries"]:
                result[current_key].append(line.strip().strip('- '))

        return result

    async def complete_task(self, task: Task):
        completion_message = f"恭喜！任务 '{task.name}' 已完成。\n"
        completion_message += f"任务描述: {task.description}\n"
        completion_message += f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"

        await self.send_to_app(completion_message, save_to_redis=True)
        
        # Remove the task from the tasks dictionary
        del self.tasks[task.name]
        
        # Save the updated tasks to Redis
        await self.save_tasks_to_redis()

    async def process_task_check_result(self, task: Task, result: Dict):
        task.last_update = result['progress_assessment']

        if result['suggestions'] and not "N/A" in result['suggestions']:
            confirmed_suggestions = await self.confirm_suggestions(task.name, result['suggestions'])
            if confirmed_suggestions:
                response = f"任务 '{task.name}' 检查更新：\n"
                response += f"进展评估：{result['progress_assessment']}\n"
                response += "建议下一步：\n" + "\n".join(f"- {suggestion}" for suggestion in confirmed_suggestions)
                await self.send_to_app(response, save_to_redis=True)

        if result['search_queries'] and not "N/A" in result['search_queries']:
            search_results = await self.perform_searches(result['search_queries'])
            if search_results:
                merged_results = await self.merge_search_results(task.name, search_results)
                if merged_results:
                    await self.send_to_app(merged_results, save_to_redis=True)

        if result['need_visual_analysis'] and not "N/A" in result['visual_analysis_reason']:
            await self.trigger_visual_analysis(f"请分析当前场景，看是否有助于解决任务 '{task.name}'。原因：{result['visual_analysis_reason']}")

        await self.save_tasks_to_redis()

    async def confirm_suggestions(self, task_name: str, suggestions: List[str]) -> List[str]:
        system_prompt = f"""
        You are an AI assistant tasked with confirming whether suggestions for a task are worth sending to the user.
        Task: {task_name}

        For each suggestion, determine if it's valuable, relevant, and not redundant with previous conversations.
        Respond with whether suggestions are necessary and if so, list the confirmed suggestions.
        Content should be concise, informative, and not repeat information from previous conversations.

        Respond in the following format:
        is_suggestions_necessary: [Yes/No]
        necessary_suggestions: List of confirmed suggestions, separated by commas, or "N/A" if not necessary
        """

        suggestions_text = "\n".join(f"- {suggestion}" for suggestion in suggestions)
        self.task_llm.clear_history()
        additional_user_message = Message(role=MessageRole.user, content=f"Suggestions:\n{suggestions_text}\n\nPlease respond in the format required by the system prompt: is_suggestions_necessary: , necessary_suggestions: separated by commas, or 'N/A' if not necessary.")
        response = await self.task_llm.generate_text(self._user, model=config.llm.model, strict_mode=False, use_redis_history=True, system_prompt=system_prompt, addtional_user_message=additional_user_message)

        is_necessary = False
        confirmed_suggestions = []

        for line in response.strip().split('\n'):
            if line.startswith("is_suggestions_necessary:"):
                is_necessary = line.split(':', 1)[1].strip().lower() == "yes"
            elif line.startswith("necessary_suggestions:"):
                suggestions_str = line.split(':', 1)[1].strip()
                if suggestions_str.lower() != "n/a":
                    confirmed_suggestions = [s.strip() for s in suggestions_str.split(',')]

        return confirmed_suggestions if is_necessary else []

    async def perform_searches(self, queries: List[str]) -> Dict[str, str]:
        search_results = {}
        for query in queries:
            result = await search(query)
            if result:
                search_results[query] = result
        return search_results

    async def merge_search_results(self, task_name: str, search_results: Dict[str, str]) -> str:
        system_prompt = f"""
        You are an AI assistant tasked with merging and summarizing search results for a task.
        Task: {task_name}

        Analyze the search results, merge relevant information, and create a concise summary.
        Determine if the merged information is valuable enough to send to the user.

        Respond in the following format:
        is_valuable: [Yes/No]
        merged_summary: 
        [Your merged and summarized content. This can be multiple lines if needed. Content should be concise and informative.]

        Ensure that 'is_valuable:' and 'merged_summary:' are on separate lines.
        """

        search_text = "\n\n".join(f"Query: {query}\nResults: {results}" for query, results in search_results.items())
        self.task_llm.clear_history()
        self.task_llm.add_history(Message(role=MessageRole.system, content=system_prompt))
        self.task_llm.add_history(Message(role=MessageRole.user, content=f"Search Results:\n{search_text}"))

        response = await self.task_llm.generate_text(self._user, model=config.llm.model, strict_mode=False, use_redis_history=False, system_prompt=system_prompt)

        is_valuable = False
        merged_summary = []
        current_section = ""

        for line in response.strip().split('\n'):
            if line.startswith("is_valuable:"):
                is_valuable = line.split(':', 1)[1].strip().lower() == "yes"
            elif line.startswith("merged_summary:"):
                current_section = "merged_summary"
            elif current_section == "merged_summary":
                merged_summary.append(line.strip())

        merged_summary = "\n".join(merged_summary).strip()

        return merged_summary if is_valuable else ""

