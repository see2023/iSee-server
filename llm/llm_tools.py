import os
import sys
import logging
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from typing import List, Optional

from langchain_core.tools import Tool
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from common.search import search

async def calculate_math(input: str) -> str:
    try:
        await asyncio.sleep(0.001)
        result = eval(input)
        return str(result)
    except:
        logging.error("Invalid input. Please enter a valid mathematical expression.")
        return ""


class ToolActions:
    LLM = "LLM"
    VLLM = "VLLM"
    PASSTHROUGH = "PASSTHROUGH"

class ToolNames:
    SEARCH = "Search"
    MATH_CALCULATOR = "MathCalculator"
    TAKE_PHOTO = "TakePhoto"

    @staticmethod
    def all() -> List[str]:
        return [ToolNames.SEARCH, ToolNames.MATH_CALCULATOR, ToolNames.TAKE_PHOTO]
    
switch = {
    ToolNames.SEARCH: ToolActions.LLM,
    ToolNames.MATH_CALCULATOR: ToolActions.PASSTHROUGH,
    ToolNames.TAKE_PHOTO: ToolActions.VLLM,
}

# async function to search bing with a single search term using the bing_web_search_with_page_async function
SearchTool = Tool(
    name = ToolNames.SEARCH,
    func=None,
    coroutine=search,
    description="useful for when you need to answer questions about current events or the current state of the world. The input to this should be a single short and simple search term: ['shanghai news']",
)

MathCalculatorTool = Tool(
    name=ToolNames.MATH_CALCULATOR,
    # func=calculate_math,
    func=None,
    coroutine=calculate_math,
    description="Useful for calculating mathematical expressions. The input to this should be a valid mathematical expression: ['1+2/3']",
)


TakePhotoTool = Tool(
    name=ToolNames.TAKE_PHOTO,
    func=None,
    description="Called when the other person needs to show something, or you need to see a live scene. No input is required for this tool.",
)

class Tools:
    def __init__(self):
        self.tools = [SearchTool, MathCalculatorTool, TakePhotoTool]
        self.openai_functions =[convert_to_openai_function(t) for t in self.tools]
        self.sinple_functions = [ {"name": t.name, "description": t.description} for t in self.tools]

    def get_tool(self, name: str) -> Tool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    # 获tool输出后的下一步处理方式
    def get_next_step(self, name: str) -> Optional[str]:
        return switch.get(name, None)
    
    def set_coroutine(self, name: str, coroutine):
        tool = self.get_tool(name)
        if tool is None:
            logging.error(f"Tool {name} not found.")
            return None
        tool.coroutine = coroutine
    
    # input 输入变量序列 List[any]
    async def call_tool(self, name: str, input: str):
        tool = self.get_tool(name)
        if tool is None:
            logging.error(f"Tool {name} not found.")
            return None
        if tool.coroutine is None:
            logging.error(f"Tool {name} has no coroutine.")
            return None
        return await tool.coroutine(input)
    
    def get_functions_openai_style(self):
        return self.openai_functions

    def get_functions_simple_style(self):
        return self.sinple_functions


async def function_test():

    query = "从1加到10等于几？"
    import openai
    tools = [MathCalculatorTool]
    functions = [convert_to_openai_function(t) for t in tools]
    messages = [
        {"role": "system", "content": "Your responses should be concise and factual.。"},
        {"role": "user", "content": query}
    ]

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        n=1,
        stream=False,
        functions=functions,
        messages=messages,
    )
    choice = completion.choices[0]
    if choice.finish_reason == "function_call":
        result = choice.function_call.result
        logging.info(f"Result: {result}")
    else:
        # print content
        logging.info(f"Result: {choice.message.content}")
    '''
    return value:
        Choice(finish_reason='function_call', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', 
            function_call=FunctionCall(arguments='{"__arg1":"1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10"}', name='MathCalculator'), tool_calls=None))
    '''

 

    # from langchain_openai import ChatOpenAI
    # from langchain_core.messages import HumanMessage
    # model = ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)
    # message = model.invoke([HumanMessage(content=query)], functions=functions)
    # logging.info(message)

async def tool_test():
    tools = Tools()
    query = "松江天气"
    logging.info(tools.get_functions_openai_style())
    logging.info(tools.get_functions_simple_style())
    out = await MathCalculatorTool.ainvoke("1+2/3")
    logging.info(out)
    out = await SearchTool.ainvoke(query)
    logging.info(out)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    # asyncio.run(function_test())
    asyncio.run(tool_test())
