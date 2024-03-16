import json
import os


prompt_common = '''
可以使用的工具：
    Search:  当需要实时信息时，可以用搜索引擎搜索相关内容。输入参数为搜索关键词，例如：“天气预报”
    MathCalculator: 需要执行数学计算时调用，输入参数为可在python里执行的算式，例如：“1+2+3+4+5+6+7+8+9+10”
    TakePhoto: 需要通过摄像头获取对面的图像时调用，没有参数。

输出格式为：
    {
        "Text": "回复用户的内容，如果需要使用工具，在这里对用户说明原因",
        "Tool": "工具名, 可以为空，例如：Search",
        "Args": "工具使用的参数字符串, 可以为空"
    }

示例1:
input:
    现在外面冷吗？
output：
    {
        "Text": "我需要先搜索一下. "
        "Tool": "Search"
        "Args": "天气预报。"
    }

示例2:
input：
    从一加到10等于多少？
output：
    {
        "Text": "我先算一下。"
        "Tool": "MathCalculator"
        "Args": "1+2+3+4+5+6+7+8+9+10"
    }

示例3:
input：
    你看我多大了？
output：
    {
        "Text": "我先从摄像头看一下你。",
        "Tool": "TakePhoto",
        "Args": ""
    }

示例4:
input： 我想看电影。
output： 
    {
        "Text": "哎呀，作业做完了啊，想看什么电影呢？"
        "Tool": "",
        "Args": ""
    }

'''

prompt_for_data =  """通过大语言模型给一个学生对话机器人生成指令微调数据集，输入格式任意，内容覆盖生活、学习、工作、娱乐等多个领域。
""" + prompt_common + '''现在请按照上述例子，输出10条问答对，以json格式输出。 
这次可专注需要实时照相的场景，比如: 这是什么品种的苹果？是不是很酸？ 
'''

system_prompt = "假设你是一个可以使用外部工具的智能助手，当用户咨询你问题时，如果有把握回答，请直接输出内容； 否则请给出原因、工具名和参数。"+ prompt_common + '''以下是用户的问题：'''


import random
from http import HTTPStatus
import dashscope


def get_case_by_qwen(user_message=''):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_message}]
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_max,
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0]['message']['content']
        print(content)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

def get_data_from_json_file():
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path, 'data.json')
    return json.load(open(file_path, 'r', encoding='utf-8'))

def write_data_to_qwen1_json_file(cases):
    # '''
    # [
    #   {
    #     "id": "identity_0",
    #     "conversations": [
    #       {
    #         "from": "user",
    #         "value": "你好"
    #       },
    #       {
    #         "from": "assistant",
    #         "value": "我是一个语言模型，我叫通义千问。"
    #       }
    #     ]
    #   }
    # ]
    # '''
    with open('fine_tune.json', 'w', encoding='utf-8') as f:
        cases_new_format = []
        id = 0
        for case in cases:
            if not case['input'] or not case['output']:
                continue
            outout_str = ''
            if type(case['output']) is str:
                outout_str = case['output']
            else:
                outout_str = 'Reason: ' + case['output']['Reason'] + '\nTool: ' + case['output']['Tool'] + '\nArgs: ' + ('None' if case['output']['Args'] is None else case['output']['Args'])
            new_case = {
                "id": f"identity_{id}",
                "conversations": [
                    {
                        "from": "user",
                        "value": case['input']
                    },
                    {
                        "from": "assistant",
                        "value": outout_str
                    }
                ]
            }
            cases_new_format.append(new_case)
            id += 1
        json.dump(cases_new_format, f, ensure_ascii=False, indent=4)

def write_qwen1_5_jsonl_file(cases):
#     '''
# {
#     "type": "chatml",
#     "messages": [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": "Tell me something about large language models."
#         },
#         {
#             "role": "assistant",
#             "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
#         }
#     ],
#     "source": "unknown"
# }
#     '''

    with open('fine_tune_2.jsonl', 'w', encoding='utf-8') as f:
        for case in cases:
            if not case['input'] or not case['output']:
                continue
            outout_str = ''
            if type(case['output']) is str:
                # outout_str = case['output']
                outout_str = '{"Text": "' + case['output'] + '", "Tool": "", "Args": ""}'
            else:
                # outout_str = 'Reason: ' + case['output']['Reason'] + '\nTool: ' + case['output']['Tool'] + '\nArgs: ' + ('None' if case['output']['Args'] is None else case['output']['Args'])
                outout_str = '{"Text": "' + case['output']['Reason'] + '", "Tool": "' + case['output']['Tool'] + '", "Args": "' + ('' if case['output']['Args'] is None else case['output']['Args']) + '"}'
            new_case = {
                "type": "chatml",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": case['input']
                    },
                    {
                        "role": "assistant",
                        "content": outout_str
                    }
                ],
                "source": "unknown"
            }
            json.dump(new_case, f, ensure_ascii=False)
            f.write('\n')


def print_cases(cases):
    for case in cases:
        if not case['input'] or not case['output']:
            continue
        if type(case['output']) is str:
            print(f"no tool:\n\tinput: {case['input']}\n\toutput: {case['output']}")
        else:
            print(f"tool: \n\tinput: {case['input']}\n\ttool: {case['output']['Tool']}\n\targs: {case['output']['Args']}\n\treason: {case['output']['Reason']}")


if __name__ == '__main__':
    # get_case_by_qwen(prompt)
    cases = get_data_from_json_file()
    print(len(cases))
    # write_data_to_qwen1_json_file(cases)
    write_qwen1_5_jsonl_file(cases)
