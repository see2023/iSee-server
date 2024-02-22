source ~/.zshrc.plus

# curl -v https://api.openai.com/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
#     "model": "gpt-3.5-turbo",
#     "messages": 
# 	[{"role": "system", "content": "你叫桔子，今年1岁，是一个聪明、友善的助手。"}, {"role": "user", "content": "从1加到10等于几？"}],
# 	"functions":
# 	[{"name": "Search", "description": "useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term.", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}, {"name": "MathCalculator", "description": "Useful for calculating mathematical expressions. The input to this should be a valid mathematical expression.]", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}, {"name": "TakePhoto", "description": "Called when the other person needs to show something, or you need to see a live scene. No input is required for this tool.", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}]
#   }'

echo $DASHSCOPE_API_KEY

curl -v 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation' -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H 'Content-Type: application/json' -d '
{
	"model": "qwen-max", 
	"parameters": {
		"result_format": "message", 
		"incremental_output": false
	}, 
	"input": {
		"messages": [{"role": "system", "content": "你叫桔子，今年1岁，是一个聪明、友善的助手。"}, {"role": "user", "content": "你好，今天好开心啊！"}
		],
	 	"functions":
 	[{"name": "Search", "description": "useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term.", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}, {"name": "MathCalculator", "description": "Useful for calculating mathematical expressions. The input to this should be a valid mathematical expression.]", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}, {"name": "TakePhoto", "description": "Called when the other person needs to show something, or you need to see a live scene. No input is required for this tool.", "parameters": {"properties": {"__arg1": {"title": "__arg1", "type": "string"}}, "required": ["__arg1"], "type": "object"}}]
	}
}
'
