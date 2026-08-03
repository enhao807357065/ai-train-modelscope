from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")
resp = client.chat.completions.create(
    model="output/final-merged",
    messages=[{"role": "system", "content": "you are a helpful assistant"},
              {"role": "user", "content": "我找不到我的课程在什么地方"}],
    temperature=0
)
print(f"\n response: {resp}")
print(f"\n response.choices[0].message.content: {resp.choices[0].message.content}")