from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="empty")
resp = client.chat.completions.create(
    model="./output/v4-20260727-171700/checkpoint-282-merged",
    messages=[{"role": "system", "content": "you are a helpful assistant"},
              {"role": "user", "content": "快递能不能退？"}],
    temperature=0
)
print(f"\n response: {resp}")
print(f"\n response.choices[0].message.content: {resp.choices[0].message.content}")