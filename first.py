import torch
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 这个文件就是 transformer 推理的最小完整流程，可以对应到四个阶段
# 加载模型/分词器
#       ↓
#   构造 Chat 格式（apply_chat_template，Chat 模型专有）
#       ↓
#   输入编码（tokenizer → token id 张量）
#       ↓
#   模型推理（model.generate，自回归逐 token 生成）
#       ↓
#   截去输入部分，只保留新生成的 token
#       ↓
#   输出解码（batch_decode → 可读文本）
#
#   这个流程是所有 HuggingFace/ModelScope 上 transformer 推理代码的骨架，后续无论是加 RAG、加 LoRA 微调、加
#   vLLM 加速，本质上都是在这个骨架的各个环节上做扩展。

# 打印 PyTorch 版本、CUDA 是否可用、CUDA 版本，用于环境验证
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

# 从环境变量读取 ModelScope 模型缓存目录
modelscope_path = os.getenv("MODELSCOPE_CACHE")
print(f"\n modelscope_path: {modelscope_path}")

model_name = "Qwen/Qwen3.5-2B"

# 加载因果语言模型
# model_name: HuggingFace/ModelScope 上的模型标识符
# torch_dtype="auto": 自动选择精度（BF16/FP16/FP32）
# device_map="auto": 自动将模型层分配到可用 GPU/CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# 加载对应的分词器，用于文本与 token id 之间的转换
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

prompt = "你好，我叫张飞"

# 构造对话消息列表，遵循 ChatML 格式
# role: "system" 设定模型角色，"user" 为用户输入
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 将消息列表转换为模型可接受的单条字符串
# tokenize=False: 只做模板渲染，不做 token 编码
# add_generation_prompt=True: 在末尾追加触发模型回复的特殊前缀
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 对文本进行编码并移至模型所在设备（GPU/CPU）
# return_tensors="pt": 返回 PyTorch 张量
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 执行自回归生成
# max_new_tokens: 限制最多生成 512 个新 token，避免无限输出
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 从生成结果中截去输入部分，只保留新生成的 token id
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 将 token id 解码回可读文本，skip_special_tokens=True 去除特殊标记
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\n type: {type(response)}, response: {response}")
