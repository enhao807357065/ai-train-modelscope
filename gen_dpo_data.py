"""
gen_dpo_data.py - 生成 DPO 偏好对数据
用法: python gen_dpo_data.py --num 100 --output data/dpo_train.jsonl --append
"""

import argparse
import json
import hashlib
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://ai-service.tal.com/openai-compatible/v1",
    api_key="*"
)

SYSTEM_PROMPT = "你是一名专业的在线教育平台客服，负责解答学员和家长的咨询。\n你的职责包括：课程介绍、报名流程指引、学习问题答疑、退费政策说明、技术问题排查。\n要求：态度友善专业，回答简洁清晰，必要时主动询问以精准定位问题。"

GENERATE_PROMPT = """你是一个数据生成助手。请生成 DPO（直接偏好优化）训练所需的偏好对数据。

场景：在线教育平台客服

请生成一组数据，包含：
1. 一段客服对话上下文（1-3轮 user/assistant 交互作为历史）
2. 用户的最新提问
3. chosen（更好的回答）：专业、热情、信息完整、主动提供解决方案
4. rejected（更差的回答）：可能存在以下问题之一：
   - 态度冷漠/敷衍
   - 信息缺失（缺关键细节如价格、时间、步骤）
   - 推诿（让用户自己去查/找别人）
   - 过于简短（只有几个字）
   - 答非所问
   - 编造不存在的政策或信息

请按以下 JSON 格式输出，不要输出其他内容：
```json
{
  "context": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "question": "用户最新的问题",
  "chosen": "更好的客服回答",
  "rejected": "更差的客服回答",
  "reject_reason": "差的原因（态度差/信息缺失/推诿/过于简短/答非所问/编造信息）"
}
```

要求：
- 场景多样化：涵盖报名咨询、退费、投诉、技术问题、课程变更、优惠活动等
- chosen 和 rejected 差异明显但都要像真人写的（rejected 不能太离谱）
- context 可以为空数组（表示用户第一句话）
- 不要出现真实品牌名（学而思、新东方、作业帮等）
- 对话用中文"""


def generate_one_batch(batch_size=5):
    """生成一批偏好对数据"""
    prompt = GENERATE_PROMPT + f"\n\n请一次生成 {batch_size} 条数据，用 JSON 数组格式输出。"

    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=4096
    )

    content = resp.choices[0].message.content.strip()
    # 提取 JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    items = json.loads(content)
    if isinstance(items, dict):
        items = [items]
    return items


def to_dpo_format(item):
    """转换为 swift DPO 训练格式

    swift 4.4.x 要求：chosen response 放在 messages 最后一条，
    rejected 用 rejected_response 字段（字符串）表示。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 添加历史上下文
    if item.get("context"):
        messages.extend(item["context"])

    # 添加用户最新问题
    messages.append({"role": "user", "content": item["question"]})
    # chosen response 作为 messages 最后一条
    messages.append({"role": "assistant", "content": item["chosen"]})

    return {
        "messages": messages,
        "rejected_response": item["rejected"],
        "reject_reason": item.get("reject_reason", "")
    }


def dedup_key(item):
    """用 messages 做去重 key（messages 最后一条即 chosen response）"""
    raw = json.dumps(item["messages"], ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()


def load_existing(path):
    """加载已有数据的去重 key"""
    keys = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    keys.add(dedup_key(item))
    return keys


def main():
    parser = argparse.ArgumentParser(description="生成 DPO 偏好对数据")
    parser.add_argument("--num", type=int, default=100, help="生成数量")
    parser.add_argument("--output", type=str, default="data/dpo_train.jsonl", help="输出文件路径")
    parser.add_argument("--append", action="store_true", help="追加模式（去重）")
    parser.add_argument("--batch_size", type=int, default=5, help="每次请求生成几条")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 加载已有数据用于去重
    existing_keys = load_existing(args.output) if args.append else set()
    print(f"已有数据: {len(existing_keys)} 条")

    mode = "a" if args.append else "w"
    generated = 0
    duplicates = 0
    errors = 0

    with open(args.output, mode, encoding="utf-8") as f:
        while generated < args.num:
            try:
                batch = generate_one_batch(args.batch_size)
                for item in batch:
                    if generated >= args.num:
                        break
                    dpo_item = to_dpo_format(item)
                    key = dedup_key(dpo_item)
                    if key in existing_keys:
                        duplicates += 1
                        continue
                    existing_keys.add(key)
                    f.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
                    generated += 1

                print(f"进度: {generated}/{args.num} (重复: {duplicates}, 错误: {errors})")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                errors += 1
                print(f"解析错误，跳过: {e}")
                continue

    print(f"\n完成！生成 {generated} 条，去重跳过 {duplicates} 条，错误 {errors} 次")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
