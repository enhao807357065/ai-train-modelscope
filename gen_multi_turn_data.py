"""
教育客服多轮对话数据生成脚本
使用 LLM 批量生成多轮对话训练数据，输出为 swift 可用的 messages 格式 jsonl。

用法：
    python gen_multi_turn_data.py --total 100 --min_turns 3 --max_turns 6

    # 追加模式（在已有数据基础上补充，自动去重）
    python gen_multi_turn_data.py --total 200 --append

参数：
    --total       生成对话总条数（默认 100）
    --min_turns   最少轮数，1轮=1组user+assistant（默认 3）
    --max_turns   最多轮数（默认 6）
    --output      输出文件路径（默认 data/edu_customer_service.jsonl）
    --batch_size  每次 API 调用生成几条对话（默认 3，太多质量下降）
    --append      追加模式：保留已有数据，新数据追加后统一去重
    --dedup_threshold  去重相似度阈值（默认 0.7，越低越严格）

输出：
    每行一个 JSON 对象，格式：
    {"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]}
"""

import argparse
import json
import os
import random
import time
from difflib import SequenceMatcher
from openai import OpenAI

# === 配置 ===
API_BASE = "http://ai-service.tal.com/openai-compatible/v1"
API_KEY = os.environ.get("LLM_API_KEY", "*")
MODEL = "deepseek-v4-flash"

# 教育客服的 system prompt（训练时模型要学的角色）
SYSTEM_PROMPT = """你是一名专业的在线教育平台客服，负责解答学员和家长的咨询。
你的职责包括：课程介绍、报名流程指引、学习问题答疑、退费政策说明、技术问题排查。
要求：态度友善专业，回答简洁清晰，必要时主动询问以精准定位问题。"""

# === 场景定义 ===
# 每个场景包含：描述、示例话题、期望的对话风格
SCENARIOS = [
    {
        "name": "课程咨询",
        "description": "家长或学员咨询课程信息，包括课程内容、适合年龄、师资、上课时间等",
        "topics": [
            "小学数学思维课适合几年级的孩子",
            "英语口语课和阅读课有什么区别",
            "暑假班什么时候开课",
            "老师是什么背景",
            "线上课和线下课的区别",
            "有没有试听课",
            "一个班多少人",
            "课程体系是怎么设计的",
        ],
    },
    {
        "name": "报名与支付",
        "description": "用户想报名、了解价格、支付方式、优惠活动等",
        "topics": [
            "怎么报名",
            "有什么优惠活动吗",
            "可以分期付款吗",
            "报名后多久开始上课",
            "老学员续费有折扣吗",
            "两个孩子一起报有团购价吗",
            "支持哪些支付方式",
            "报错课了能换吗",
        ],
    },
    {
        "name": "学习问题",
        "description": "学员在学习过程中遇到的问题，如跟不上进度、作业不会做、需要请假等",
        "topics": [
            "孩子跟不上课程进度怎么办",
            "作业太难了不会做",
            "想请假两周可以补课吗",
            "录播回放在哪里看",
            "学习报告在哪里查看",
            "能不能换个班级",
            "孩子不想上课了怎么办",
            "课后有答疑服务吗",
        ],
    },
    {
        "name": "退费与投诉",
        "description": "用户不满意要退费，或对服务有投诉，情绪可能比较激动",
        "topics": [
            "上了几节课不满意想退费",
            "退费流程是什么",
            "为什么退费要扣手续费",
            "课程质量太差了",
            "老师上课迟到好几次了",
            "承诺的服务没有兑现",
            "申请退费好几天了还没处理",
            "我要投诉你们的课程顾问",
        ],
    },
    {
        "name": "技术问题",
        "description": "用户在使用平台（APP/网站）时遇到技术问题",
        "topics": [
            "APP 打不开了",
            "视频卡顿加载不出来",
            "登录不上去怎么办",
            "课程视频没有声音",
            "怎么切换到横屏模式",
            "下载的课件打不开",
            "摄像头打不开老师看不到我",
            "系统提示课程已过期但我明明还在有效期内",
        ],
    },
]

# 生成 prompt 模板
GENERATE_PROMPT = """请你模拟一段真实的在线教育平台客服对话。

【场景】{scenario_name}：{scenario_desc}
【话题方向】{topic}
【要求】
1. 对话轮数：恰好 {num_turns} 轮（1轮 = 用户说一句 + 客服回一句）
2. 用户角色：学员家长或学员本人，口语化、自然
3. 客服角色：专业、友善、简洁，必要时主动追问
4. 对话要自然流畅，有上下文关联（后面的话要接前面的）
5. 不要出现"好的还有什么可以帮您的"这种强行结尾
6. 用户的问题要有层次递进，不要每轮都问无关的问题
7. 客服回答要有实质内容，不要全是套话

【输出格式】
严格按以下 JSON 格式输出，不要加任何其他文字：
{{"conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

注意：conversation 数组应该有 {total_messages} 个元素（{num_turns} 轮 × 2）。"""

GENERATE_BATCH_PROMPT = """请你模拟 {batch_size} 段真实的在线教育平台客服对话。

【场景】{scenario_name}：{scenario_desc}
【话题方向参考】{topics}
【要求】
1. 每段对话轮数：{num_turns} 轮（1轮 = 用户说一句 + 客服回一句）
2. 用户角色：学员家长或学员本人，口语化、自然
3. 客服角色：专业、友善、简洁，必要时主动追问
4. 对话要自然流畅，有上下文关联（后面的话要接前面的）
5. 不要出现"好的还有什么可以帮您的"这种强行结尾
6. 用户的问题要有层次递进，不要每轮都问无关的问题
7. 客服回答要有实质内容，不要全是套话
8. {batch_size} 段对话之间要有差异，不要雷同

【输出格式】
严格按以下 JSON 格式输出，不要加任何其他文字：
{{"conversations": [{{"conversation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}, ...]}}

每个 conversation 数组应该有 {total_messages} 个元素（{num_turns} 轮 × 2）。共 {batch_size} 段对话。"""


def parse_args():
    parser = argparse.ArgumentParser(description="生成教育客服多轮对话数据")
    parser.add_argument("--total", type=int, default=100, help="生成对话总条数")
    parser.add_argument("--min_turns", type=int, default=3, help="最少轮数")
    parser.add_argument("--max_turns", type=int, default=6, help="最多轮数")
    parser.add_argument("--output", type=str, default="data/edu_customer_service.jsonl", help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=3, help="每次 API 调用生成几条")
    parser.add_argument("--append", action="store_true", help="追加模式（不覆盖已有数据，自动对新旧数据去重）")
    parser.add_argument("--dedup_threshold", type=float, default=0.7, help="去重相似度阈值（0~1，默认0.7）")
    return parser.parse_args()


def extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON，容忍 markdown 包裹"""
    text = text.strip()
    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```json 和末行 ```
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def validate_conversation(conv: list, expected_turns: int) -> bool:
    """验证对话格式是否正确"""
    if not isinstance(conv, list):
        return False
    if len(conv) < 2:  # 至少1轮
        return False
    # 检查交替格式：user, assistant, user, assistant...
    for i, msg in enumerate(conv):
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            return False
        if not msg["content"].strip():
            return False
    return True


def generate_single(client: OpenAI, scenario: dict, num_turns: int) -> list | None:
    """生成单条对话"""
    topic = random.choice(scenario["topics"])
    prompt = GENERATE_PROMPT.format(
        scenario_name=scenario["name"],
        scenario_desc=scenario["description"],
        topic=topic,
        num_turns=num_turns,
        total_messages=num_turns * 2,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是对话数据生成专家。严格按要求的 JSON 格式输出，不要输出其他内容。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85,
            max_tokens=4096,
        )
        result = extract_json(response.choices[0].message.content)
        if result and "conversation" in result:
            conv = result["conversation"]
            if validate_conversation(conv, num_turns):
                return conv
    except Exception as e:
        print(f"  [错误] API 调用失败: {e}")
    return None


def generate_batch(client: OpenAI, scenario: dict, num_turns: int, batch_size: int) -> list[list]:
    """批量生成对话"""
    topics = random.sample(scenario["topics"], min(batch_size, len(scenario["topics"])))
    prompt = GENERATE_BATCH_PROMPT.format(
        scenario_name=scenario["name"],
        scenario_desc=scenario["description"],
        topics="、".join(topics),
        num_turns=num_turns,
        total_messages=num_turns * 2,
        batch_size=batch_size,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是对话数据生成专家。严格按要求的 JSON 格式输出，不要输出其他内容。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85,
            max_tokens=8192,
        )
        result = extract_json(response.choices[0].message.content)
        if result and "conversations" in result:
            valid = []
            for item in result["conversations"]:
                conv = item.get("conversation", item) if isinstance(item, dict) else item
                if validate_conversation(conv, num_turns):
                    valid.append(conv)
            return valid
    except Exception as e:
        print(f"  [错误] API 调用失败: {e}")
    return []


def build_messages_item(conversation: list) -> dict:
    """构建 swift messages 格式"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation)
    return {"messages": messages}


def get_first_user_message(item: dict) -> str:
    """提取对话中第一条用户消息（用于去重比较）"""
    for msg in item["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def deduplicate(items: list, threshold: float = 0.7) -> tuple[list, int]:
    """基于首轮用户消息相似度去重，返回 (去重后列表, 删除数量)"""
    if not items:
        return items, 0

    first_messages = [get_first_user_message(item) for item in items]
    keep_mask = [True] * len(items)

    for i in range(len(items)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(items)):
            if not keep_mask[j]:
                continue
            ratio = SequenceMatcher(None, first_messages[i], first_messages[j]).ratio()
            if ratio > threshold:
                keep_mask[j] = False

    result = [item for item, keep in zip(items, keep_mask) if keep]
    removed = len(items) - len(result)
    return result, removed


def main():
    args = parse_args()

    print(f"""
{'='*60}
教育客服多轮对话数据生成
{'='*60}
目标总数: {args.total} 条对话
轮数范围: {args.min_turns} ~ {args.max_turns} 轮
批次大小: {args.batch_size}
输出文件: {args.output}
模型: {MODEL}
{'='*60}
""")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # === 追加模式：加载已有数据 ===
    existing_items = []
    if args.append and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_items.append(json.loads(line))
        print(f"追加模式：已加载 {len(existing_items)} 条已有数据")

    all_items = []
    failed_count = 0
    generated = 0

    while generated < args.total:
        # 随机选场景和轮数
        scenario = random.choice(SCENARIOS)
        num_turns = random.randint(args.min_turns, args.max_turns)
        remaining = args.total - generated
        current_batch = min(args.batch_size, remaining)

        print(f"[{generated + 1}/{args.total}] 场景: {scenario['name']}, "
              f"轮数: {num_turns}, 批次: {current_batch}")

        if current_batch == 1:
            conv = generate_single(client, scenario, num_turns)
            if conv:
                all_items.append(build_messages_item(conv))
                generated += 1
                print(f"  ✓ 生成成功 ({len(conv)//2} 轮)")
            else:
                failed_count += 1
                print(f"  ✗ 生成失败，跳过")
        else:
            convs = generate_batch(client, scenario, num_turns, current_batch)
            if convs:
                for conv in convs:
                    if generated >= args.total:
                        break
                    all_items.append(build_messages_item(conv))
                    generated += 1
                print(f"  ✓ 批量生成成功 {len(convs)} 条")
            else:
                failed_count += 1
                print(f"  ✗ 批量生成失败，回退到逐条生成")
                # fallback: 逐条生成
                conv = generate_single(client, scenario, num_turns)
                if conv:
                    all_items.append(build_messages_item(conv))
                    generated += 1

        # 避免 rate limit
        time.sleep(0.5)

        # 安全阀：失败太多就停
        if failed_count > args.total * 0.3:
            print(f"\n⚠️ 失败次数过多 ({failed_count})，提前终止")
            break

    # === 去重 + 合并 ===
    if args.append and existing_items:
        # 合并新旧数据后统一去重
        combined = existing_items + all_items
        print(f"\n合并前: 已有 {len(existing_items)} + 新增 {len(all_items)} = {len(combined)} 条")
        combined, dedup_removed = deduplicate(combined, threshold=args.dedup_threshold)
        print(f"去重后: {len(combined)} 条（删除 {dedup_removed} 条相似对话）")
        all_items = combined
    else:
        # 非追加模式也做去重
        all_items, dedup_removed = deduplicate(all_items, threshold=args.dedup_threshold)
        if dedup_removed > 0:
            print(f"\n去重: 删除 {dedup_removed} 条相似对话")

    # 写出 jsonl
    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    turn_counts = {}
    for item in all_items:
        turns = (len(item["messages"]) - 1) // 2  # 减去 system
        turn_counts[turns] = turn_counts.get(turns, 0) + 1

    print(f"""
{'='*60}
生成完成！
{'='*60}
总计: {len(all_items)} 条对话
失败: {failed_count} 次
输出: {args.output}

轮数分布:""")
    for turns in sorted(turn_counts):
        count = turn_counts[turns]
        pct = count / len(all_items) * 100
        bar = "█" * int(pct / 2)
        print(f"  {turns} 轮: {count:3d} 条 ({pct:5.1f}%) {bar}")

    print(f"""
{'='*60}
⚠️  下一步：
  1. 人工抽检 {args.output}
  2. 确认对话自然度和客服回答质量
  3. 用于 swift SFT 训练
{'='*60}
""")


if __name__ == "__main__":
    main()
