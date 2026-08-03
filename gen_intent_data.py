"""
意图识别数据集生成脚本
使用 LLM 批量生成 5 类意图的训练数据，输出为 swift 可用的 messages 格式 jsonl。
生成后需要人工审核！

用法：
    python gen_intent_data.py

输出：
    data/intent_raw.jsonl    - 原始生成数据（需审核）
"""

import json
import os
import time
from openai import OpenAI

# === 配置 ===
# 改成你的 LLM 后端
API_BASE = "http://ai-service.tal.com/openai-compatible/v1"
API_KEY = os.environ.get("LLM_API_KEY", "*")
MODEL = "deepseek-v4-flash"  # 用好一点的模型生成数据

# 每个标签生成多少条
NUM_PER_LABEL = 100

# 输出路径
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "intent_raw.jsonl")

# System prompt（和训练/推理时一致）
SYSTEM_PROMPT = (
    "你是意图路由助手。根据用户消息判断意图类别，只输出标签。"
    "可选标签：chitchat, product, technical, complaint, other"
)

# === 标签定义 + 生成 prompt ===
LABEL_PROMPTS = {
    "chitchat": """请生成 {n} 条用户闲聊消息。要求：
1. 像真实用户在对话框里随手打的，口语化
2. 包含：打招呼、寒暄、闲扯、无明确意图的话
3. 长短混合（有的就1-3个字，有的一句话）
4. 多样化：不要重复句式
5. 不涉及产品/技术/投诉内容
6. 每条至少 3 个字（不要出现单字如"嗯""哦""对"）
7. 不要以"请问""你好"开头的超过一半，注意开头多样性

示例风格：
- 你好呀
- 在吗？
- 今天好累啊
- 哈哈哈哈
- 周末有什么推荐的电影吗
- 吃饭了吗
- 好无聊啊""",

    "product": """请生成 {n} 条用户咨询产品的消息。场景：用户在使用一个 SaaS/互联网产品，想了解功能、用法、价格、对比等。要求：
1. 口语化，像真实用户提问
2. 覆盖：功能咨询、使用方法、价格询问、版本区别、是否支持某能力
3. 长短混合（最短不少于 5 字，最长不超过 50 字）
4. 不涉及技术实现细节（那属于 technical）
5. 不带投诉情绪（不能有"为什么还没""怎么这么差"之类的负面表达）
6. 句式多样化：疑问句、陈述句、祈使句都要有
7. 不要所有句子都以"你们"开头

示例风格：
- 你们支持导出 PDF 吗
- 免费版和付费版有什么区别
- 怎么邀请团队成员
- 能对接企业微信吗
- 一个月多少钱
- 我想了解下批量导入功能
- 有没有移动端 app""",

    "technical": """请生成 {n} 条用户咨询技术问题的消息。场景：用户是开发者，在集成/使用某产品的 API 或技术组件时遇到问题。要求：
1. 口语化但带技术术语
2. 覆盖：API 调用问题、报错排查、集成方式、SDK 用法、性能问题、配置问题
3. 可以包含错误码、HTTP 状态码、具体技术名词
4. 长短混合（最短不少于 5 字，最长不超过 80 字）
5. 不带投诉情绪（纯技术求助，不要"什么破接口""垃圾文档"之类）
6. 不要所有句子都是疑问句，可以有陈述式的求助（"我遇到了xxx问题"）
7. 涵盖多种技术栈：不要全是 HTTP/REST，也包括 WebSocket、SDK、数据库、部署等

示例风格：
- 接口返回 401 是什么原因
- Python SDK 支持异步调用吗
- webhook 回调没收到，怎么排查
- 请求超时了，有重试机制吗
- 怎么配置 SSO 单点登录
- 批量导入接口有并发限制吗
- 我在对接 WebSocket 推送的时候连接老断""",

    "complaint": """请生成 {n} 条用户投诉/抱怨的消息。场景：用户对产品或服务不满意，带有负面情绪。要求：
1. 口语化，情绪真实（有轻微不满的、也有很生气的）
2. 覆盖：功能 bug 投诉、服务态度差、响应慢、催促处理、要求退款、威胁差评
3. 长短混合（最短不少于 4 字，最长不超过 80 字）
4. 关键特征：带有负面情绪 + 明确或隐含的诉求
5. 不是纯技术求助（那属于 technical）—— 如果只是报错不带情绪就不算投诉
6. 情绪程度要有梯度：从"有点不满"到"很生气"都要覆盖
7. 不要全是脏话/极端用语，也要有冷静但坚决的投诉

示例风格：
- 什么破产品，用着用着就崩了
- 都三天了还没人回复我，你们客服是摆设吗
- 我要退款
- 这个 bug 反馈了一个月了还没修
- 再不解决我就去投诉了
- 体验越来越差了，考虑换别家了
- 能不能上点心？每次都要催好几遍""",

    "other": """请生成 {n} 条不属于闲聊、产品咨询、技术问题、投诉的用户消息。要求：
1. 口语化
2. 覆盖：离题请求（写作文、翻译、算数学题）、无法归类的模糊消息、测试消息、误发消息
3. 长短混合（最短不少于 2 字，最长不超过 50 字）
4. 这些是"兜底"类别，模型遇到这类应该不强行分到前4类
5. 不要超过 20% 是纯乱码/测试消息，大部分应该是"合理但不属于前4类"的请求
6. 不要和前面 4 个标签有歧义（比如"帮我查下物流"可能是产品咨询，不要放这里）

示例风格：
- 帮我翻译一下这段英文
- 1+1等于几
- test123
- 今天几号了
- 帮我写一封请假邮件
- 给我讲个笑话
- 推荐几首歌""",
}


def generate_samples(client: OpenAI, label: str, n: int) -> list[str]:
    """调用 LLM 生成指定标签的样本"""
    prompt = LABEL_PROMPTS[label].format(n=n)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是数据标注专家，负责生成高质量的训练数据。只输出数据本身，每行一条，不要编号、不要解释、不要 markdown 格式。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,  # 高一点增加多样性
        max_tokens=4096,
    )

    text = response.choices[0].message.content.strip()
    # 按行分割，去掉空行和可能的编号前缀
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉常见的编号前缀：1. / 1、/ - / * 
        import re
        line = re.sub(r'^[\d]+[.、)\]]\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def build_message_item(query: str, label: str) -> dict:
    """构建 swift messages 格式的单条数据"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": label},
        ]
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    all_items = []

    for label in LABEL_PROMPTS:
        print(f"\n{'='*50}")
        print(f"生成 [{label}] 标签数据（目标 {NUM_PER_LABEL} 条）...")
        print(f"{'='*50}")

        samples = generate_samples(client, label, NUM_PER_LABEL)
        print(f"  实际生成: {len(samples)} 条")

        # 如果生成不够，再补一轮
        if len(samples) < NUM_PER_LABEL:
            print(f"  不够 {NUM_PER_LABEL} 条，补充生成中...")
            extra = generate_samples(client, label, NUM_PER_LABEL - len(samples))
            samples.extend(extra)
            print(f"  补充后: {len(samples)} 条")

        # 去重
        samples = list(dict.fromkeys(samples))  # 保持顺序去重
        print(f"  去重后: {len(samples)} 条")

        # 构建 messages 格式
        for query in samples[:NUM_PER_LABEL]:  # 最多取 NUM_PER_LABEL 条
            all_items.append(build_message_item(query, label))

        # 打印几条样例
        print(f"  样例:")
        for s in samples[:3]:
            print(f"    - {s}")

        time.sleep(1)  # 避免 rate limit

    # 写出 jsonl
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"完成！共 {len(all_items)} 条数据")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"{'='*50}")
    print(f"\n⚠️  下一步：人工审核 {OUTPUT_FILE}")
    print(f"   - 检查每条是否标签正确")
    print(f"   - 删除不自然/重复/有歧义的")
    print(f"   - 特别关注边界 case")


if __name__ == "__main__":
    main()
