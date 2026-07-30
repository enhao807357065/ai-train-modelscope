"""
意图识别模型评测脚本
用验证集跑模型，输出 Accuracy / 分类报告 / 混淆矩阵 / Bad Case

用法：
    python eval_intent.py

依赖：
    pip install openai scikit-learn
"""

import json
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === 配置 ===
# 改成你本地 vllm 服务地址
API_BASE = "http://localhost:8000/v1"
API_KEY = "empty"
MODEL = "./output/Qwen2.5-1.5B-Instruct/v0-20260729-181121/checkpoint-54-merged"  # curl http://localhost:8000/v1/models 查看

VAL_FILE = "datasets/intent/intent_val.jsonl"
SYSTEM_PROMPT = "你是意图路由助手。根据用户消息判断意图类别，只输出标签。可选标签：chitchat, product, technical, complaint, other"
VALID_LABELS = ["chitchat", "product", "technical", "complaint", "other"]


def load_val_data(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item["messages"][1]["content"]
            label = item["messages"][2]["content"]
            items.append((query, label))
    return items


def predict(client, query):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def main():
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    val_data = load_val_data(VAL_FILE)
    print(f"验证集: {len(val_data)} 条\n")

    y_true = []
    y_pred = []
    bad_cases = []

    for i, (query, true_label) in enumerate(val_data):
        pred_label = predict(client, query)

        # 容错：模型可能输出带空格或大小写不一致
        pred_clean = pred_label.lower().strip()
        if pred_clean not in VALID_LABELS:
            pred_clean = "INVALID"

        y_true.append(true_label)
        y_pred.append(pred_clean)

        status = "✓" if pred_clean == true_label else "✗"
        if pred_clean != true_label:
            bad_cases.append((query, true_label, pred_clean))

        print(f"  [{i+1:02d}] {status} query=\"{query[:30]}...\" 真实={true_label} 预测={pred_clean}")

    # === 评测结果 ===
    print(f"\n{'='*60}")
    print(f"总体准确率: {accuracy_score(y_true, y_pred):.4f} ({sum(1 for a,b in zip(y_true,y_pred) if a==b)}/{len(y_true)})")
    print(f"{'='*60}")

    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, labels=VALID_LABELS, digits=4))

    print(f"混淆矩阵 (行=真实, 列=预测):")
    print(f"{'':12s} {'  '.join(f'{l:>8s}' for l in VALID_LABELS)}")
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    for i, row in enumerate(cm):
        print(f"{VALID_LABELS[i]:12s} {'  '.join(f'{v:>8d}' for v in row)}")

    if bad_cases:
        print(f"\n{'='*60}")
        print(f"Bad Cases ({len(bad_cases)} 条):")
        print(f"{'='*60}")
        for query, true_l, pred_l in bad_cases:
            print(f"  query: {query}")
            print(f"  真实: {true_l} → 预测: {pred_l}")
            print()


if __name__ == "__main__":
    main()
