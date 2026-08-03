"""
数据集划分脚本
按标签分层抽样，划分为训练集和验证集（90:10）

用法：
    python split_dataset.py

输入：data/intent_raw.jsonl
输出：data/intent_train.jsonl + data/intent_val.jsonl
"""

import json
import random
from collections import defaultdict

# === 配置 ===
INPUT_FILE = "data/intent_raw.jsonl"
TRAIN_FILE = "data/intent_train.jsonl"
VAL_FILE = "data/intent_val.jsonl"
VAL_RATIO = 0.1  # 验证集占比
SEED = 42  # 固定随机种子，保证可复现

def main():
    random.seed(SEED)

    # 按标签分组
    by_label = defaultdict(list)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label = item["messages"][-1]["content"]
            by_label[label].append(item)

    print("原始数据分布:")
    for label, items in sorted(by_label.items()):
        print(f"  {label:12s}: {len(items)} 条")
    print(f"  {'总计':12s}: {sum(len(v) for v in by_label.values())} 条")

    # 分层抽样
    train_items = []
    val_items = []

    for label, items in by_label.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * VAL_RATIO))  # 每类至少 1 条进验证集
        val_items.extend(items[:split_idx])
        train_items.extend(items[split_idx:])

    # shuffle 训练集（验证集不需要 shuffle）
    random.shuffle(train_items)

    # 写出
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计结果
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_items)} 条 → {TRAIN_FILE}")
    print(f"  验证集: {len(val_items)} 条 → {VAL_FILE}")

    # 验证集分布
    val_labels = defaultdict(int)
    for item in val_items:
        val_labels[item["messages"][-1]["content"]] += 1
    print(f"\n验证集标签分布:")
    for label, count in sorted(val_labels.items()):
        print(f"  {label:12s}: {count} 条")


if __name__ == "__main__":
    main()
