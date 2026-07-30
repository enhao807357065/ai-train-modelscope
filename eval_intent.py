"""
教育客服意图识别 - 评估脚本
用法: python eval_intent.py --result output/eval_result.jsonl

功能:
1. 整体准确率 (Accuracy)
2. 每个意图的 Precision / Recall / F1
3. 混淆矩阵（文本版）
4. 错误样本列表（方便排查）
"""

"""
**课程咨询**
• 意图标签: course_inquiry
• 典型用户话术: "你们有什么课程"、"适合几岁孩子"、"课程怎么安排的"

**价格咨询**
• 意图标签: price_inquiry
• 典型用户话术: "多少钱"、"有没有优惠"、"能不能打折"

**试听预约**
• 意图标签: trial_booking
• 典型用户话术: "能试听吗"、"预约一节体验课"、"想先试试看"

**退费/退课**
• 意图标签: refund_request
• 典型用户话术: "我要退款"、"不想学了退费"、"课程不满意怎么退"

**排课/调课**
• 意图标签: schedule_change
• 典型用户话术: "能换个时间吗"、"这周请假"、"想调到周末"

**上课问题**
• 意图标签: class_issue
• 典型用户话术: "进不去教室"、"老师没来"、"视频卡顿"、"链接打不开"

**教师相关**
• 意图标签: teacher_inquiry
• 典型用户话术: "老师是哪里的"、"能换老师吗"、"老师教得怎么样"

**学习进度**
• 意图标签: progress_inquiry
• 典型用户话术: "孩子学到哪了"、"有没有学习报告"、"效果怎么样"

**投诉建议**
• 意图标签: complaint
• 典型用户话术: "态度太差了"、"我要投诉"、"不满意你们的服务"

**续费/续报**
• 意图标签: renewal
• 典型用户话术: "怎么续费"、"课时用完了"、"再报一期"

**其他/闲聊**
• 意图标签: chitchat
• 典型用户话术: "在吗"、"谢谢"、"好的"、"你是机器人吗"
"""

import json
import argparse
from collections import defaultdict


# ========== 所有意图标签 ==========
ALL_INTENTS = [
    "course_inquiry", "price_inquiry", "trial_booking",
    "refund_request", "schedule_change", "class_issue",
    "teacher_inquiry", "progress_inquiry", "complaint",
    "renewal", "chitchat"
]


def load_results(filepath):
    """
    加载推理结果文件，返回 (真实标签, 预测标签) 列表
    兼容多种 Swift 输出格式
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())

            # 格式1: {"messages": [...], "response": "预测"}
            # 真实标签在 messages 最后一个 assistant 消息里
            if "messages" in item:
                msgs = item["messages"]
                # 找最后一个 assistant 消息作为 ground truth
                label = None
                for msg in msgs:
                    if msg["role"] == "assistant":
                        label = msg["content"].strip()
                pred = item.get("response", "").strip()

            # 格式2: {"query": "...", "response": "预测", "label": "真实"}
            elif "label" in item:
                label = item["label"].strip()
                pred = item.get("response", "").strip()

            # 格式3: {"ground_truth": "...", "predict": "..."}
            elif "ground_truth" in item:
                label = item["ground_truth"].strip()
                pred = item.get("predict", "").strip()

            else:
                print(f"[WARN] 无法解析行: {line[:80]}...")
                continue

            if label is None:
                continue

            pairs.append((label, pred))

    return pairs


def evaluate(pairs):
    """计算各项指标"""
    # --- 整体准确率 ---
    correct = sum(1 for label, pred in pairs if label == pred)
    total = len(pairs)
    accuracy = correct / total if total > 0 else 0

    # --- 每个意图的 TP/FP/FN ---
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for label, pred in pairs:
        if label == pred:
            tp[label] += 1
        else:
            fn[label] += 1  # 该类漏识别
            fp[pred] += 1   # 预测类多识别

    # --- 计算 P/R/F1 ---
    metrics = {}
    for intent in ALL_INTENTS:
        p = tp[intent] / (tp[intent] + fp[intent]) if (tp[intent] + fp[intent]) > 0 else 0
        r = tp[intent] / (tp[intent] + fn[intent]) if (tp[intent] + fn[intent]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[intent] + fn[intent]  # 该类真实样本数
        metrics[intent] = {"precision": p, "recall": r, "f1": f1, "support": support}

    # --- 检查是否有未知标签 ---
    unknown_preds = set()
    for label, pred in pairs:
        if pred not in ALL_INTENTS:
            unknown_preds.add(pred)

    return accuracy, total, correct, metrics, unknown_preds


def print_report(accuracy, total, correct, metrics, unknown_preds, errors):
    """打印评估报告"""
    print("=" * 60)
    print("  教育客服意图识别 - 评估报告")
    print("=" * 60)
    print(f"\n  总样本数: {total}")
    print(f"  正确数:   {correct}")
    print(f"  准确率:   {accuracy:.1%}")
    print()

    # --- 分意图指标 ---
    print("-" * 60)
    print(f"  {'意图':<20} {'Precision':>9} {'Recall':>9} {'F1':>9} {'样本数':>6}")
    print("-" * 60)

    for intent in ALL_INTENTS:
        m = metrics[intent]
        if m["support"] > 0:  # 只显示测试集中出现过的意图
            print(f"  {intent:<20} {m['precision']:>8.1%} {m['recall']:>8.1%} {m['f1']:>8.1%} {m['support']:>6}")

    print("-" * 60)

    # --- Macro 平均 ---
    active = [m for m in metrics.values() if m["support"] > 0]
    if active:
        macro_p = sum(m["precision"] for m in active) / len(active)
        macro_r = sum(m["recall"] for m in active) / len(active)
        macro_f1 = sum(m["f1"] for m in active) / len(active)
        print(f"  {'Macro Avg':<20} {macro_p:>8.1%} {macro_r:>8.1%} {macro_f1:>8.1%}")
    print()

    # --- 未知预测标签 ---
    if unknown_preds:
        print(f"  ⚠️  模型输出了未知标签 ({len(unknown_preds)} 种):")
        for p in sorted(unknown_preds):
            print(f"     - \"{p}\"")
        print("  → 这些会被统计为预测错误")
        print()

    # --- 错误样本 ---
    if errors:
        print(f"  错误样本 (共 {len(errors)} 条，展示前 20):")
        print(f"  {'输入':<30} {'真实':>18} {'预测':>18}")
        print("  " + "-" * 66)
        for query, label, pred in errors[:20]:
            q = query[:28] + ".." if len(query) > 30 else query
            print(f"  {q:<30} {label:>18} {pred:>18}")
        print()


def extract_errors(filepath, pairs):
    """提取错误样本的原始输入文本"""
    errors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= len(pairs):
                break
            label, pred = pairs[i]
            if label != pred:
                item = json.loads(line.strip())
                # 提取用户输入
                query = ""
                if "messages" in item:
                    for msg in item["messages"]:
                        if msg["role"] == "user":
                            query = msg["content"]
                elif "query" in item:
                    query = item["query"]
                errors.append((query, label, pred))
    return errors


def main():
    parser = argparse.ArgumentParser(description="教育客服意图识别评估")
    parser.add_argument("--result", required=True, help="Swift infer 输出的结果文件路径")
    args = parser.parse_args()

    # 加载 & 评估
    pairs = load_results(args.result)
    if not pairs:
        print("❌ 未能解析出任何结果，请检查文件格式")
        return

    accuracy, total, correct, metrics, unknown_preds = evaluate(pairs)
    errors = extract_errors(args.result, pairs)
    print_report(accuracy, total, correct, metrics, unknown_preds, errors)


if __name__ == "__main__":
    main()
