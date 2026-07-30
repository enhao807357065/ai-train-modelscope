# ai-train-modelscope

基于 ModelScope + Swift + vLLM 的 Qwen2.5-1.5B 微调实践项目，覆盖从模型推理、LoRA 微调到 vLLM 部署的完整流程。

## 项目结构

```
├── first.py              # Transformer 推理最小示例（环境验证）
├── download.py           # ModelScope 模型 & 数据集下载
├── show_dataset.py       # 数据集内容查看
├── self_sft_01.py        # 手写 LoRA 训练循环（不依赖 Swift CLI）
├── swift_self.sh         # Swift CLI 一键 SFT 微调脚本
├── qwen_vllm_first.py    # vLLM 服务推理客户端示例
├── eval_intent.py        # 意图识别评测（Accuracy / F1 / 混淆矩阵）
├── eval_scope.py         # 意图识别模型评测（基于 vLLM 服务）
└── knowledge.txt         # 领域知识文档
```

## 环境

- Python 3.12
- PyTorch 2.11 + CUDA 13.0
- [ModelScope](https://github.com/modelscope/modelscope)
- [ms-swift](https://github.com/modelscope/ms-swift)
- [vLLM](https://github.com/vllm-project/vllm) 0.26.0
- GPU: RTX 4090 Laptop (16GB)

## 快速开始

### 1. 下载模型

```bash
python download.py
```

模型默认保存到 `models/` 目录。

### 2. 验证推理环境

```bash
python first.py
```

### 3. LoRA 微调

使用 Swift CLI：

```bash
bash swift_self.sh
```

或使用手写训练循环（便于理解底层细节）：

```bash
python self_sft_01.py
```

checkpoint 保存在 `output/` 目录。

### 4. 启动 vLLM 服务

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

CUDA_VISIBLE_DEVICES=0 vllm serve models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master \
    --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 8 \
    --lora-modules intent=output/v7-20260728-191401/checkpoint-250 \
    --gpu_memory_utilization 0.85 \
    --port 8000 \
    --max-model-len 4096
```

> 使用本地路径启动可避免 HuggingFace 网络请求卡死的问题。

### 5. 评测意图识别模型

```bash
python eval_intent.py
```

## 意图标签

| 标签 | 含义 |
|------|------|
| `course_inquiry` | 课程咨询 |
| `price_inquiry` | 价格咨询 |
| `trial_booking` | 试听预约 |
| `refund_request` | 退费/退课 |
| `schedule_change` | 排课/调课 |
| `class_issue` | 上课问题 |
| `teacher_inquiry` | 教师相关 |
