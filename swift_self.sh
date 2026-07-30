# 完整训练循环图
# ┌─────────────────────────────────────────────────────┐
  #│                    一个训练步                          │
  #│                                                       │
  #│  训练数据  ──→  Tokenize  ──→  模型前向传播           │
  #│  (instruction      (文字→数字)      (预测每个token)    │
  #│   + output)                              │             │
  #│                                          ▼             │
  #│                                    计算 Loss           │
  #│                                    (预测 vs 答案)      │
  #│                                          │             │
  #│                                          ▼             │
  #│                                    反向传播             │
  #│                                    (算梯度)            │
  #│                                          │             │
  #│                                          ▼             │
  #│                              梯度累积满 16 步？         │
  #│                              /        \                │
  #│                           否 /          \ 是            │
  #│                            /              \             │
  #│                     继续下一条        更新参数（优化器）  │
  #│                                      重置梯度           │
  #│                                          │             │
  #│                                          ▼             │
  #│                                    记录 loss/lr         │
  #│                                    (logging_steps=5)   │
  #└─────────────────────────────────────────────────────┘
  #                              │
  #                    重复直到数据跑完 × epoch 数

# SFT 和其他微调方式的关系
  #
  #**SFT**
  #• 全称: Supervised Fine-Tuning
  #• 数据格式: 指令+标准答案
  #• 目标: 让模型学会遵循指令
  #
  #RLHF
  #• 全称: Reinforcement Learning from Human Feedback
  #• 数据格式: 人类偏好排序
  #• 目标: 让模型输出更符合人类偏好
  #
  #DPO
  #• 全称: Direct Preference Optimization
  #• 数据格式: 好/坏回答对比
  #• 目标: 同上，但不需要训练奖励模型
  #
  #你现在做的就是 SFT——最基础也是最直观的微调方式。它的 pipeline 是：
# base model（只会续写）
  #    ↓ SFT（学会听指令）
  #chat model（能对话了）
  #    ↓ RLHF/DPO（学会什么是"好回答"）
  #aligned model（回答质量高、安全）

# 当前的目标是让模型学会"我是 enhao-robot"。理论上只用 swift/self-cognition#500 就够了，但 500 条数据太少，模型学不稳格式。加 alpaca 是为了凑量 + 稳定格式，不是为了教它新技能。
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3.5-2B \                             # ModelScope Hub 上的模型 ID，自动下载
    --tuner_type lora \                                   # 微调方式：lora / full（全参数） / adalora / longlora 等
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \   # 数据集 可以有多个，#500表示随机采样500条
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \                              # 模型加载精度。bf16 省显存且训练稳定，需 Ampere+ 架构（30系/A100）
    --num_train_epochs 3 \                                # 整体训练轮数。=1表示数据集过一遍就停，可以多训几轮。数据越少 → epoch 越不能多。反直觉但正确——数据少时多看几遍反而加速过拟合。
    --per_device_train_batch_size 1 \                     # 每个设备上的训练批次大小，影响内存占用和梯度估计质量
                                                          # 每 GPU 每步处理的样本数 总数据：1500 条。 1 相当于只往一个方向踩一脚，凭这一脚的感觉决定走哪边。可能踩到了一个局部小坑，判断不准。16 往 16 个方向各踩一脚，综合感知哪边整体更低。判断更靠谱，但花了 16 倍时间探路。和gradient_accumulation_steps相辅相成
                                                          # 这就是为什么 batch 太小会"震荡"（每次探的方向噪声大，走一步偏左走一步偏右），batch 大了更"平滑"（方向估计准，路径更直）。
                                                          # 等效 batch：1 × 16 = 16
                                                          # 总优化步数：1500 ÷ 16 ≈ 94 步
    --per_device_eval_batch_size 1 \                      # 每个设备上的评估批次大小，通常与训练batch size相同
    --learning_rate 1e-4 \                                # 学习率。学习率决定了每次参数更新迈多大的步子。控制参数更新步长，较大值收敛快但可能震荡。想象你蒙着眼在山上找最低点（loss 最小处）：
#                                                            - 学习率太大（如 1e-2）→ 步子太大，在山谷两边反复跳来跳去，永远找不到底
#                                                            - 学习率太小（如 1e-6）→ 步子太小，走半天还在原地，训练极慢
#                                                            - 合适的学习率（如 1e-4）→ 稳步下山
    --lora_rank 8 \                                       # lora的秩。越大表示能力越强，但参数越多、越容易过拟合，常规设置8
                                                          # 如何选择lora_rank： Rank 的本质是"信息瓶颈"的宽度 — 你的任务需要多少"新知识"，就需要多宽的通道。
                                                          # 经验速查表
                                                          #
                                                          #风格迁移 / 语气调整
                                                          #• 推荐 rank: 4-8
                                                          #• 原因: 只需微调表达方式，改动小
                                                          #
                                                          #单领域问答 / 指令跟随
                                                          #• 推荐 rank: 8-16
                                                          #• 原因: 需要学新知识，但范围有限
                                                          #
                                                          #多任务 / 复杂推理
                                                          #• 推荐 rank: 32-64
                                                          #• 原因: 多种能力同时调整
                                                          #
                                                          #接近全量微调效果
                                                          #• 推荐 rank: 128-256
                                                          #• 原因: 几乎重塑模型行为
    --lora_alpha 32 \                                     # lora的缩放系数。实际作用 = alpha / rank，这里是32/8=4倍放大
                                                          # - α/r 越大 → LoRA 分支的贡献越大 → 模型变化越激进
                                                          # - α/r 越小 → LoRA 分支的贡献越小 → 模型更保守
    --target_modules all-linear \                         # 对哪些层加lora。all-linear=所有线性层（最彻底）。Transformer 每一层里有很多线性层（nn.Linear），LoRA 的本质是"选几个线性层，给它们加旁路"。target_modules 就是告诉 LoRA 选哪些。也可以只选q_proj v_proj    # 只对 Q 和 V 加（经典做法，参数最少）
    --gradient_accumulation_steps 16 \                    # 梯度累积步数。相当于每看 16 条样本才更新一次权重。 和per_device_train_batch_size相辅相成
    --eval_steps 50 \                                     # 每 50 步在验证集上评估一次
    --save_steps 50 \                                     # 每 50 步保存一次 checkpoint
    --save_total_limit 2 \                                # 最多保留 2 个 checkpoint（旧的自动删除，省磁盘）
    --logging_steps 5 \                                   # 每 5 步记录一次 loss 到 TensorBoard
    --max_length 2048 \                                   # 每条训练样本的最大token数，超出会截断
    --output_dir output \                                 # checkpoint 和日志的输出目录
    --system 'You are a helpful assistant.' \             # 系统提示词，会拼在每条训练数据前面
    --warmup_ratio 0.05 \                                 # 预热比例（前 5% 步数 LR 从 0 线性升到目标值）？ Warmup 的意义：训练刚开始时参数随机性大，如果 LR 直接拉满容易震荡。先用小 LR"热身"，让模型找到一个合理的优化方向后再加速。
    --dataloader_num_workers 4 \                          # 数据加载的子进程数，加速预处理（和CPU核数相关）
    --model_author enhao \                                # 训练出来的模型"作者"名
    --model_name enhao-robot                              # 训练出来的模型"名字"。这两个配合 swift/self-cognition 数据集使用。该数据集里有类似"你是谁？""谁创造了你？"的问答，swift 会自动把 model_author 和 model_name 填入答案模板，让模型学会回答"我是 enhao 创建的 enhao-robot"。
#    --evaluation_strategy epoch                          # 评估策略，设为epoch表示每个epoch结束后进行一次验证评估
#    --save_strategy epoch                                # 模型保存策略，设为epoch表示每个epoch结束后保存一次检查点
#    --load_best_model_at_end true                        # 训练结束时是否自动加载验证集上表现最好的模型
#    --lr_scheduler_type cosine                           # 学习率调度策略，三种常见调度策略。 Cosine 为什么最常用？ 因为训练后期模型已经接近最优解了，此时需要更小的 lr 来"精细调整"而不是"大步跨过"。Cosine 在后期自然地把 lr 压低，相当于自动做了精细化。
                                                                    #
                                                                    #**linear**
                                                                    #• 曲线形状: 直线下降
                                                                    #• 特点: warmup 后匀速降到 0，简单粗暴
                                                                    #
                                                                    #**cosine** 默认
                                                                    #• 曲线形状: 余弦曲线
                                                                    #• 特点: 前期降得慢，后期降得快，最常用
                                                                    #
                                                                    #**constant**
                                                                    #• 曲线形状: 平坦
                                                                    #• 特点: warmup 后保持不变，适合数据少/epoch 少
#    --early_stopping_patience 3                          # eval loss 连续 3 次不降就停

# LoRA 核心思想：不动原始权重 W，旁路插一个低秩矩阵 ΔW = B×A（rank=8 意味着 A 是 d×8，B 是 8×d），只训练这个小矩阵。

# output 产出物说明
# v4-20260727-115155/
  #├── checkpoint-50/              # 第 50 步快照
  #├── checkpoint-94/              # 最终快照（训练完成）
  #│   ├── adapter_config.json     # LoRA 配置
  #│   ├── adapter_model.safetensors  # ★ LoRA 权重（推理用这个）
  #│   ├── additional_config.json  # swift 额外配置
  #│   ├── args.json               # 训练参数（可读版）
  #│   ├── optimizer.pt            # 优化器状态（续训用）
  #│   ├── scheduler.pt            # LR scheduler 状态（续训用）
  #│   ├── rng_state.pth           # 随机数状态（续训用，保证可复现）
  #│   ├── trainer_state.json      # 训练进度（当前 step、best metric 等）
  #│   ├── training_args.bin       # TrainingArguments 完整对象（续训用）
  #│   └── README.md               # 模型卡片（自动生成）
  #├── images/                     # 训练曲线可视化
  #│   ├── train_loss.png          # loss 曲线
  #│   ├── train_learning_rate.png # LR 变化
  #│   ├── train_token_acc.png     # token 级准确率
  #│   ├── train_grad_norm.png     # 梯度范数（监控训练稳定性）
  #│   └── ...
  #├── runs/                       # TensorBoard 日志
  #│   └── Jul27_11-52-22_.../
  #│       └── events.out.tfevents...  # 用 tensorboard 打开看交互式图表
  #├── args.json                   # 完整训练参数（可读 JSON）
  #└── logging.jsonl               # 逐步训练日志

# training_args.bin  transformers Trainer 的默认行为——它用 torch.save() 把整个 TrainingArguments 对象序列化为 pickle 格式（.bin）。

# batch_size vs gradient_accumulation_steps
  #
  #本质区别：batch_size 是"一次性同时处理多少条"，gradient_accumulation 是"分几次凑够这么多条"。
# 核心对比
  #
  #工作方式
  #• batch_size: 多条样本同时进GPU并行计算
  #• gradient_accumulation_steps: 多条样本依次进GPU串行计算
  #
  #显存占用
  #• batch_size: 成倍增长
  #• gradient_accumulation_steps: 不变（和 batch=1 一样）
  #
  #计算速度
  #• batch_size: 快（GPU并行能力被充分利用）
  #• gradient_accumulation_steps: 慢（串行 N 次）
  #
  #梯度效果
  #• batch_size: N 条的平均梯度
  #• gradient_accumulation_steps: N 条的累加梯度（等价）
# 用下山类比
  #
  #假设你想"探 16 个方向取平均再决定往哪走"：
  #
  #- batch_size=16 → 派 16 个人同时往不同方向踩一脚，瞬间汇报结果。需要 16 个人（= 16 倍显存）
  #- batch_size=1, accumulation=16 → 你一个人依次往 16 个方向各踩一脚，记下来，最后综合判断。只需要 1 个人（= 1 倍显存），但花了 16 倍时间