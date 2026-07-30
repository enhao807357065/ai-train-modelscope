import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import (
    get_model_processor,
    load_dataset,
    get_template,
    EncodePreprocessor,
)
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial

logger = get_logger()

model_id_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
system = """
You are a helpful assistant
"""
output_dir = "output"

dataset = [
    "AI-ModelScope/alpaca-gpt4-data-zh#500",
    "AI-ModelScope/alpaca-gpt4-data-en#500",
    "swift/self-cognition#500"
]

data_seed = 42  # 默认42
max_length = 2048
batch_size = 2
learning_rate = 1e-4
weight_decay = 0.01
gradient_accumulation_steps = 8

split_dataset_ratio = 0.01  # 切分验证集
num_proc = 4  # 预处理的进程数
# 替换自我认知数据集中的填充符：{{NAME}}, {{AUTHOR}}
model_name = ['恩皓', 'enhao']  # 模型的中文名和英文名
model_author = ['梁皓', 'lianghao']  # 模型作者的中文名和英文名

# lora
lora_rank = 8
lora_alpha = 32

# 训练超参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

# 下载并载入数据集，并切分成训练集和验证集，
output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

model, tokenizer = get_model_processor(model_id_or_path)
logger.info(f'model_info: {model.model_info}')
# 4.4.2 get_template 签名变了，第一个参数已经不再是 template_type，而是直接变成了 processor
template = get_template(tokenizer, default_system=system, max_length=max_length, template_type=model.model_meta.template)
template.set_mode('train')

target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# 打印模型结构和训练的参数量
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
        model_name=model_name, model_author=model_author, seed=data_seed)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

# 下载并载入数据集，并切分成训练集和验证集，
# 然后将文本编码成tokens：
train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# 打印一条样本
template.print_inputs(train_dataset[0])

# 初始化trainer并开始训练：
# 。Swift 4.4.2 的 Seq2SeqTrainer 在内部自动生成 data_collator（通过
#  self._get_data_collator(args, template) 第 48 行），然后自己传给父类。你再外部传入
#  data_collator=template.data_collator，就导致父类收到了两个 data_collator。
model.enable_input_require_grads()  # 兼容gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    template=template,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

# 可视化训练的loss。其中浅黄色线条代表真实loss值，黄色线条代表经过0.9平滑系数平滑后的loss值。

# 你也可以使用tensorboard进行实时可视化，在命令行输入tensorboard --logdir '{output_dir}/runs'。
images_dir = os.path.join(output_dir, 'images')
logger.info(f'images_dir: {images_dir}')
plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # 保存图片

# 展示图片
from IPython.display import display
from PIL import Image
image = Image.open(os.path.join(images_dir, 'train_loss.png'))
display(image)