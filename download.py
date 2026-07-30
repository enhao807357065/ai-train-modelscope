from modelscope.hub.snapshot_download import snapshot_download

# model_dir = snapshot_download("Qwen/Qwen3.5-2B")
# print(f"\n Downloaded model to {model_dir}")

# 2.5模型成熟度较高
model_dir = snapshot_download("Qwen/Qwen2.5-1.5B")
print(f"\n Downloaded model to {model_dir}")

# 下载和加载数据集
from modelscope import MsDataset
ds_dict = MsDataset.load('swift/self-cognition')
print(ds_dict['train'][0])