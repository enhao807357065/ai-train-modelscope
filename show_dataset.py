# 默认从huggingface找，需要改为从modelscope找
# from datasets import load_dataset
from modelscope.msdatasets import MsDataset


# 查看 alpaca 中文数据集前 3 条
ds = MsDataset.load("AI-ModelScope/alpaca-gpt4-data-zh", split="train")
print(f"=== alpaca-zh: 总条数 {len(ds)}, 字段: {ds.column_names} ===")
for i in range(3):
    print(f"\n--- 第 {i+1} 条 ---")
    for k, v in ds[i].items():
        print(f"  {k}: {str(v)[:200]}")

print("\n\n")

# 查看 self-cognition 数据集前 3 条
ds2 = MsDataset.load("swift/self-cognition", split="train")
print(f"=== self-cognition: 总条数 {len(ds2)}, 字段: {ds2.column_names} ===")
for i in range(3):
    print(f"\n--- 第 {i+1} 条 ---")
    for k, v in ds2[i].items():
        print(f"  {k}: {str(v)[:200]}")
