from huggingface_hub import hf_hub_download
import pandas as pd
import os

os.makedirs("dataset/raw", exist_ok=True)

path = hf_hub_download(
    repo_id="amazeng-co/git-jit-defects",
    filename="commits.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(path)

df.to_parquet(
    "dataset/raw/commits.parquet",
    index=False
)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())