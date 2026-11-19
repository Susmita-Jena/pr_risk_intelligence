import pandas as pd

df = pd.read_parquet("dataset/raw/commits.parquet")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nBug Distribution:")
print(df["y"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())