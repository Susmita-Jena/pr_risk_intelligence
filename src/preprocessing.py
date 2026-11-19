import pandas as pd
import os

# Load raw dataset
df = pd.read_parquet("dataset/raw/commits.parquet")

print("Original Dataset Shape:", df.shape)

# Remove duplicate commits
df = df.drop_duplicates(subset=["repo", "sha"])

# Remove rows with missing target values
df = df.dropna(subset=["y"])

# Keep only valid target labels (0 and 1)
df = df[df["y"].isin([0, 1])]

# Select useful features
features = [
    "nfiles",
    "la",
    "ld",
    "hunks",
    "ndirs",
    "nexts",
    "test_ratio",
    "a_prior_n",
    "a_prior_rate",
    "a_days",
    "msg_len",
    "msg_ref"
]

# Keep features, target and repository ID
df = df[features + ["y", "repo"]]

# Create processed folder if it doesn't exist
os.makedirs("dataset/processed", exist_ok=True)

# Save cleaned dataset
df.to_parquet(
    "dataset/processed/commits_processed.parquet",
    index=False
)

print("\nPreprocessing completed successfully!")
print("Processed Dataset Shape:", df.shape)

print("\nBug Distribution:")
print(df["y"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())