import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load processed dataset
df = pd.read_parquet(
    "dataset/processed/commits_processed.parquet"
)

# Create folder to save graphs
os.makedirs("notebooks/graphs", exist_ok=True)

print("Dataset Shape:", df.shape)

# Graph 1: Bug vs Clean Commits
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="y")
plt.title("Bug-Inducing vs Clean Commits")
plt.xlabel("Commit Type (0 = Clean, 1 = Bug)")
plt.ylabel("Number of Commits")
plt.savefig("notebooks/graphs/bug_distribution.png")
plt.show()

# Use a sample for faster plotting
sample = df.sample(
    n=min(20000, len(df)),
    random_state=42
)

# Graph 2: Lines Added
plt.figure(figsize=(6, 4))
sns.boxplot(data=sample, x="y", y="la")
plt.title("Lines Added vs Bug Labels")
plt.savefig("notebooks/graphs/lines_added.png")
plt.show()

# Graph 3: Files Modified
plt.figure(figsize=(6, 4))
sns.boxplot(data=sample, x="y", y="nfiles")
plt.title("Files Modified vs Bug Labels")
plt.savefig("notebooks/graphs/files_modified.png")
plt.show()

# Graph 4: Feature Correlation
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.savefig("notebooks/graphs/correlation.png")
plt.show()

print("\nEDA completed successfully!")
print("All graphs saved in notebooks/graphs/")