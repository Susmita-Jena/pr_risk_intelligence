
from pathlib import Path
import urllib.request
import pyarrow.parquet as pq

DATASET_URL = "https://huggingface.co/datasets/amazeng-co/git-jit-defects/resolve/main/commits.parquet"

RAW_DIR = Path(__file__).resolve().parent / "raw"
DATASET_FILE = RAW_DIR / "commits.parquet"
TEMP_FILE = RAW_DIR / "commits_download.tmp"


def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_FILE.exists():
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading dataset from Hugging Face...")

    try:
        urllib.request.urlretrieve(DATASET_URL, TEMP_FILE)

        table = pq.read_table(TEMP_FILE)

        if table.num_rows != 360463 or len(table.column_names) != 18:
            raise ValueError("Dataset validation failed.")

        TEMP_FILE.replace(DATASET_FILE)

        print("Dataset downloaded and validated successfully!")

    except Exception as error:
        TEMP_FILE.unlink(missing_ok=True)
        print(f"Download failed: {error}")
        raise


if __name__ == "__main__":
    download_dataset()
