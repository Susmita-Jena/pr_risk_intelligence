
# PR Risk Intelligence

A machine learning-based system for predicting pull request risk using historical software development data.

## Features
- Pull request risk prediction
- Historical commit analysis
- Exploratory Data Analysis (EDA)
- Machine learning model training
- Automated dataset download

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- PyArrow
- Matplotlib, Seaborn

## Setup

Clone the repository:

```bash
git clone https://github.com/Susmita-Jena/pr_risk_intelligence.git
cd pr_risk_intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python dataset/download_dataset.py
```

The dataset will be saved in `dataset/raw/commits.parquet`.

## Dataset Source

[Git-JIT Defects Dataset](https://huggingface.co/datasets/amazeng-co/git-jit-defects)

The raw dataset is excluded from this repository. Please follow the original dataset's CC BY 4.0 licence and attribution requirements.
