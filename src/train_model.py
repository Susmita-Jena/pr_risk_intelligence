import pandas as pd
import joblib
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

# Load processed dataset
df = pd.read_parquet(
    "dataset/processed/commits_processed.parquet"
)

# Select input features
features = [
    "nfiles", "la", "ld", "hunks",
    "ndirs", "nexts", "test_ratio",
    "a_prior_n", "a_prior_rate",
    "a_days", "msg_len", "msg_ref"
]

X = df[features]
y = df["y"]
groups = df["repo"]

# Split data by repository
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Check that both classes are present
if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Train or test split contains only one class. "
        "Try another random_state."
    )

# Handle class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

# Create machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=negative / positive,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    ))
])

# Train model
print("Training the model...")

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nROC-AUC:")
print(roc_auc_score(y_test, probabilities))

print("\nPR-AUC:")
print(average_precision_score(y_test, probabilities))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/pr_risk_model.pkl")

print("\nModel trained and saved successfully!")