
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="PR Risk Intelligence",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pr_risk_model.pkl"

FEATURES = [
    "nfiles", "la", "ld", "hunks",
    "ndirs", "nexts", "test_ratio",
    "a_prior_n", "a_prior_rate",
    "a_days", "msg_len", "msg_ref"
]

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Header
st.title("🔍 PR Risk Intelligence")
st.subheader("AI-Powered Bug Risk Prediction")
st.write(
    "Predict whether a software code change "
    "is likely to introduce defects."
)

st.divider()

# Model information
c1, c2, c3 = st.columns(3)
c1.metric("ML Model", "XGBoost")
c2.metric("Input Features", "12")
c3.metric("Model Status", "Loaded ✅")

# Prediction function
def predict(data):
    data = data[FEATURES].apply(
        pd.to_numeric, errors="raise"
    )

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Input contains invalid numeric values.")

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    results = data.copy()
    results["Prediction"] = np.where(
        predictions == 1,
        "Bug-Inducing",
        "Clean"
    )
    results["Bug Probability (%)"] = (
        probabilities * 100
    ).round(2)

    return results, probabilities


# Dashboard tabs
tab1, tab2, tab3 = st.tabs([
    "Single PR Analysis",
    "CSV Batch Analysis",
    "SHAP Explanation"
])

# SINGLE PREDICTION
with tab1:

    st.header("Analyze Code Changes")

    with st.form("pr_form"):

        defaults = {
            "nfiles": 5,
            "la": 100,
            "ld": 20,
            "hunks": 10,
            "ndirs": 2,
            "nexts": 2,
            "test_ratio": 0.2,
            "a_prior_n": 10,
            "a_prior_rate": 0.1,
            "a_days": 100,
            "msg_len": 40,
            "msg_ref": 0
        }

        values = {}

        cols = st.columns(3)

        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=float(defaults[feature]),
                    step=1.0
                )

        submitted = st.form_submit_button(
            "Predict Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            results, probabilities = predict(input_data)

            st.session_state["last_input"] = input_data

            probability = probabilities[0]
            prediction = results["Prediction"].iloc[0]

            st.divider()
            st.subheader("Prediction Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "Bug Probability",
                f"{probability * 100:.2f}%"
            )

            c2.metric(
                "Model Prediction",
                prediction
            )

            st.progress(float(probability))

            st.info(
                "This is a model-based risk estimate, "
                "not confirmation of an actual bug."
            )

        except Exception as e:
            st.error(str(e))


# CSV BATCH PREDICTION
with tab2:

    st.header("Batch PR Analysis")

    uploaded = st.file_uploader(
        "Upload CSV with all 12 model features",
        type=["csv"]
    )

    if uploaded:

        try:
            csv_data = pd.read_csv(uploaded)

            missing = [
                f for f in FEATURES
                if f not in csv_data.columns
            ]

            if missing:
                st.error(f"Missing columns: {missing}")

            elif csv_data.empty:
                st.warning("CSV file is empty.")

            else:
                results, _ = predict(csv_data)

                st.success(
                    f"Analyzed {len(results)} records."
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.download_button(
                    "Download Predictions",
                    results.to_csv(index=False),
                    "pr_risk_results.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"CSV analysis failed: {e}")


# SHAP EXPLANATION
with tab3:

    st.header("Prediction Explainability")

    if "last_input" not in st.session_state:

        st.info(
            "Make a prediction in the first tab "
            "to view its SHAP explanation."
        )

    else:

        try:
            classifier = model.named_steps["classifier"]
            imputer = model.named_steps["imputer"]

            transformed = imputer.transform(
                st.session_state["last_input"][FEATURES]
            )

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)

            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[:, :, 1]

            contributions = pd.Series(
                values[0],
                index=FEATURES
            ).sort_values()

            fig, ax = plt.subplots(figsize=(9, 5))

            colors = [
                "green" if x < 0 else "red"
                for x in contributions.values
            ]

            ax.barh(
                contributions.index,
                contributions.values,
                color=colors
            )

            ax.set_title("SHAP Feature Contributions")
            ax.set_xlabel("SHAP value")

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Positive values push the model output "
                "towards the bug-inducing class; negative "
                "values push it away."
            )

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")


# Footer
st.divider()
st.caption(
    "PR Risk Intelligence | XGBoost + SHAP | "
    "Python Machine Learning Project"
)
