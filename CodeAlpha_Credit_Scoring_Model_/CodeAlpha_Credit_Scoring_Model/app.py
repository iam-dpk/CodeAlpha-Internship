import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "dataset" / "german_credit.csv"
MODEL_PATH = BASE / "models" / "logistic_regression_credit_model.joblib"
METRICS_PATH = BASE / "results" / "final_metrics.json"

st.title("💳 Credit Scoring Prediction System")
st.caption("CodeAlpha Machine Learning Internship — Task 1")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# ---------- Sidebar ----------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Dashboard", "EDA", "Model Performance", "Credit Prediction"]
)

# ---------- Dashboard ----------
if page == "Dashboard":
    st.header("📊 Project Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", f"{len(df):,}")
    if "credit_risk" in df.columns:
        good = int((df["credit_risk"] == 0).sum())
        bad = int((df["credit_risk"] == 1).sum())
        c2.metric("Good Credit", f"{good:,}")
        c3.metric("Bad Credit", f"{bad:,}")
    c4.metric("Features", f"{df.shape[1]-1}")

    st.subheader("Credit Risk Distribution")
    if "credit_risk" in df.columns:
        counts = df["credit_risk"].map({0: "Good Credit", 1: "Bad Credit"}).value_counts()
        st.bar_chart(counts)

    st.info(
        "This dashboard uses the trained machine learning pipeline saved in the project. "
        "The prediction page is intended for educational demonstration."
    )

# ---------- EDA ----------
elif page == "EDA":
    st.header("🔎 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if "credit_risk" in df.columns:
        st.subheader("Target Distribution")
        counts = df["credit_risk"].map({0: "Good Credit", 1: "Bad Credit"}).value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Good vs Bad Credit")
        ax.set_xlabel("Credit Risk")
        ax.set_ylabel("Applicants")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "credit_risk"]

    if numeric_cols:
        selected = st.selectbox("Select a numerical feature", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[selected].dropna(), bins=20)
        ax.set_title(f"Distribution of {selected}")
        ax.set_xlabel(selected)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# ---------- Model Performance ----------
elif page == "Model Performance":
    st.header("🤖 Model Performance")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        cols = st.columns(5)
        metric_items = [
            ("Accuracy", metrics.get("accuracy")),
            ("Precision", metrics.get("precision")),
            ("Recall", metrics.get("recall")),
            ("F1-Score", metrics.get("f1_score")),
            ("ROC-AUC", metrics.get("roc_auc")),
        ]

        for col, (label, value) in zip(cols, metric_items):
            col.metric(label, f"{value:.1%}" if value is not None else "N/A")

        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size == 4:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm)
            ax.set_xticks([0, 1], ["Good", "Bad"])
            ax.set_yticks([0, 1], ["Good", "Bad"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Final metrics file was not found.")

# ---------- Prediction ----------
elif page == "Credit Prediction":
    st.header("🔮 Live Credit Risk Prediction")
    st.write("Enter an applicant's data and run the trained model.")

    if "credit_risk" not in df.columns:
        st.error("Target column 'credit_risk' is missing from the dataset.")
        st.stop()

    feature_cols = [c for c in df.columns if c != "credit_risk"]

    # Build controls from the dataset so the app matches the trained pipeline.
    values = {}
    left, right = st.columns(2)

    for i, col in enumerate(feature_cols):
        series = df[col]
        container = left if i % 2 == 0 else right

        if pd.api.types.is_numeric_dtype(series):
            min_v = float(series.min())
            max_v = float(series.max())
            median_v = float(series.median())
            step = 1.0 if series.dropna().nunique() <= 50 else (max_v - min_v) / 100
            if step <= 0:
                step = 1.0
            values[col] = container.number_input(
                col, min_value=min_v, max_value=max_v,
                value=median_v, step=float(step)
            )
        else:
            options = series.dropna().astype(str).unique().tolist()
            values[col] = container.selectbox(col, options)

    if st.button("🚀 Predict Credit Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values])
        prediction = int(model.predict(input_df)[0])
        probability = model.predict_proba(input_df)[0, 1]

        st.divider()

        if prediction == 1:
            st.error("⚠️ Prediction: BAD CREDIT RISK")
        else:
            st.success("✅ Prediction: GOOD CREDIT RISK")

        p1, p2 = st.columns(2)
        p1.metric("Bad Credit Probability", f"{probability:.1%}")
        p2.metric("Good Credit Probability", f"{1-probability:.1%}")

        st.progress(float(probability))
        st.caption("Probability shown is the model's estimated probability for the Bad Credit class.")
