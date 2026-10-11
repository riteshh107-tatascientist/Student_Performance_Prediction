"""Streamlit dashboard for the Student Performance Analytics project."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline import run_training, ARTIFACTS, load_data
import joblib

st.set_page_config(page_title="Student Performance Analytics", page_icon="📊", layout="wide")
st.title("Student Performance Analytics & Early Prediction")
st.caption("Educational data mining demonstration · UCI Student Performance dataset · Research/learning use only")

@st.cache_data(show_spinner="Fetching the documented UCI dataset…")
def get_data():
    return load_data()

@st.cache_resource(show_spinner="Training and evaluating models…")
def get_model_bundle():
    data, result = run_training()
    metrics, best_name, X, y, X_test, y_test, model = result
    return data, metrics, best_name, X, y, model

try:
    data, metrics, best_name, X, y, model = get_model_bundle()
except Exception as exc:
    st.error("Dataset download or model training failed. Check your internet connection and installed requirements, then retry.")
    st.exception(exc)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset records", f"{len(data):,}")
c2.metric("Input features", f"{X.shape[1]}")
c3.metric("Average final grade", f"{data['G3'].mean():.2f} / 20")
c4.metric("Selected model", best_name)

st.subheader("1. Explore the dataset")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots()
    sns.histplot(data=data, x="G3", bins=11, kde=True, ax=ax)
    ax.set_title("Distribution of final grades (G3)")
    st.pyplot(fig)
    plt.close(fig)
with right:
    if "studytime" in data.columns:
        summary = data.groupby("studytime", as_index=False)["G3"].mean()
        fig, ax = plt.subplots()
        sns.barplot(data=summary, x="studytime", y="G3", ax=ax)
        ax.set_title("Mean final grade by study-time category")
        ax.set_xlabel("Study time category (1 = <2 h/week; 4 = >10 h/week)")
        st.pyplot(fig)
        plt.close(fig)

st.subheader("2. Model evaluation")
st.write("Models are compared on a fixed held-out test set. Five-fold cross-validation on training data selects the model with the lowest mean MAE. G1 and G2 are excluded from inputs.")
st.dataframe(metrics, use_container_width=True, hide_index=True)
st.download_button("Download model metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "model_metrics.csv", "text/csv")

st.subheader("3. Explore a hypothetical profile")
st.info("Enter educational/context variables to get a model estimate. This is a research demonstration, not a high-stakes assessment or a substitute for educator judgment.")
with st.form("profile"):
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        target_col = cols[i % 3]
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            low = float(series.min())
            high = float(series.max())
            default = float(series.median())
            values[col] = target_col.number_input(str(col), min_value=low, max_value=high, value=default, step=1.0 if series.nunique() < 20 else 0.5)
        else:
            opts = sorted(series.dropna().astype(str).unique().tolist())
            values[col] = target_col.selectbox(str(col), opts if opts else ["unknown"])
    submitted = st.form_submit_button("Estimate final grade")
if submitted:
    sample = pd.DataFrame([values], columns=X.columns)
    estimate = float(model.predict(sample)[0])
    st.metric("Estimated final grade", f"{max(0, min(20, estimate)):.1f} / 20")
    st.caption("Estimate clipped to the dataset's 0–20 grade scale for display. Model errors and dataset limitations still apply.")

st.subheader("4. Data preview")
st.dataframe(data.head(20), use_container_width=True)
st.download_button("Download data snapshot CSV", data.to_csv(index=False).encode("utf-8"), "student_performance_data.csv", "text/csv")
st.caption("Dataset source: Cortez (2008), UCI Machine Learning Repository, DOI: 10.24432/C5TG7T. Dataset observations come from two Portuguese secondary schools and may not generalize to other populations.")
