"""Reusable data loading, preprocessing, training, and evaluation for the project."""
from __future__ import annotations
from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "G3"
LEAKAGE_COLUMNS = {"G1", "G2", "G3"}


def load_data() -> pd.DataFrame:
    """Fetch the official UCI Mathematics CSV from the dataset archive."""
    import io
    import zipfile
    import requests
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        # Use the explicitly named Mathematics file, not the Portuguese subset.
        with archive.open("student-mat.csv") as source:
            data = pd.read_csv(source, sep=";")
    if TARGET not in data.columns:
        raise ValueError("The official UCI Mathematics CSV did not contain G3.")
    return data


def build_pipeline(X: pd.DataFrame, estimator):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def evaluate_and_train(data: pd.DataFrame):
    if TARGET not in data:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    y = pd.to_numeric(data[TARGET], errors="coerce")
    # Early-risk use case: intentionally exclude G1 and G2 to avoid using earlier
    # period grades. The UCI documentation warns these are strongly correlated
    # with G3, so including them would change the business question.
    X = data.drop(columns=[c for c in LEAKAGE_COLUMNS if c in data.columns])
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Histogram gradient boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, l2_regularization=1.0, random_state=42),
    }
    results, fitted = [], {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, estimator in models.items():
        pipe = build_pipeline(X_train, estimator)
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                   scoring={"mae":"neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error", "r2":"r2"},
                                   n_jobs=None)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "model": name,
            "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
            "cv_mae_std": float(cv_scores["test_mae"].std()),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)),
        })
        fitted[name] = pipe
    metrics = pd.DataFrame(results).sort_values("cv_mae_mean").reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, ARTIFACTS / "best_model.joblib")
    metrics.to_csv(ARTIFACTS / "model_metrics.csv", index=False)
    X_test.assign(actual=y_test, prediction=best_model.predict(X_test)).to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    with open(ARTIFACTS / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"dataset":"UCI Student Performance, Mathematics subset", "uci_id":320,
                   "target":TARGET, "excluded_features":["G1","G2"], "random_state":42,
                   "test_size":0.2, "best_model_by_cv_mae":best_name,
                   "n_rows":int(len(data)), "n_features":int(X.shape[1])}, f, indent=2)
    return metrics, best_name, X, y, X_test, y_test, best_model


def run_training():
    data = load_data()
    data.to_csv(ARTIFACTS / "source_data_snapshot.csv", index=False)
    return data, evaluate_and_train(data)
