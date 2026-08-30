"""General helper functions for the IDBF Streamlit application."""

from __future__ import annotations

import io
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate column names made unique."""
    out = df.copy()
    counts: dict[str, int] = {}
    cols: list[str] = []
    for col in map(str, out.columns):
        counts[col] = counts.get(col, 0) + 1
        cols.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
    out.columns = cols
    return out


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def to_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def serialize_joblib(obj: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return buffer.read()


def safe_sample(df: pd.DataFrame, n: int = 3000, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def load_sample_dataset(name: str) -> tuple[pd.DataFrame, str | None]:
    """Load built-in demonstration datasets. Returns dataframe and suggested target."""
    if name == "Iris Classification":
        data = load_iris(as_frame=True)
        df = data.frame.rename(columns={"target": "species"})
        df["species"] = df["species"].map(dict(enumerate(data.target_names)))
        return df, "species"
    if name == "Breast Cancer Classification":
        data = load_breast_cancer(as_frame=True)
        df = data.frame.rename(columns={"target": "diagnosis"})
        df["diagnosis"] = df["diagnosis"].map(dict(enumerate(data.target_names)))
        return df, "diagnosis"
    if name == "Titanic-style Classification":
        rng = np.random.default_rng(42)
        n = 700
        sex = rng.choice(["female", "male"], n, p=[0.42, 0.58])
        pclass = rng.choice([1, 2, 3], n, p=[0.22, 0.27, 0.51])
        age = np.clip(rng.normal(31, 14, n), 1, 80).round(1)
        fare = np.clip(rng.gamma(2.2, 18, n) + (4 - pclass) * 12, 5, 250).round(2)
        embarked = rng.choice(["S", "C", "Q"], n, p=[0.70, 0.20, 0.10])
        logit = 1.4 * (sex == "female") + 0.65 * (pclass == 1) - 0.45 * (pclass == 3) - 0.015 * age + 0.004 * fare - 0.2
        prob = 1 / (1 + np.exp(-logit))
        survived = (rng.random(n) < prob).astype(int)
        df = pd.DataFrame({
            "Passenger_ID": np.arange(1, n + 1), "Pclass": pclass, "Sex": sex, "Age": age,
            "Fare": fare, "Embarked": embarked, "Survived": survived,
        })
        df.loc[rng.choice(n, 35, replace=False), "Age"] = np.nan
        return df, "Survived"
    if name == "California Housing Regression":
        try:
            data = fetch_california_housing(as_frame=True)
            df = data.frame.rename(columns={"MedHouseVal": "target_house_value"})
            return df, "target_house_value"
        except Exception:
            data = load_diabetes(as_frame=True)
            return data.frame.rename(columns={"target": "target_house_value"}), "target_house_value"
    if name == "Diabetes Regression":
        data = load_diabetes(as_frame=True)
        return data.frame.rename(columns={"target": "disease_progression"}), "disease_progression"
    return pd.DataFrame(), None


def metric_direction(metric: str) -> bool:
    """True when higher values are better."""
    return metric.lower() not in {"mae", "mse", "rmse", "mape", "training time"}