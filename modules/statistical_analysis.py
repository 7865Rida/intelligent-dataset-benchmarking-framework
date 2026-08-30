"""Descriptive statistical analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numerical_statistics(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numerical_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "Feature": col, "Count": int(s.count()), "Mean": s.mean(), "Median": s.median(),
            "Std Dev": s.std(), "Variance": s.var(), "Min": s.min(), "Max": s.max(),
            "Q1": q1, "Q3": q3, "IQR": q3 - q1, "Skewness": s.skew(), "Kurtosis": s.kurtosis(),
        })
    return pd.DataFrame(rows)


def categorical_statistics(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in categorical_cols:
        s = df[col].dropna()
        mode = s.mode()
        top = mode.iloc[0] if not mode.empty else None
        freq = int((s == top).sum()) if top is not None else 0
        rows.append({
            "Feature": col, "Unique Values": int(s.nunique()), "Most Frequent": top,
            "Frequency": freq, "Top Percentage": (freq / len(s) * 100) if len(s) else 0,
        })
    return pd.DataFrame(rows)


def categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    counts = df[column].astype(str).value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = [column, "Count"]
    counts["Percentage"] = counts["Count"] / len(df) * 100
    return counts