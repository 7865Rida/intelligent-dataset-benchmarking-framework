"""Correlation and relationship analysis."""

from __future__ import annotations

import pandas as pd


def correlation_matrix(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    if len(numerical_cols) < 2:
        return pd.DataFrame()
    return df[numerical_cols].corr(method="pearson")


def correlation_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    rows = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1:]:
            value = corr.loc[col_a, col_b]
            strength = "Strong positive" if value >= threshold else "Strong negative" if value <= -threshold else "Weak/Moderate"
            rows.append({"Feature A": col_a, "Feature B": col_b, "Correlation": value, "Relationship": strength})
    return pd.DataFrame(rows).sort_values("Correlation", key=lambda s: s.abs(), ascending=False)