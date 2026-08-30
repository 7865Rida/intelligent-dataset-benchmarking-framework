"""Dataset inspection, feature typing and quality scoring."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.helpers import dataframe_memory_mb


ID_PATTERN = re.compile(r"(^id$|_id$|id_|identifier|uuid|student_id|customer_id|employee_id)", re.I)


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_features: list[str] = []
    for col in df.columns:
        if col in numerical:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.75 and df[col].nunique(dropna=True) > 1:
            date_features.append(col)
            if col in categorical:
                categorical.remove(col)
    ids = [c for c in df.columns if ID_PATTERN.search(str(c)) or df[c].nunique(dropna=True) == len(df)]
    return {"numerical": numerical, "categorical": categorical, "date": date_features, "id": ids}


def suggest_targets(df: pd.DataFrame) -> list[str]:
    preferred = ["target", "label", "class", "y", "outcome", "diagnosis", "survived"]
    suggestions = [c for c in df.columns if str(c).lower() in preferred]
    if len(df.columns) > 0 and df.columns[-1] not in suggestions:
        suggestions.append(df.columns[-1])
    return suggestions


def determine_task_type(y: pd.Series) -> str | None:
    y_clean = y.dropna()
    if y_clean.nunique() < 2:
        return None
    if not pd.api.types.is_numeric_dtype(y_clean):
        return "classification"
    unique_ratio = y_clean.nunique() / max(len(y_clean), 1)
    if y_clean.nunique() <= 20 or unique_ratio <= 0.05:
        return "classification"
    return "regression"


def quality_report(df: pd.DataFrame) -> tuple[int, pd.DataFrame, list[str]]:
    rows, cols = df.shape
    cells = max(rows * cols, 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    invalid_numeric = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        invalid_numeric += int(np.isinf(df[col]).sum())
    high_cardinality = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique(dropna=True) > max(50, rows * 0.5)]

    score = 100
    score -= min(35, (missing / cells) * 140)
    score -= min(20, (duplicates / max(rows, 1)) * 80)
    score -= min(15, len(constant_cols) * 5)
    score -= min(15, invalid_numeric * 2)
    score -= min(15, len(high_cardinality) * 4)
    score = int(max(0, round(score)))

    def status(count: int, critical_at: int | None = None) -> str:
        if count == 0:
            return "Good"
        if critical_at is not None and count >= critical_at:
            return "Critical"
        return "Warning"

    checks = pd.DataFrame([
        {"Quality Check": "Missing Values", "Status": status(missing, int(cells * 0.2)), "Count": missing},
        {"Quality Check": "Duplicate Rows", "Status": status(duplicates, max(1, int(rows * 0.1))), "Count": duplicates},
        {"Quality Check": "Constant Columns", "Status": status(len(constant_cols)), "Count": len(constant_cols)},
        {"Quality Check": "Invalid Numeric Values", "Status": status(invalid_numeric), "Count": invalid_numeric},
        {"Quality Check": "High-cardinality Categoricals", "Status": status(len(high_cardinality)), "Count": len(high_cardinality)},
    ])
    reasons = [
        f"Missing values: {missing} of {cells} cells.",
        f"Duplicate rows: {duplicates}.",
        f"Constant columns: {', '.join(map(str, constant_cols)) if constant_cols else 'none'}.",
        f"High-cardinality categorical columns: {', '.join(map(str, high_cardinality)) if high_cardinality else 'none'}.",
    ]
    return score, checks, reasons


def evaluate_dataset(df: pd.DataFrame) -> dict:
    types = detect_column_types(df)
    score, checks, reasons = quality_report(df)
    return {
        "shape": df.shape,
        "memory_mb": dataframe_memory_mb(df),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).reset_index().rename(columns={"index": "Column", 0: "Data Type"}),
        "types": types,
        "quality_score": score,
        "quality_checks": checks,
        "quality_reasons": reasons,
        "suggested_targets": suggest_targets(df),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }