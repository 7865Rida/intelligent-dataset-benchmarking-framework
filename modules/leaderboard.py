"""Leaderboard ranking helpers."""

from __future__ import annotations

import pandas as pd


def performance_label(score: float, task_type: str, metric: str) -> str:
    if pd.isna(score):
        return "Unavailable"
    if metric in {"RMSE", "MAE", "MSE", "MAPE"}:
        return "Excellent" if score <= 0.1 else "Very Good" if score <= 0.25 else "Good" if score <= 0.5 else "Needs Review"
    if score >= 0.90:
        return "Excellent"
    if score >= 0.80:
        return "Very Good"
    if score >= 0.70:
        return "Good"
    return "Needs Review"


def build_leaderboard(table: pd.DataFrame, task_type: str, primary_metric: str) -> pd.DataFrame:
    if table.empty:
        return table
    metric_col = "R2" if primary_metric == "R2" else primary_metric
    board = table[["Rank", "Model", metric_col, "CV Mean", "CV Std", "Training Time"]].copy()
    board = board.rename(columns={metric_col: "Score"})
    medals = {1: "1", 2: "2", 3: "3"}
    board["Rank"] = board["Rank"].map(lambda r: f"{medals.get(int(r), str(int(r)))}")
    board["Performance"] = board["Score"].apply(lambda x: performance_label(x, task_type, primary_metric))
    return board


def leaderboard_highlights(table: pd.DataFrame, primary_metric: str) -> dict[str, str | float | None]:
    if table.empty:
        return {"best_model": None, "best_score": None, "fastest_model": None, "stable_model": None}
    score_col = "R2" if primary_metric == "R2" else primary_metric
    stable = table.dropna(subset=["CV Std"]).sort_values("CV Std").head(1)
    return {
        "best_model": table.iloc[0]["Model"],
        "best_score": table.iloc[0].get(score_col),
        "fastest_model": table.sort_values("Training Time").iloc[0]["Model"],
        "stable_model": None if stable.empty else stable.iloc[0]["Model"],
    }