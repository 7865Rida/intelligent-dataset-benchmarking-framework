"""Data-driven scientific insight generation."""

from __future__ import annotations

import pandas as pd


def generate_insights(evaluation: dict | None, corr_pairs: pd.DataFrame | None, benchmark: pd.DataFrame | None,
                      feature_importance: pd.DataFrame | None, task_type: str | None, primary_metric: str | None) -> list[str]:
    insights: list[str] = []
    if evaluation:
        rows, cols = evaluation["shape"]
        missing = evaluation["missing_values"]
        cells = max(rows * cols, 1)
        insights.append(f"The dataset contains {rows:,} samples and {cols:,} columns with a quality score of {evaluation['quality_score']}/100.")
        insights.append(f"Missing data represents {(missing / cells) * 100:.2f}% of all cells and is handled inside the training pipeline using imputation fitted only on training data.")
        if evaluation["duplicate_rows"]:
            insights.append(f"The dataset contains {evaluation['duplicate_rows']:,} duplicate rows, which may affect generalization if duplicates represent repeated observations.")
    if corr_pairs is not None and not corr_pairs.empty:
        strong = corr_pairs[corr_pairs["Correlation"].abs() >= 0.7]
        insights.append(f"The numerical feature set contains {len(strong)} highly correlated feature pairs using an absolute Pearson threshold of 0.70.")
        if not strong.empty:
            top = strong.iloc[0]
            insights.append(f"{top['Feature A']} and {top['Feature B']} show the strongest observed correlation ({top['Correlation']:.3f}).")
    if benchmark is not None and not benchmark.empty and primary_metric:
        metric_col = "R2" if primary_metric == "R2" else primary_metric
        best = benchmark.iloc[0]
        insights.append(f"The {best['Model']} model achieved the best {primary_metric} score of {best[metric_col]:.4f} on the holdout test set.")
        stable = benchmark.dropna(subset=["CV Std"]).sort_values("CV Std")
        if not stable.empty:
            row = stable.iloc[0]
            insights.append(f"{row['Model']} demonstrated the lowest cross-validation standard deviation ({row['CV Std']:.4f}), indicating the most stable validation performance among completed models.")
    if feature_importance is not None and not feature_importance.empty:
        top = feature_importance.iloc[0]
        insights.append(f"{top['Feature']} is the most influential feature for the selected model based on the computed importance score ({top['Importance']:.4f}).")
    if task_type:
        insights.append(f"The selected target is treated as a {task_type} task, so metrics and ranking rules are selected accordingly.")
    return insights


def final_recommendation(benchmark: pd.DataFrame, primary_metric: str) -> str:
    if benchmark.empty:
        return "No recommendation is available because no model completed successfully."
    metric_col = "R2" if primary_metric == "R2" else primary_metric
    best = benchmark.iloc[0]
    stable_text = "cross-validation stability was unavailable"
    if pd.notna(best.get("CV Std")):
        stable_text = f"its CV standard deviation was {best['CV Std']:.4f}"
    return (f"{best['Model']} is recommended because it ranked first on the selected primary metric "
            f"({primary_metric} = {best[metric_col]:.4f}), completed training in {best['Training Time']:.3f} seconds, "
            f"and {stable_text}. This recommendation balances predictive performance, validation consistency and computational cost.")