"""Feature importance extraction from fitted pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from modules.preprocessing import transformed_feature_names


def model_importance(pipeline, top_n: int = 25) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline.named_steps["preprocessor"])
    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    if values is None:
        return pd.DataFrame()
    n = min(len(names), len(values))
    df = pd.DataFrame({"Feature": names[:n], "Importance": values[:n]})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True).assign(Rank=lambda d: range(1, len(d) + 1))


def permutation_importance_table(pipeline, X_test, y_test, scoring: str, top_n: int = 25) -> pd.DataFrame:
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1)
    df = pd.DataFrame({"Feature": X_test.columns, "Importance": result.importances_mean, "Std": result.importances_std})
    return df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True).assign(Rank=lambda d: range(1, len(d) + 1))