"""Metric calculation for classification and regression tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float | None]:
    average = "weighted"
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "ROC-AUC": None,
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            metrics["ROC-AUC"] = None
    return metrics


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    non_zero = np.asarray(y_true) != 0
    mape = float(np.mean(np.abs((np.asarray(y_true)[non_zero] - np.asarray(y_pred)[non_zero]) / np.asarray(y_true)[non_zero]))) if non_zero.any() else np.nan
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred), "MAPE": mape}