"""Automated model training and benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from modules.evaluation_metrics import classification_metrics, regression_metrics
from modules.preprocessing import build_preprocessor
from utils.helpers import metric_direction


CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=160, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=160, random_state=42, n_jobs=-1),
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Support Vector Regressor": SVR(),
    "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=160, random_state=42, n_jobs=-1),
}


@dataclass
class BenchmarkResult:
    table: pd.DataFrame
    pipelines: dict[str, Pipeline]
    failures: dict[str, str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: dict[str, Any]


def available_models(task_type: str) -> dict[str, Any]:
    return CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS


def scoring_name(task_type: str, metric: str) -> str:
    if task_type == "classification":
        return {"Accuracy": "accuracy", "F1": "f1_weighted", "ROC-AUC": "roc_auc_ovr_weighted"}.get(metric, "accuracy")
    return {"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"}.get(metric, "r2")


def run_benchmark(df: pd.DataFrame, target: str, task_type: str, test_size: float, cv_folds: int,
                  random_state: int, selected_models: list[str], primary_metric: str,
                  progress_callback=None) -> BenchmarkResult:
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target]
    if task_type == "classification":
        y = y.astype(str) if not pd.api.types.is_numeric_dtype(y) else y
    stratify = y if task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    models = available_models(task_type)
    rows, pipelines, failures, predictions = [], {}, {}, {}
    for index, name in enumerate(selected_models, start=1):
        if progress_callback:
            progress_callback(index - 1, len(selected_models), name)
        try:
            preprocessor, _, _ = build_preprocessor(X_train)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", clone(models[name]))])
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            y_pred = pipeline.predict(X_test)
            predictions[name] = y_pred
            if task_type == "classification":
                y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
                metrics = classification_metrics(y_test, y_pred, y_proba)
            else:
                metrics = regression_metrics(y_test, y_pred)
            cv_mean, cv_std, cv_min, cv_max = np.nan, np.nan, np.nan, np.nan
            if len(X_train) >= cv_folds and (task_type != "classification" or y_train.value_counts().min() >= cv_folds):
                splitter = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state) if task_type == "classification" else KFold(cv_folds, shuffle=True, random_state=random_state)
                scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring=scoring_name(task_type, primary_metric), error_score=np.nan)
                if primary_metric in {"RMSE", "MAE"}:
                    scores = -scores
                cv_mean, cv_std, cv_min, cv_max = np.nanmean(scores), np.nanstd(scores), np.nanmin(scores), np.nanmax(scores)
            row = {"Model": name, **metrics, "CV Mean": cv_mean, "CV Std": cv_std, "CV Min": cv_min, "CV Max": cv_max, "Training Time": train_time}
            rows.append(row)
            pipelines[name] = pipeline
        except Exception as exc:
            failures[name] = str(exc)
    table = pd.DataFrame(rows)
    if not table.empty:
        rank_col = "R2" if primary_metric == "R2" else primary_metric
        ascending = not metric_direction(primary_metric)
        table = table.sort_values(rank_col, ascending=ascending, na_position="last").reset_index(drop=True)
        table.insert(0, "Rank", range(1, len(table) + 1))
    if progress_callback:
        progress_callback(len(selected_models), len(selected_models), "Completed")
    return BenchmarkResult(table, pipelines, failures, X_train, X_test, y_train, y_test, predictions)