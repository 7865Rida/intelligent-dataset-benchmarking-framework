# Intelligent Dataset Benchmarking Framework (IDBF)

## Project Overview

Intelligent Dataset Benchmarking Framework is a complete Python and Streamlit application for automated dataset evaluation, statistical analysis, machine learning benchmarking, feature importance analysis, model comparison, and scientific report generation.

The application runs with:

```bash
streamlit run app.py
```

## Problem Statement

Manual machine learning experimentation can be slow and inconsistent. Users often need to inspect data quality, choose preprocessing steps, train several algorithms, compare metrics, evaluate stability, and document the results. IDBF automates this workflow in one reproducible research dashboard.

## Objectives

- Upload or load a sample dataset.
- Inspect data schema, feature types, missing values, duplicates and quality issues.
- Generate numerical and categorical statistics.
- Analyze correlations and possible multicollinearity.
- Select a target variable and automatically identify classification or regression.
- Build leakage-safe preprocessing pipelines with Scikit-learn.
- Benchmark multiple machine learning algorithms.
- Rank models using user-selected primary metrics.
- Explain model performance, stability and feature importance.
- Export the leaderboard, best model pipeline and PDF research report.

## Features

- Professional dark Streamlit dashboard.
- Sidebar navigation for overview, dataset evaluation, statistics, correlations, benchmarking, leaderboard, feature importance, model comparison and research report.
- KPI cards for rows, columns, feature counts, missing values, duplicate rows, best model and best score.
- Data quality score from 0 to 100 with detailed checks.
- Automatic numerical, categorical, date and ID-like column detection.
- Target suggestions based on common names and the last column.
- Classification and regression task detection.
- Train/test split with configurable test size and random state.
- Cross-validation using 3, 5 or 10 folds when sample size supports it.
- Individual model failure handling so one failed model does not stop the benchmark.
- Plotly visualizations for distributions, correlations, leaderboard and comparisons.
- Prediction interface using the fitted preprocessing plus model pipeline.
- Joblib download for the best model pipeline.
- ReportLab PDF report generation.

## Modules

- `modules/dataset_evaluator.py`: dataset inspection, feature typing, target suggestions, task detection and quality scoring.
- `modules/statistical_analysis.py`: descriptive statistics and categorical distributions.
- `modules/correlation_analysis.py`: Pearson correlation matrix and pairwise relationship extraction.
- `modules/preprocessing.py`: Scikit-learn `ColumnTransformer` with numerical and categorical preprocessing.
- `modules/model_benchmark.py`: algorithm registry, train/test split, fitting, metrics, cross-validation and ranking.
- `modules/evaluation_metrics.py`: classification and regression metric functions.
- `modules/feature_importance.py`: native model importance, linear coefficients and permutation importance.
- `modules/leaderboard.py`: leaderboard construction and highlight extraction.
- `modules/insight_generator.py`: data-driven scientific insights and final recommendation.
- `modules/report_generator.py`: PDF research report generation.
- `utils/helpers.py`: shared utilities, sample datasets and serialization helpers.

## System Architecture

The Streamlit UI in `app.py` controls session state and navigation. Data is loaded from CSV or Scikit-learn sample datasets. The evaluator module profiles the dataset. After target selection, preprocessing and models are composed into Scikit-learn pipelines, ensuring preprocessing is fitted only on training data. Benchmark results are stored in `st.session_state` and reused across pages without unnecessary retraining.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn
- Joblib
- ReportLab
- OpenPyXL

## Machine Learning Algorithms

Classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes
- Extra Trees Classifier

Regression models:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor
- Extra Trees Regressor

## Evaluation Metrics

Classification:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC when available
- Confusion matrix and classification report support in the app internals

Regression:

- MAE
- MSE
- RMSE
- R2
- MAPE when valid

Ranking direction is handled correctly: higher is better for Accuracy, Precision, Recall, F1, ROC-AUC and R2; lower is better for MAE, MSE, RMSE and MAPE.

## Dataset Handling

Users can upload arbitrary CSV files. The app also includes sample datasets for Iris, Breast Cancer, Titanic-style classification, California Housing and Diabetes regression. The California Housing loader falls back safely if network access is unavailable.

## Feature Importance

Tree-based models use `feature_importances_`. Linear models use absolute coefficients. Permutation importance is also available and maps scores to original input columns.

## Leaderboard

The leaderboard ranks completed models by the selected primary metric and displays best model, best score, fastest model and most stable model based on cross-validation standard deviation.

## Report Generation

The PDF report includes executive summary, dataset quality, scientific insights, benchmark table, leaderboard, feature importance, stability analysis and a final recommendation based on performance, stability, training time and generalization indicators.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

## Project Structure

```text
Intelligent_Dataset_Benchmarking_Framework/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_datasets/
├── modules/
│   ├── __init__.py
│   ├── dataset_evaluator.py
│   ├── statistical_analysis.py
│   ├── preprocessing.py
│   ├── correlation_analysis.py
│   ├── model_benchmark.py
│   ├── evaluation_metrics.py
│   ├── feature_importance.py
│   ├── leaderboard.py
│   ├── insight_generator.py
│   └── report_generator.py
├── utils/
│   └── helpers.py
└── .streamlit/
    └── config.toml
```

## Screenshots Placeholder

Add screenshots after running the application:

- Overview dashboard
- Dataset quality report
- Correlation heatmap
- Benchmark table
- Leaderboard
- Feature importance chart
- PDF report page

## System Workflow

Dataset upload or sample selection leads to dataset evaluation. The user selects a target variable and experiment settings. The benchmarking engine splits the data, fits preprocessing only on the training split, trains selected models, evaluates metrics, performs cross-validation, ranks results, generates feature importance, compares models and produces downloadable artifacts.

## Viva Questions and Answers

1. What problem does IDBF solve?
   It automates dataset profiling, model benchmarking, comparison and reporting for classification and regression datasets.

2. How does the system prevent data leakage?
   Preprocessing is inside a Scikit-learn pipeline and is fitted only on the training data during train/test evaluation and cross-validation.

3. Why use `ColumnTransformer`?
   It applies different preprocessing steps to numerical and categorical columns in one reproducible pipeline.

4. How is task type detected?
   Non-numeric targets are classification. Numeric targets with few unique values or low unique ratio are classification; otherwise regression.

5. How are models ranked?
   Models are sorted by the selected primary metric with the correct higher-is-better or lower-is-better direction.

6. What happens if a model fails?
   The failure is captured and displayed, while other models continue training.

7. Why is cross-validation useful?
   It estimates validation stability across different folds rather than relying only on one holdout split.

8. What does lower CV standard deviation mean?
   It means the model is more consistent across validation folds.

9. How is the best model saved?
   The complete fitted preprocessing and model pipeline is serialized with Joblib.

10. Why is permutation importance included?
    It provides a model-agnostic estimate of feature influence on predictive performance.

## Limitations

- Very large datasets may require sampling or additional performance tuning.
- Automated target type detection can be reviewed and corrected by selecting the target manually.
- Some metrics such as ROC-AUC are unavailable for unsupported model or target configurations.
- Generated insights are based on available computed statistics and do not replace domain expert review.

## Future Scope

- Add automated hyperparameter tuning.
- Add SHAP-based explainability.
- Add time-series benchmarking.
- Add database connectors.
- Add experiment tracking and model registry integration.
- Add fairness and bias analysis.
- Add automated data cleaning suggestions.

## Conclusion

IDBF provides a complete scientific workflow from dataset inspection to model benchmarking, evaluation, leaderboard generation, feature importance analysis and downloadable reporting using Python and Streamlit only.