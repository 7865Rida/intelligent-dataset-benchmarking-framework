"""Intelligent Dataset Benchmarking Framework Streamlit application."""
 
from __future__ import annotations
import base64
 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
 
from modules.correlation_analysis import correlation_matrix, correlation_pairs
from modules.dataset_evaluator import determine_task_type, evaluate_dataset
from modules.feature_importance import model_importance, permutation_importance_table
from modules.insight_generator import generate_insights
from modules.leaderboard import build_leaderboard, leaderboard_highlights
from modules.model_benchmark import available_models, run_benchmark, scoring_name
from modules.report_generator import generate_pdf_report
from modules.statistical_analysis import categorical_distribution, categorical_statistics, numerical_statistics
from utils.helpers import load_sample_dataset, normalize_columns, safe_sample, serialize_joblib, to_percent
 
 
st.set_page_config(page_title="IDBF", page_icon="🧪", layout="wide")
 
# ---------------------------------------------------------------------------
# Registrant details shown on the landing page
# ---------------------------------------------------------------------------
FULL_NAME = "Sayyed Rida Fatima Sharif"
EMAIL = "sayyedridafatima32@gmail.com"
 
 
def set_background(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
 
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(5, 10, 20, 0.72),
                    rgba(5, 10, 20, 0.72)
                ),
                url("data:image/avif;base64,{encoded}");
 
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
 
 
set_background(
    r"C:\Users\midha\OneDrive\Desktop\SparkIIT internship\intelligent-dataset-benchmarking-framework (1)\assets\intelligent.avif"
)
 
 
def init_state() -> None:
    defaults = {
        "df": None, "dataset_name": None, "evaluation": None, "target": None, "task_type": None,
        "benchmark": None, "pipelines": {}, "failures": {}, "splits": None, "predictions": {},
        "primary_metric": None, "importance": pd.DataFrame(), "corr_pairs": pd.DataFrame(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
 
 
def css() -> None:
    st.markdown(
        """
        <style>
        .main-title{font-size:2.4rem;font-weight:800;color:#e0f2fe;margin-bottom:.2rem}
        .subtitle{color:#94a3b8;font-size:1.05rem;margin-bottom:1.5rem}
        div[data-testid="stMetric"]{background:#111827;border:1px solid #243041;border-radius:14px;padding:14px}
        .status-line{padding:.45rem .7rem;border-left:3px solid #38bdf8;background:#111827;margin:.25rem 0;border-radius:8px}
        .good{color:#22c55e}.warn{color:#f59e0b}.bad{color:#ef4444}
        .welcome-card{background:#111827;border:1px solid #243041;border-left:4px solid #38bdf8;border-radius:12px;padding:14px 18px;margin-bottom:1.2rem}
        .welcome-name{font-size:1.05rem;font-weight:700;color:#e0f2fe}
        .welcome-email{font-size:0.9rem;color:#94a3b8}
        </style>
        """,
        unsafe_allow_html=True,
    )
 
 
def welcome_card() -> None:
    st.markdown(
        f"""
        <div class="welcome-card">
            <div class="welcome-name">Name: {FULL_NAME}</div>
            <div class="welcome-email">Email: {EMAIL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def load_data_panel() -> None:
    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose dataset", ["Upload CSV", "Iris Classification", "Breast Cancer Classification", "Titanic-style Classification", "California Housing Regression", "Diabetes Regression"])
    df, target = None, None
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                target = None
            except pd.errors.EmptyDataError:
                st.sidebar.error("The uploaded CSV is empty.")
            except Exception as exc:
                st.sidebar.error(f"Could not read CSV: {exc}")
    else:
        df, target = load_sample_dataset(source)
    if df is not None and not df.empty:
        df = normalize_columns(df)
        if st.session_state.dataset_name != source or st.session_state.df is None or not st.session_state.df.equals(df):
            st.session_state.df = df
            st.session_state.dataset_name = source
            st.session_state.evaluation = evaluate_dataset(df)
            st.session_state.target = target
            st.session_state.task_type = determine_task_type(df[target]) if target in df.columns else None
            st.session_state.benchmark = None
            st.session_state.pipelines = {}
            st.session_state.importance = pd.DataFrame()
 
 
def sidebar_config() -> None:
    df = st.session_state.df
    if df is None:
        return
    st.sidebar.header("Experiment Settings")
    suggestions = st.session_state.evaluation["suggested_targets"]
    default_index = df.columns.get_loc(st.session_state.target) if st.session_state.target in df.columns else (df.columns.get_loc(suggestions[0]) if suggestions else 0)
    target = st.sidebar.selectbox("Select Target Variable", df.columns, index=int(default_index))
    st.session_state.target = target
    st.session_state.task_type = determine_task_type(df[target])
    if st.session_state.task_type is None:
        st.sidebar.error("Target must contain at least two valid classes/values.")
        return
    st.sidebar.success(f"Task Type: {st.session_state.task_type.title()}")
    test_size = st.sidebar.selectbox("Test Size", [0.10, 0.20, 0.25, 0.30], index=1, format_func=lambda x: f"{int(x*100)}%")
    cv_folds = st.sidebar.selectbox("Cross Validation", [3, 5, 10], index=1, format_func=lambda x: f"{x}-fold")
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    metric_options = ["Accuracy", "F1", "ROC-AUC"] if st.session_state.task_type == "classification" else ["R2", "RMSE", "MAE"]
    primary_metric = st.sidebar.selectbox("Primary Metric", metric_options)
    st.session_state.primary_metric = primary_metric
    model_names = list(available_models(st.session_state.task_type).keys())
    selected = st.sidebar.multiselect("Models", model_names, default=model_names)
    if st.sidebar.button("Run Benchmark", type="primary", use_container_width=True):
        if len(df) < 8:
            st.sidebar.error("At least 8 rows are recommended for benchmarking.")
            return
        if not selected:
            st.sidebar.error("Select at least one model.")
            return
        with st.spinner("Benchmarking models..."):
            bar = st.progress(0, text="Starting benchmark")
 
            def progress(done: int, total: int, name: str) -> None:
                bar.progress(done / max(total, 1), text=f"Benchmarking: {name}")
 
            result = run_benchmark(df, target, st.session_state.task_type, test_size, cv_folds, int(random_state), selected, primary_metric, progress)
            st.session_state.benchmark = result.table
            st.session_state.pipelines = result.pipelines
            st.session_state.failures = result.failures
            st.session_state.splits = (result.X_train, result.X_test, result.y_train, result.y_test)
            st.session_state.predictions = result.predictions
            if not result.table.empty:
                st.sidebar.success("Benchmark completed.")
            if result.failures:
                st.sidebar.warning(f"{len(result.failures)} model(s) failed but the benchmark continued.")
 
 
def header() -> None:
    st.markdown('<div class="main-title">Intelligent Dataset Benchmarking Framework</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated statistics, analytics, model benchmarking, feature importance and scientific reporting.</div>', unsafe_allow_html=True)
 
 
def kpis() -> None:
    ev, table = st.session_state.evaluation, st.session_state.benchmark
    if not ev:
        st.info("Upload a CSV or select a sample dataset from the sidebar to begin.")
        return
    types = ev["types"]
    highlights = leaderboard_highlights(table, st.session_state.primary_metric or "Accuracy") if table is not None else {}
    values = [
        ("Dataset Rows", ev["shape"][0]), ("Dataset Columns", ev["shape"][1]),
        ("Numerical Features", len(types["numerical"])), ("Categorical Features", len(types["categorical"])),
        ("Missing Values", ev["missing_values"]), ("Duplicate Rows", ev["duplicate_rows"]),
        ("Models Benchmarked", 0 if table is None else len(table)), ("Best Model", highlights.get("best_model", "N/A")),
        ("Best Score", "N/A" if highlights.get("best_score") is None else f"{highlights['best_score']:.4f}"),
    ]
    cols = st.columns(3)
    for i, (label, value) in enumerate(values):
        cols[i % 3].metric(label, value)
 
 
def page_overview() -> None:
    header()
    welcome_card()
    st.write("Intelligent Dataset Benchmarking Framework automatically evaluates datasets and scientifically compares machine learning algorithms.")
    kpis()
    stages = [
        ("Dataset Loaded", st.session_state.df is not None), ("Data Quality Checked", st.session_state.evaluation is not None),
        ("Statistics Generated", st.session_state.df is not None), ("Features Analyzed", st.session_state.evaluation is not None),
        ("Models Benchmarked", st.session_state.benchmark is not None), ("Leaderboard Generated", st.session_state.benchmark is not None and not st.session_state.benchmark.empty),
        ("Report Generated", st.session_state.benchmark is not None and not st.session_state.benchmark.empty),
    ]
    st.subheader("Analysis Status")
    for name, done in stages:
        st.markdown(f"<div class='status-line'>{'✅' if done else '⏳'} {name}</div>", unsafe_allow_html=True)
 
 
def page_dataset() -> None:
    header()
    df, ev = st.session_state.df, st.session_state.evaluation
    if df is None:
        st.info("Load a dataset from the sidebar.")
        return
    rows = st.selectbox("Preview rows", [10, 20, 50])
    st.dataframe(df.head(rows), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Shape", f"{df.shape[0]} x {df.shape[1]}")
    c2.metric("Memory Usage", f"{ev['memory_mb']:.2f} MB")
    c3.metric("Data Quality Score", f"{ev['quality_score']} / 100")
    tab1, tab2, tab3 = st.tabs(["Schema", "Feature Categories", "Quality Checks"])
    with tab1:
        st.dataframe(ev["dtypes"], use_container_width=True)
        st.write("Columns:", ", ".join(map(str, ev["columns"])))
    with tab2:
        for key, label in [("numerical", "Numerical Features"), ("categorical", "Categorical Features"), ("date", "Date Features"), ("id", "Possible ID Columns")]:
            st.write(f"**{label}:** {', '.join(map(str, ev['types'][key])) or 'None detected'}")
        if ev["types"]["id"]:
            st.warning("ID-like columns were detected. They are usually not meaningful model features and may leak row identity.")
    with tab3:
        styled = ev["quality_checks"].copy()
        styled["Status"] = styled["Status"].map({"Good": "✅ Good", "Warning": "⚠️ Warning", "Critical": "❌ Critical"})
        st.dataframe(styled, use_container_width=True)
        for reason in ev["quality_reasons"]:
            st.caption(reason)
 
 
def page_statistics() -> None:
    header()
    df, ev = st.session_state.df, st.session_state.evaluation
    if df is None:
        st.info("Load a dataset first.")
        return
    num_cols, cat_cols = ev["types"]["numerical"], ev["types"]["categorical"]
    t1, t2, t3 = st.tabs(["Numerical", "Categorical", "Visualizations"])
    with t1:
        st.dataframe(numerical_statistics(df, num_cols), use_container_width=True)
    with t2:
        st.dataframe(categorical_statistics(df, cat_cols), use_container_width=True)
        if cat_cols:
            col = st.selectbox("Categorical distribution", cat_cols)
            st.dataframe(categorical_distribution(df, col), use_container_width=True)
    with t3:
        if num_cols:
            col = st.selectbox("Numerical feature", num_cols)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}"), use_container_width=True)
            c2.plotly_chart(px.box(df, y=col, title=f"Box Plot of {col}"), use_container_width=True)
        if cat_cols:
            col = st.selectbox("Category chart", cat_cols, key="cat_chart")
            dist = categorical_distribution(df, col)
            st.plotly_chart(px.bar(dist, x=col, y="Count", title=f"Frequency of {col}"), use_container_width=True)
 
 
def page_correlations() -> None:
    header()
    df, ev = st.session_state.df, st.session_state.evaluation
    if df is None:
        st.info("Load a dataset first.")
        return
    num_cols = ev["types"]["numerical"]
    corr = correlation_matrix(df, num_cols)
    if corr.empty:
        st.warning("At least two numerical columns are required for correlation analysis.")
        return
    pairs = correlation_pairs(corr)
    st.session_state.corr_pairs = pairs
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson Correlation Heatmap"), use_container_width=True)
    st.dataframe(pairs, use_container_width=True)
    strong = pairs[pairs["Correlation"].abs() >= 0.7]
    if not strong.empty:
        st.warning(f"Possible multicollinearity: {len(strong)} strongly correlated pair(s) found.")
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        x = c1.selectbox("Scatter X", num_cols)
        y = c2.selectbox("Scatter Y", num_cols, index=1 if len(num_cols) > 1 else 0)
        st.plotly_chart(px.scatter(safe_sample(df), x=x, y=y, title=f"{x} vs {y}"), use_container_width=True)
 
 
def page_benchmark() -> None:
    header()
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset and configure the experiment in the sidebar.")
        return
    st.subheader("Target Column Selection")
    st.write(f"Selected target: **{st.session_state.target}**")
    st.write(f"Task type: **{(st.session_state.task_type or 'Unknown').title()}**")
    if st.session_state.target:
        target_counts = df[st.session_state.target].value_counts().head(25).reset_index()
        target_counts.columns = ["Value", "Count"]
        chart = px.bar(target_counts, x="Value", y="Count", title="Target Distribution") if st.session_state.task_type == "classification" else px.histogram(df, x=st.session_state.target, title="Target Distribution")
        st.plotly_chart(chart, use_container_width=True)
    st.subheader("Benchmark Results")
    table = st.session_state.benchmark
    if table is None:
        st.info("Use the Run Benchmark button in the sidebar.")
    elif table.empty:
        st.error("No model completed successfully.")
    else:
        st.dataframe(table, use_container_width=True)
    if st.session_state.failures:
        with st.expander("Model Failures"):
            st.json(st.session_state.failures)
    best_name = table.iloc[0]["Model"] if table is not None and not table.empty else None
    if best_name:
        st.download_button("Download Best Model", serialize_joblib(st.session_state.pipelines[best_name]), file_name=f"{best_name.replace(' ', '_')}_pipeline.joblib")
        prediction_interface(best_name)
 
 
def prediction_interface(model_name: str) -> None:
    st.subheader("Prediction Interface")
    df, target = st.session_state.df, st.session_state.target
    X = df.drop(columns=[target])
    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(X.columns):
            series = X[col]
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    values[col] = st.number_input(str(col), value=float(series.median() if series.notna().any() else 0.0))
                else:
                    opts = series.dropna().astype(str).unique().tolist()[:100]
                    values[col] = st.selectbox(str(col), opts or [""])
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([values])
        pipe = st.session_state.pipelines[model_name]
        pred = pipe.predict(row)[0]
        st.success(f"Prediction: {pred}")
        if st.session_state.task_type == "classification" and hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(row)[0]
            classes = pipe.named_steps["model"].classes_
            st.dataframe(pd.DataFrame({"Class": classes, "Probability": probs}), use_container_width=True)
 
 
def page_leaderboard() -> None:
    header()
    table = st.session_state.benchmark
    if table is None or table.empty:
        st.info("Run benchmarking first.")
        return
    board = build_leaderboard(table, st.session_state.task_type, st.session_state.primary_metric)
    highlights = leaderboard_highlights(table, st.session_state.primary_metric)
    st.subheader("🏆 MODEL LEADERBOARD")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🥇 Best Performing Model", highlights["best_model"])
    c2.metric("📊 Best Score", f"{highlights['best_score']:.4f}")
    c3.metric("⚡ Fastest Model", highlights["fastest_model"])
    c4.metric("🎯 Most Stable Model", highlights["stable_model"] or "N/A")
    st.dataframe(board, use_container_width=True)
    st.plotly_chart(px.bar(board, x="Model", y="Score", color="Performance", title="Leaderboard Scores"), use_container_width=True)
    st.download_button("Download CSV Leaderboard", board.to_csv(index=False).encode("utf-8"), file_name="idbf_leaderboard.csv", mime="text/csv")
 
 
def page_importance() -> None:
    header()
    table = st.session_state.benchmark
    if table is None or table.empty:
        st.info("Run benchmarking first.")
        return
    model_name = st.selectbox("Select model", table["Model"].tolist())
    pipeline = st.session_state.pipelines[model_name]
    builtin = model_importance(pipeline)
    scoring = scoring_name(st.session_state.task_type, st.session_state.primary_metric)
    X_train, X_test, y_train, y_test = st.session_state.splits
    try:
        perm = permutation_importance_table(pipeline, X_test, y_test, scoring)
    except Exception as exc:
        perm = pd.DataFrame()
        st.warning(f"Permutation importance could not be calculated: {exc}")
    choice = st.radio("Importance method", ["Model Native", "Permutation"], horizontal=True)
    imp = builtin if choice == "Model Native" and not builtin.empty else perm
    st.session_state.importance = imp
    if imp.empty:
        st.warning("The selected model does not expose native importance and permutation importance is unavailable.")
        return
    st.dataframe(imp, use_container_width=True)
    st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title=f"Feature Importance: {model_name}"), use_container_width=True)
    top = imp.iloc[0]
    st.info(f"{top['Feature']} is the most influential feature for the selected model.")
 
 
def page_comparison() -> None:
    header()
    table = st.session_state.benchmark
    if table is None or table.empty:
        st.info("Run benchmarking first.")
        return
    selected = st.multiselect("Select up to 5 models", table["Model"].tolist(), default=table["Model"].head(5).tolist(), max_selections=5)
    comp = table[table["Model"].isin(selected)]
    st.dataframe(comp, use_container_width=True)
    metrics = ["Accuracy", "F1", "ROC-AUC"] if st.session_state.task_type == "classification" else ["RMSE", "R2", "MAE"]
    metric = st.selectbox("Metric comparison", [m for m in metrics if m in comp.columns])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(comp, x="Model", y=metric, title=f"{metric} Comparison"), use_container_width=True)
    c2.plotly_chart(px.bar(comp, x="Model", y="Training Time", title="Training Time Comparison"), use_container_width=True)
    radar_metrics = [m for m in metrics + ["CV Mean"] if m in comp.columns]
    fig = go.Figure()
    for _, row in comp.iterrows():
        vals = [row[m] if pd.notna(row[m]) else 0 for m in radar_metrics]
        fig.add_trace(go.Scatterpolar(r=vals, theta=radar_metrics, fill="toself", name=row["Model"]))
    fig.update_layout(title="Radar-style Metric Comparison", polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)
    for _, row in comp.iterrows():
        with st.expander(row["Model"]):
            st.write(f"Strengths: rank {int(row['Rank'])} with training time {row['Training Time']:.3f}s.")
            st.write(f"Weaknesses: inspect metrics where the model trails stronger alternatives.")
            st.write(f"Stability: CV standard deviation is {row['CV Std']:.4f}" if pd.notna(row["CV Std"]) else "Stability: CV was unavailable for this model.")
 
 
def page_report() -> None:
    header()
    ev, table = st.session_state.evaluation, st.session_state.benchmark
    if ev is None:
        st.info("Load a dataset first.")
        return
    board = build_leaderboard(table, st.session_state.task_type, st.session_state.primary_metric) if table is not None and not table.empty else pd.DataFrame()
    insights = generate_insights(ev, st.session_state.corr_pairs, table, st.session_state.importance, st.session_state.task_type, st.session_state.primary_metric)
    st.subheader("Scientific Insights")
    for insight in insights:
        st.write(f"- {insight}")
    if table is not None and not table.empty:
        st.subheader("Model Stability Analysis")
        st.dataframe(table[["Model", "CV Mean", "CV Std", "CV Min", "CV Max"]], use_container_width=True)
        st.caption("Lower standard deviation indicates more consistent performance across validation folds.")
        pdf = generate_pdf_report(ev, st.session_state.task_type, st.session_state.target, table, board, st.session_state.importance, insights, st.session_state.primary_metric)
        st.download_button("Download Report", pdf, file_name="idbf_research_report.pdf", mime="application/pdf")
        st.download_button("Download CSV Leaderboard", board.to_csv(index=False).encode("utf-8"), file_name="idbf_leaderboard.csv", mime="text/csv")
        best = table.iloc[0]["Model"]
        st.download_button("Download Model", serialize_joblib(st.session_state.pipelines[best]), file_name="idbf_best_model.joblib")
    else:
        st.info("Run benchmarking to include model comparisons, stability and recommendations in the report.")
 
 
def main() -> None:
    init_state()
    css()
    st.sidebar.title("🧪 IDBF")
    st.sidebar.caption("Navigation")
    pages = {
        "🏠 Overview": page_overview, "📂 Dataset Evaluation": page_dataset, "📊 Statistics": page_statistics,
        "🔗 Correlations": page_correlations, "🤖 Benchmarking": page_benchmark, "🏆 Leaderboard": page_leaderboard,
        "🧠 Feature Importance": page_importance, "📈 Model Comparison": page_comparison, "📑 Research Report": page_report,
    }
    page = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")
    load_data_panel()
    sidebar_config()
    pages[page]()
 
 
if __name__ == "__main__":
    main()
 