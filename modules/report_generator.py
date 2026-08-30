"""Report generation with ReportLab."""

from __future__ import annotations

import io

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from modules.insight_generator import final_recommendation


def _table_from_df(df: pd.DataFrame, max_rows: int = 12) -> Table:
    small = df.head(max_rows).copy()
    small = small.fillna("N/A")
    for col in small.columns:
        if pd.api.types.is_numeric_dtype(small[col]):
            small[col] = small[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    data = [small.columns.tolist()] + small.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    return table


def generate_pdf_report(evaluation: dict, task_type: str, target: str, benchmark: pd.DataFrame,
                        leaderboard: pd.DataFrame, importance: pd.DataFrame, insights: list[str],
                        primary_metric: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [Paragraph("Intelligent Dataset Benchmarking Framework Report", styles["Title"]), Spacer(1, 12)]
    rows, cols = evaluation.get("shape", (0, 0))
    summary = [
        f"Dataset samples: {rows:,}", f"Dataset columns: {cols:,}", f"Target variable: {target}",
        f"Task type: {task_type}", f"Quality score: {evaluation.get('quality_score', 'N/A')}/100",
    ]
    if not benchmark.empty:
        best = benchmark.iloc[0]
        metric_col = "R2" if primary_metric == "R2" else primary_metric
        summary.extend([f"Best model: {best['Model']}", f"Best {primary_metric}: {best[metric_col]:.4f}"])
    story.append(Paragraph("Executive Summary", styles["Heading2"]))
    for item in summary:
        story.append(Paragraph(item, styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Dataset Quality", styles["Heading2"]))
    story.append(_table_from_df(evaluation["quality_checks"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Scientific Insights", styles["Heading2"]))
    for insight in insights:
        story.append(Paragraph(f"- {insight}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    if not benchmark.empty:
        story.append(Paragraph("Model Benchmarking", styles["Heading2"]))
        story.append(_table_from_df(benchmark))
        story.append(Spacer(1, 10))
    if not leaderboard.empty:
        story.append(Paragraph("Leaderboard", styles["Heading2"]))
        story.append(_table_from_df(leaderboard))
        story.append(Spacer(1, 10))
    if importance is not None and not importance.empty:
        story.append(Paragraph("Feature Importance", styles["Heading2"]))
        story.append(_table_from_df(importance))
        story.append(Spacer(1, 10))
    story.append(Paragraph("Final Recommendation", styles["Heading2"]))
    story.append(Paragraph(final_recommendation(benchmark, primary_metric), styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer.read()