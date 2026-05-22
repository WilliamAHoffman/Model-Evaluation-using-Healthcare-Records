from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.tree import plot_tree

from pathlib import Path
OUTPUT_DIR = Path("outputs")

def _save_current_figure(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_model_metrics(
    results_df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    filename: str = "model_metrics_comparison.png",
) -> None:
    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    metrics = [m for m in metrics if m in results_df.columns]
    if not metrics:
        raise ValueError("No requested metrics were found in results dataframe.")

    df = results_df.copy()

    if "Label" not in df.columns:
        df["Label"] = (
            df["Model"].astype(str)
            + " ("
            + df["Setting"].astype(str)
            + ")"
        )

    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=metrics)

    if df.empty:
        raise ValueError("No valid numeric metric values found to plot.")

    df = df.sort_values(by=metrics[-1], ascending=False)

    x = np.arange(len(df))
    width = 0.8 / len(metrics)

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, df[metric].to_numpy(), width=width, label=metric)

    plt.xticks(
        x + width * (len(metrics) - 1) / 2,
        df["Label"],
        rotation=45,
        ha="right"
    )
    plt.ylabel("Score")
    plt.title("Model Metric Comparison")
    plt.legend()

    _save_current_figure(filename)

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: Sequence[str] = ("No Diabetes", "Diabetes"),
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
) -> None:
    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    _save_current_figure(filename)


def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    title: str = "ROC Curve",
    filename: str = "roc_curve.png",
) -> None:
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    _save_current_figure(filename)

def plot_class_distribution(
    y: Sequence[int],
    title: str = "Class Distribution",
    filename: str = "class_distribution.png",
) -> None:
    values = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["No Diabetes", "Diabetes"], values.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)

    _save_current_figure(filename)

def plot_features(
    features: pd.DataFrame,
    model_name: str,
    filename: str,
    top_n: int = 10,
) -> None:

    if features is None or features.empty:
        return

    filename = filename.replace(" ", "_").lower()

    feature_type = features.iloc[0]["Type"]

    features = (
        features
        .sort_values(by="Value", ascending=False)
        .head(top_n)
        .sort_values(by="Value", ascending=True)
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        features["Feature"],
        features["Value"]
    )

    plt.xlabel(feature_type)
    plt.ylabel("Feature")

    plt.title(
        f"{model_name} {feature_type} Analysis"
    )

    _save_current_figure(
        filename + "_features.png"
    )

def generate_visualizations(results_df, y):
    results_df["Label"] = (
        results_df["Model"].astype(str)
        + " ("
        + results_df["Setting"].astype(str)
        + ")"
    )

    plot_class_distribution(y)

    plot_model_metrics(results_df)

    for row in results_df.itertuples():
        plot_features(features=row.Features, filename=row.Label, model_name=row.Label)
