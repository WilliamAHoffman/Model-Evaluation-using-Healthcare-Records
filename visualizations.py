from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_current_figure(filename: str) -> None:
    """Save the current matplotlib figure to the outputs directory."""
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:

    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe.")

    df = results_df.copy()
    df = df.sort_values(by=metric, ascending=False)

    plt.figure(figsize=(10, 5))

    plt.bar(df["Label"], df[metric])

    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(title or f"{metric} Comparison")

    plt.xticks(rotation=45, ha="right")

    _save_current_figure(
        filename or f"{metric.lower().replace('-', '_')}_comparison.png"
    )


def plot_model_metrics(
    results_df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    filename: str = "model_metrics_comparison.png",
) -> None:
    """Plot several metrics side by side for model comparison."""
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
    """Plot a confusion matrix using sklearn's display helper."""
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
    """Plot a ROC curve from true labels and predicted probabilities."""
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title(title)
    _save_current_figure(filename)


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 10,
    title: str = "Feature Importance",
    filename: str = "feature_importance.png",
) -> None:
    """Plot the top-N feature importances."""
    importance_df = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance": list(importances),
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    importance_df = importance_df.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.title(title)

    _save_current_figure(filename)


def plot_class_distribution(
    y: Sequence[int],
    title: str = "Class Distribution",
    filename: str = "class_distribution.png",
) -> None:
    """Plot the distribution of target classes."""
    values = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["No Diabetes", "Diabetes"], values.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)

    _save_current_figure(filename)

def generate_visualizations(results_df, y):
    results_df["Label"] = (
        results_df["Model"].astype(str)
        + " ("
        + results_df["Setting"].astype(str)
        + ")"
    )

    plot_class_distribution(y)

    plot_model_metrics(results_df)

    plot_metric_comparison(
        results_df,
        metric="Recall"
    )

    plot_metric_comparison(
        results_df,
        metric="Accuracy"
    )
