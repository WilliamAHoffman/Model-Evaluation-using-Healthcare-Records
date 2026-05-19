from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from preprocessing import load_data
from models import create_tree, create_knn, create_logistic_regression
from evaluation import evaluate_model

OUTPUT_DIR = Path("output")
RANDOM_STATE = 0

# Model search spaces
TREE_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
K_VALUES = [3, 5, 7]
LR_ITERATIONS = [1000, 2000]


def create_experiments():
    experiments: List[Dict[str, Any]] = []

    for depth in TREE_DEPTHS:
        experiments.append(
            {
                "model_name": "Decision Tree",
                "parameter": "max_depth",
                "value": depth,
                "model": create_tree(depth),
            }
        )

    for k in K_VALUES:
        experiments.append(
            {
                "model_name": "KNN",
                "parameter": "k",
                "value": k,
                "model": create_knn(k),
            }
        )

    for iterations in LR_ITERATIONS:
        experiments.append(
            {
                "model_name": "Logistic Regression",
                "parameter": "max_iter",
                "value": iterations,
                "model": create_logistic_regression(iterations),
            }
        )

    return experiments


def run_tests() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y, X = load_data()

    print(f"Loaded dataset with {len(X):,} rows and {X.shape[1]} features.")
    print("Class counts:")
    print(y.value_counts().sort_index())
    print()

    experiments = create_experiments()
    results: List[Dict[str, Any]] = []

    for exp in experiments:
        print(
            f"Running {exp['model_name']} | "
            f"{exp['parameter']}={exp['value']}"
        )
        result = evaluate_model(
            model=exp["model"],
            X=X,
            y=y,
            model_name=exp["model_name"],
            param_name=exp["parameter"],
            param_value=exp["value"],
            n_splits=5,
            random_state=RANDOM_STATE,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Rank by recall first, then ROC-AUC, then accuracy.
    sort_cols = ["Recall", "ROC-AUC", "Accuracy"]
    sort_cols = [c for c in sort_cols if c in results_df.columns]
    if sort_cols:
        results_df = results_df.sort_values(by=sort_cols, ascending=False)

    results_path = OUTPUT_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    print()
    print("Saved results to:", results_path)
    print()
    print(results_df[["Model", "Parameter", "Setting", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_string(index=False))

    best_row = results_df.iloc[0]
    print()
    print("Best model configuration:")
    print(
        f"{best_row['Model']} | {best_row['Parameter']}={best_row['Setting']} | "
        f"Recall={best_row['Recall']:.4f} | Accuracy={best_row['Accuracy']:.4f}"
    )