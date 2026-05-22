from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from preprocessing import load_data
from models import create_tree, create_logistic_regression, create_random_forest
from evaluation import evaluate_model

from pathlib import Path
OUTPUT_DIR = Path("outputs")

RANDOM_STATE = 0

# Model search spaces
TREE_DEPTHS = [3, 6, 9]
FOREST_DEPTHS = [3, 6, 9]
LR_ITERATIONS = [1000]

def create_experiments():
    experiments: List[Dict[str, Any]] = []

    for depth in TREE_DEPTHS:
        model = create_tree(depth)
        experiments.append(
            {
                "model_name": "Decision Tree",
                "parameter": "max_depth",
                "value": depth,
                "model": model,
            }
        )

    for depth in FOREST_DEPTHS:
        model = create_random_forest(depth)
        experiments.append(
            {
                "model_name": "Random Forest",
                "parameter": "max_depth",
                "value": depth,
                "model": model,
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


def run_experiments() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()

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
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Rank by recall first, then ROC-AUC, then accuracy.
    sort_cols = ["Recall", "ROC-AUC", "Accuracy"]
    sort_cols = [c for c in sort_cols if c in results_df.columns]
    if sort_cols:
        results_df = results_df.sort_values(by=sort_cols, ascending=False)

    results_path = OUTPUT_DIR / "model_results.csv"
    results_df.drop(columns="Features").to_csv(results_path, index=False)

    for row in results_df.itertuples():
        feature_df = row.Features
        if feature_df is not None:
            filename = (
                f"{row.Model}_{row.Setting}_features.csv"
                .replace(" ", "_").lower()
            )
            feature_df.to_csv(
                OUTPUT_DIR / filename,
                index=False
            )

    return results_df, y