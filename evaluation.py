import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

RANDOM_STATE = 0

def evaluate_model(
    model,
    X,
    y,
    model_name,
    param_name=None,
    param_value=None,
    n_splits=5,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    confusion_mats = []
    features = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        confusion_mats.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        if hasattr(fold_model, "predict_proba"):
            y_prob = fold_model.predict_proba(X_test)[:, 1]
            roc_aucs.append(roc_auc_score(y_test, y_prob))

        if hasattr(fold_model, "feature_importances_"):
            feature_df = pd.DataFrame({
                "Feature": X.columns,
                "Value": fold_model.feature_importances_,
                "Type": "Importance"
            })
            features.append(feature_df)

        elif hasattr(fold_model, "named_steps"):
            inner_model = fold_model.named_steps["model"]
            if hasattr(inner_model, "coef_"):
                feature_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Value": abs(inner_model.coef_[0]),
                    "Type": "Coefficient"
                })
                features.append(feature_df)

    mean_features = None
    if features:
        all_features = pd.concat(features, ignore_index=True)

        mean_features = (
            all_features
            .groupby(["Feature", "Type"])["Value"]
            .mean()
            .reset_index()
            .sort_values(by="Value", ascending=False)
        )

    return {
        "Model": model_name,
        "Parameter": param_name,
        "Setting": param_value,
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s),
        "ROC-AUC": np.mean(roc_aucs) if roc_aucs else None,
        "Confusion Matrix": np.sum(confusion_mats, axis=0),
        "Features": mean_features
    }