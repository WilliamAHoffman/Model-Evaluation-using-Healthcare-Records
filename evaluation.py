import numpy as np
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

def evaluate_model(
    model,
    X,
    y,
    model_name,
    param_name=None,
    param_value=None,
    n_splits=5,
    random_state=0
    ):

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    confusion_mats = []

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

    return {
        "Model": model_name,
        "Parameter": param_name,
        "Setting": param_value,
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s),
        "ROC-AUC": np.mean(roc_aucs) if roc_aucs else None,
        "Confusion Matrix": np.sum(confusion_mats, axis=0)
    }