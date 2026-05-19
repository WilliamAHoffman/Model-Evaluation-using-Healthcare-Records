from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

diabetes_csv = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv');

#we convert to ints and the decimal points are not used
diabetes_csv = diabetes_csv.astype(int)
 
#for this reason we combine pre-diabetes and type 2 diabetes into a single diabetes class
#pre-diabetes was the least common class and so combining the two classes makes the tree more accurate  
diabetes_csv['Diabetes_012'] = diabetes_csv['Diabetes_012'].replace(2, 1)

x = diabetes_csv.drop(columns='Diabetes_012')
y = diabetes_csv['Diabetes_012']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y,
    random_state=0
)


def evaluate_model(model, x, y, cv, model_name, param_name=None, param_value=None):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    confusion_mats = []

    for train_idx, test_idx in cv.split(x, y):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
        else:
            y_prob = None

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        confusion_mats.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        if y_prob is not None:
            roc_aucs.append(roc_auc_score(y_test, y_prob))

    return {
        "Model": model_name,
        "Setting": param_value,
        "Accuracy": sum(accuracies) / len(accuracies),
        "Precision": sum(precisions) / len(precisions),
        "Recall": sum(recalls) / len(recalls),
        "F1": sum(f1s) / len(f1s),
        "ROC-AUC": sum(roc_aucs) / len(roc_aucs) if roc_aucs else None,
        "Confusion Matrix": sum(confusion_mats)
    }


