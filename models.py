from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 0


def create_tree(depth):
    return DecisionTreeClassifier(
        criterion='gini',
        max_depth=depth,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )


def create_knn(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            n_jobs=-1
        ))
    ])


def create_logistic_regression(iterations):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=iterations,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])