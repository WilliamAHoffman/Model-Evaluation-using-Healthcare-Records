from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 0

def create_tree(depth):
    return DecisionTreeClassifier(
        criterion='gini',
        max_depth=depth,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

def create_logistic_regression(iterations):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=iterations,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])