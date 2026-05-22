from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 0

def create_tree(max_depth):
    return DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
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

def create_random_forest(max_depth):
    return RandomForestClassifier(
        n_estimators=200,
        criterion='gini',
        max_depth=max_depth,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=5
    )