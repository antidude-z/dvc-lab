import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


def train():
    X_train = pd.read_csv("data/X_train_fe.csv")
    X_test = pd.read_csv("data/X_test_fe.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=10000, random_state=42),
        param_grid={"C": [0.1, 0.3, 0.8, 2.0]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train, sample_weight=sample_weights)

    best_model = grid.best_estimator_
    print(f"Best CV ROC AUC: {grid.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Test Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Test F1: {f1_score(y_test, y_pred):.4f}")

    joblib.dump(best_model, "data/model.pkl")


if __name__ == "__main__":
    train()
