import shutil
from pathlib import Path

import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

model_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=10000, random_state=42),
        "param_grid": {"C": [0.1, 0.3, 0.8, 2.0]}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [400, 800],
            "max_depth": [12, 18, None],
            "min_samples_leaf": [1, 2],
        }
    },
}


def load_data():
    X_train = pd.read_csv("data/X_train_fe.csv")
    X_test = pd.read_csv("data/X_test_fe.csv")
    y_train = pd.read_csv("data/y_train_fe.csv").squeeze()
    y_test = pd.read_csv("data/y_test_fe.csv").squeeze()

    return X_train, X_test, y_train, y_test


def build_signature(X_train, y_train):
    # локальный fit для сигнатуры (как было)
    from sklearn.linear_model import LogisticRegression

    small_X = X_train.iloc[:100]
    small_y = y_train.iloc[:100]

    res = LogisticRegression().fit(small_X, small_y).predict(small_X)
    signature = infer_signature(small_X, res)

    return signature


def train():
    shutil.rmtree("./mlflow", ignore_errors=True)
    Path("./mlflow").mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    signature = build_signature(X_train, y_train)

    experiment_name = "Student_Dropout"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location="./mlflow/mlruns"
        )

    mlflow.set_experiment(experiment_name)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_score = -float("inf")
    best_run_id = None
    best_model_name = None

    for model_name, cfg in model_grid.items():
        print(f"\n=== {model_name} ===")

        with mlflow.start_run(run_name=model_name):
            grid = GridSearchCV(
                estimator=cfg["model"],
                param_grid=cfg["param_grid"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )

            grid.fit(X_train, y_train, sample_weight=sample_weights)
            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "test_f1": f1_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_roc_auc": roc_auc_score(y_test, y_proba),
                "test_prauc": average_precision_score(y_test, y_proba),
            }

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                best_model,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:5].copy(),
            )

            disp = ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=["Остаётся", "Отчислен"],
                cmap="Blues",
            )

            mlflow.log_figure(
                figure=disp.figure_,
                artifact_file=f"confusion_matrix_{model_name}.png",
            )

            print(f"Test ROC AUC: {metrics['test_roc_auc']:.4f}")
            print(f"Best CV ROC AUC: {grid.best_score_:.4f}")

            if metrics["test_roc_auc"] > best_score:
                best_score = metrics["test_roc_auc"]
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = model_name

    print("\n=== Downloading Best Model ===")
    print(f"Best model: '{best_model_name}' (ROC AUC = {best_score:.4f})")

    client = MlflowClient()
    client.download_artifacts(
        run_id=best_run_id,
        path="model",
        dst_path="./mlflow/best_model",
    )


if __name__ == "__main__":
    train()
