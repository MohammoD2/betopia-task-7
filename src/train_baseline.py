from pathlib import Path
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

try:
    import dagshub
except ImportError:  # pragma: no cover - optional dependency
    dagshub = None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    model_dir = Path(__file__).resolve().parents[1] / "models"
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Dagshub + MLflow if available
    if dagshub is not None:
        dagshub.init(repo_owner="MohammoD2", repo_name="betopia-task-7", mlflow=True)
    mlflow.set_experiment("lead_scoring")

    # Load processed lead-only data
    X_train = pd.read_csv(data_dir / "X_train_lead.csv")
    X_test = pd.read_csv(data_dir / "X_test_lead.csv")
    y_train = pd.read_csv(data_dir / "y_train_lead.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test_lead.csv").squeeze("columns")

    # Hyperparameter search to get a stronger baseline
    base_model = GradientBoostingClassifier(random_state=42)
    param_distributions = {
        "n_estimators": [150, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    with mlflow.start_run(run_name="baseline_lead_only"):
        mlflow.set_tag("pipeline_stage", "baseline")
        mlflow.set_tag("data_version", "lead_only")

        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob),
        }

        model_path = model_dir / "baseline_gradient_boosting.pkl"
        joblib.dump(model, model_path)

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path, artifact_path="model")
        pd.DataFrame([metrics]).to_csv(
            reports_dir / "baseline_metrics.csv", index=False
        )

        print("Best params:", search.best_params_)
        print("Improved baseline metrics:", metrics)


if __name__ == "__main__":
    main()
