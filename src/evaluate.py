from pathlib import Path

import mlflow
import pandas as pd

try:
    import dagshub
except ImportError:  # pragma: no cover - optional dependency
    dagshub = None

EXPERIMENT_NAME = "lead_scoring"
STAGE_LABELS = {
    "baseline": "Lead Only",
    "crm_feedback": "Lead + CRM",
}
METRICS = ["accuracy", "f1", "precision", "recall", "auc"]


def fetch_latest_stage_runs() -> pd.DataFrame:
    if dagshub is not None:
        dagshub.init(
            repo_owner="MohammoD2", repo_name="betopia-task-7", mlflow=True
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["attribute.start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError("No MLflow runs found for baseline or CRM stages.")

    records = []
    for stage, label in STAGE_LABELS.items():
        tag_series = runs.get("tags.pipeline_stage")
        if tag_series is None:
            raise RuntimeError(
                "Runs missing 'pipeline_stage' tag. Re-run training scripts."
            )
        stage_runs = runs[tag_series == stage]
        if stage_runs.empty:
            continue
        latest = stage_runs.iloc[0]
        record = {"model": label}
        for metric in METRICS:
            record[metric] = latest.get(f"metrics.{metric}")
        record["mlflow_run_id"] = latest["run_id"]
        records.append(record)
    if not records:
        raise RuntimeError("No runs available for comparison.")
    return pd.DataFrame(records)


def main() -> None:
    comparison = fetch_latest_stage_runs()
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "model_comparison.csv"

    comparison.to_csv(report_path, index=False)
    print("Latest model comparison:")
    print(comparison.to_markdown(index=False))
    print(f"Saved comparison table to {report_path}")


if __name__ == "__main__":
    main()
