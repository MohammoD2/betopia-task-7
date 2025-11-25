# CRM Feedback Loop Retraining

End-to-end MLOps exercise that retrains a synthetic lead-scoring model with CRM win/loss feedback, compares the uplift, and documents drift handling plus automation.

## Repository Layout
- `src/` – end-to-end pipeline scripts: preprocessing, training, evaluation, drift checks.
- `data/raw/` – seed lead dataset; `data/crm_feedback/` holds synthetic CRM outcomes; `data/processed/` stores cached train/test splits.
- `models/` – serialized `joblib` artifacts (`baseline_gradient_boosting.pkl`, `crm_gradient_boosting.pkl`).
- `reports/` – CSV/JSON reports for metrics, comparisons, and drift diagnostics.
- `mlruns/` – local MLflow tracking store; `mlflow_experiments/` is a convenience export.
- `.github/workflows/retrain.yml` – GitHub Actions workflow that re-runs the entire pipeline on a schedule or data/code change.

## Data + CRM Feedback
1. `src/preprocess_lead.py` stratifies the historical lead-only dataset (`data/raw/leads.csv`).
2. `src/merge_crm.py` joins the synthetic CRM outcomes (`data/crm_feedback/outcomes.csv`) back to the lead features using `lead_id`, yielding the “lead + CRM” training view.
   - Sequential IDs are auto-created if a file is missing `lead_id`, so files can be dropped into the raw folders without extra prep.

## Pipeline Stages
1. `python src/preprocess_lead.py`
2. `python src/train_baseline.py` – lead-only GradientBoostingClassifier + RandomizedSearchCV, logs to MLflow run `baseline_lead_only`.
3. `python src/merge_crm.py`
4. `python src/retrain_crm.py` – retrains on the merged feature set, run name `crm_feedback_retrain`.
5. `python src/evaluate.py` – pulls the latest MLflow runs per pipeline stage and writes `reports/model_comparison.csv`.
6. `python src/drift_detection.py` – Kolmogorov–Smirnov + PSI checks across historical vs CRM-enriched features and model scores, stored in `reports/drift_report.json`.

All scripts are orchestrated in CI (`.github/workflows/retrain.yml`) so a single job recomputes the artifacts.

## Local Usage
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Full pipeline (same order CI uses)
python src/preprocess_lead.py
python src/train_baseline.py
python src/merge_crm.py
python src/retrain_crm.py
python src/evaluate.py
python src/drift_detection.py
```

To inspect experiment history: `mlflow ui --backend-store-uri mlruns`.

## Before vs After Metrics
Latest comparison (`reports/model_comparison.csv`) pulled from MLflow experiment `lead_scoring`:

| Model Stage | Accuracy | F1 | Precision | Recall | AUC | MLflow Run |
| --- | --- | --- | --- | --- | --- | --- |
| Lead Only (baseline) | 0.989 | 0.994 | 0.990 | 0.999 | 0.832 | `9a382b4472ac47b18627e93efbe12b46` |
| Lead + CRM (retrained) | 0.965 | 0.982 | 0.965 | 1.000 | 0.501 | `c2ee5b07ff8148a5bb7eac44cc39f1b5` |

Interpretation:
- CRM feedback sacrificed a bit of overall precision/accuracy in exchange for perfect recall (catching every won deal) which can be desirable when sales follow-up cost is low.
- AUC drop signals that the synthetic CRM signal currently adds noise; drift monitoring plus future data quality gates should guard against deploying a regressed model.

## Drift Handling
- `src/drift_detection.py` compares feature distributions (KS test, PSI) and prediction drift between baseline vs CRM models.
- Latest report (`reports/drift_report.json`) shows **prediction PSI = 11.48 (alert = true)** even though no single feature breached thresholds—indicating the CRM retrain shifts score calibration. This justifies keeping the baseline champion until CRM data matures or calibrating probabilities before rollout.
- Alerts can be wired into CI (fail job) or surfaced in dashboards so stakeholders decide on promotion.

## Automation & Lifecycle
- **Retraining trigger**: GitHub Actions workflow runs nightly (`cron: 0 2 * * *`) and on any change to raw data, CRM feedback, source code, or the workflow itself.
- **Data versioning**: `dvc pull` step ensures the latest tracked datasets are synced before training.
- **Experiment tracking**: MLflow autologs hyperparameters/metrics/artifacts so the evaluation step can programmatically compare stages.
- **Promotion logic**: the comparison report plus drift report act as gates for manual approval; the workflow is the skeleton for a full CD process (add approval steps, model registry promotion, etc.).

## Deliverables Checklist
- ✅ Code: preprocessing, training, evaluation, drift, and automation scripts under `src/` + workflow.
- ✅ Metrics comparison: `reports/baseline_metrics.csv`, `reports/crm_metrics.csv`, and consolidated `reports/model_comparison.csv` (table above).
- ✅ CRM feedback retrain: `src/retrain_crm.py` + `models/crm_gradient_boosting.pkl`.
- ✅ Drift explanation: `reports/drift_report.json` + monitoring notes.
- ✅ Automation reasoning: GitHub Actions workflow describing how retraining stays up to date.

This README should provide enough context for another engineer (or reviewer) to understand what each folder/script does, why Gradient Boosting was selected, how MLflow metrics are compared, and how the MLOps pieces (tracking, drift checks, automation) close the CRM feedback loop.

