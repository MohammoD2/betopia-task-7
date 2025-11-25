# Lead Scoring CRM Feedback Loop (MLOps Showcase)

> Interview-ready walkthrough of a synthetic CRM feedback loop: inbound CRM outcomes retrain a Gradient Boosting lead scorer, MLflow preserves experiment lineage, drift guardrails decide promotion, and GitHub Actions keeps the loop humming.

## Why This Project Stands Out
- **Clear storytelling** – Baseline vs CRM-retrained model with hard numbers, trade-offs, and next steps.
- **Production-minded** – Data versioning, experiment tracking, CI automation, and drift gates mirror a real rollout.
- **Reviewer friendly** – One-click reproduction path, MLflow run IDs, and reports already checked in.

## TL;DR for Interviewers
- Task: *“Retrain a model using CRM feedback, show before/after metrics, explain drift + automation.”*
- Status: ✅ Complete. Code, metrics, automation, and lifecycle reasoning live here.
- Decision: CRM retrain achieved perfect recall but tanked AUC and triggered a huge prediction PSI, so the baseline remains champion until CRM quality improves.

---

## Repository Map
- `src/` – preprocessing, baseline training, CRM retraining, evaluation, drift detection.
- `data/raw/` – source leads; `data/crm_feedback/` – synthetic won/lost outcomes; `data/processed/` – cached splits.
- `models/` – serialized champions (`baseline_gradient_boosting.pkl`, `crm_gradient_boosting.pkl`).
- `reports/` – CSV/JSON artifacts (`baseline_metrics.csv`, `crm_metrics.csv`, `model_comparison.csv`, `drift_report.json`).
- `mlruns/` – MLflow tracking store (UI-ready).
- `.github/workflows/retrain.yml` – nightly + change-triggered CI orchestrating the full pipeline with DVC pulls.

## Data + Feedback Loop
1. `src/preprocess_lead.py` stratifies the historical lead dataset (`data/raw/leads.csv`).
2. `src/merge_crm.py` attaches CRM outcomes (`data/crm_feedback/outcomes.csv`) via `lead_id`.
   - Missing IDs? The script auto-injects sequential identifiers so analysts can drop CSVs with zero prep.
3. Outputs land in `data/processed/` to keep all downstream steps reproducible.

## Orchestrated Pipeline
| Stage | Script | Key Output |
| --- | --- | --- |
| 1 | `preprocess_lead.py` | `X_train_lead.csv`, `y_train_lead.csv` |
| 2 | `train_baseline.py` | Lead-only Gradient Boosting + `reports/baseline_metrics.csv` (MLflow run `baseline_lead_only`) |
| 3 | `merge_crm.py` | `X_train_merged.csv`, `y_train_merged.csv` |
| 4 | `retrain_crm.py` | CRM-enriched Gradient Boosting + `reports/crm_metrics.csv` (MLflow run `crm_feedback_retrain`) |
| 5 | `evaluate.py` | Latest MLflow comparison → `reports/model_comparison.csv` |
| 6 | `drift_detection.py` | KS + PSI diagnostics → `reports/drift_report.json` |

CI executes the exact order above on every push touching data or code and every night at `cron: 0 2 * * *`.

## Reproduce Locally
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python src/preprocess_lead.py
python src/train_baseline.py
python src/merge_crm.py
python src/retrain_crm.py
python src/evaluate.py
python src/drift_detection.py
```
Inspect history anytime with `mlflow ui --backend-store-uri mlruns`.

## MLflow Metric Comparison
Latest snapshot from `reports/model_comparison.csv`:

| Model Stage | Accuracy | F1 | Precision | Recall | AUC | MLflow Run |
| --- | --- | --- | --- | --- | --- | --- |
| Lead Only (baseline) | **0.989** | **0.994** | **0.990** | 0.999 | **0.832** | `9a382b4472ac47b18627e93efbe12b46` |
| Lead + CRM (retrained) | 0.965 | 0.982 | 0.965 | **1.000** | 0.501 | `c2ee5b07ff8148a5bb7eac44cc39f1b5` |

**Takeaways**
- CRM feedback nails recall (no won deal missed) but injects noise, dragging AUC to 0.50.
- Verdict: keep baseline champion; ship CRM variant only after calibrating or improving CRM labels.

## Drift & Guardrails
- `src/drift_detection.py` runs KS + PSI on every feature plus the score distribution difference between the two models.
- Latest report (`reports/drift_report.json`) shows **prediction PSI = 11.48 (alert = true)** even though no single feature drifted—pointing to CRM label issues rather than upstream feature shifts.
- These thresholds (KS α = 0.05, PSI alerts at 0.1 / 0.2) are wired for CI so promotion can be blocked automatically.

## Automation & Lifecycle Design
- **GitHub Actions** pulls DVC data, installs deps, and executes the full pipeline headlessly.
- **MLflow** tags (`pipeline_stage`, `data_version`) guarantee we always compare the right runs.
- **Promotion flow**: comparison CSV + drift JSON become the human approval packet; hooking into a model registry is the documented next step.

## Deliverables Checklist
- ✅ Code for preprocessing, training (baseline + CRM), evaluation, drift, and CI automation.
- ✅ Metric evidence (`reports/baseline_metrics.csv`, `reports/crm_metrics.csv`, `reports/model_comparison.csv`).
- ✅ Drift explanation with PSI alert (`reports/drift_report.json`).
- ✅ Automation rationale via `.github/workflows/retrain.yml`.
- ✅ Full ML lifecycle narrative aligned with the interview brief.

---

**Pitch to interviewers:** This repo shows how I operationalize a CRM feedback loop with the same muscles I’d use in production—MLflow for lineage, DVC for data, CI for trust, and statistical guardrails for safe promotion. Drop in new CRM outcomes, push to GitHub, and the system tells us if the new model deserves the crown.

