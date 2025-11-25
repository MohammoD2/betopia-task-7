from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

KS_ALPHA = 0.05
PSI_ALERT_THRESHOLD = 0.1
PREDICTION_PSI_THRESHOLD = 0.2


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        return 0.0
    expected_perc, _ = np.histogram(expected, bins=breakpoints)
    actual_perc, _ = np.histogram(actual, bins=breakpoints)
    expected_perc = expected_perc / (len(expected) or 1)
    actual_perc = actual_perc / (len(actual) or 1)

    psi = 0.0
    for exp_p, act_p in zip(expected_perc, actual_perc):
        exp_p = max(exp_p, 1e-6)
        act_p = max(act_p, 1e-6)
        psi += (act_p - exp_p) * np.log(act_p / exp_p)
    return psi


def compute_feature_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> List[Dict[str, float]]:
    shared_cols = [col for col in baseline.columns if col in current.columns]
    summary = []
    for col in shared_cols:
        ks_stat, ks_p = ks_2samp(baseline[col], current[col])
        psi_val = population_stability_index(baseline[col], current[col])
        summary.append(
            {
                "feature": col,
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "psi": float(psi_val),
            }
        )
    return summary


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(DATA_DIR / "X_train_lead.csv")
    current = pd.read_csv(DATA_DIR / "X_train_merged.csv")

    feature_drift = compute_feature_drift(baseline, current)
    flagged_features = [
        item
        for item in feature_drift
        if item["ks_pvalue"] < KS_ALPHA or item["psi"] >= PSI_ALERT_THRESHOLD
    ]

    baseline_model_path = MODEL_DIR / "baseline_gradient_boosting.pkl"
    crm_model_path = MODEL_DIR / "crm_gradient_boosting.pkl"
    if not baseline_model_path.exists() or not crm_model_path.exists():
        raise FileNotFoundError(
            "Baseline or CRM model artifact missing. Run training scripts first."
        )

    baseline_model = joblib.load(baseline_model_path)
    crm_model = joblib.load(crm_model_path)

    baseline_preds = baseline_model.predict_proba(baseline)[:, 1]
    crm_preds = crm_model.predict_proba(current)[:, 1]
    prediction_drift = population_stability_index(baseline_preds, crm_preds)
    prediction_alert = bool(prediction_drift >= PREDICTION_PSI_THRESHOLD)

    report = {
        "ks_alpha": KS_ALPHA,
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "prediction_psi_threshold": PREDICTION_PSI_THRESHOLD,
        "feature_drift": feature_drift,
        "flagged_features": [item["feature"] for item in flagged_features],
        "prediction_drift": prediction_drift,
        "prediction_drift_alert": prediction_alert,
    }
    report_path = REPORTS_DIR / "drift_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("Feature drift summary saved to", report_path)
    print("Flagged features:", report["flagged_features"])
    status = "ALERT" if prediction_alert else "OK"
    print(f"Prediction drift PSI: {prediction_drift:.4f} -> {status}")


if __name__ == "__main__":
    main()
