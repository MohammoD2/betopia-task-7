from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_LEADS = DATA_DIR / "raw" / "leads.csv"
CRM_OUTCOMES = DATA_DIR / "crm_feedback" / "outcomes.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "converted"
LEAD_TARGET_COL = f"{TARGET_COL}_lead"
ID_COL = "lead_id"


def ensure_identifier(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ensure the dataframe has a lead identifier for joining."""
    if ID_COL in df.columns:
        return df
    df = df.copy()
    df.insert(0, ID_COL, range(1, len(df) + 1))
    print(f"[merge_crm] '{name}' missing '{ID_COL}', generated sequential IDs.")
    return df


def main() -> None:
    leads = pd.read_csv(RAW_LEADS)
    outcomes = pd.read_csv(CRM_OUTCOMES)

    if TARGET_COL not in leads.columns:
        raise ValueError(f"'{TARGET_COL}' column missing in {RAW_LEADS}")
    if TARGET_COL not in outcomes.columns:
        raise ValueError(f"'{TARGET_COL}' column missing in {CRM_OUTCOMES}")

    leads = ensure_identifier(leads, "leads").rename(
        columns={TARGET_COL: LEAD_TARGET_COL}
    )
    outcomes = ensure_identifier(outcomes, "outcomes")

    merged = leads.merge(
        outcomes[[ID_COL, TARGET_COL]],
        on=ID_COL,
        how="inner",
        suffixes=("", "_crm"),
    )
    if merged.empty:
        raise ValueError("Merge produced zero rows; check identifiers.")

    feature_cols = [col for col in leads.columns if col not in {LEAD_TARGET_COL, ID_COL}]
    X = merged[feature_cols]
    y = merged[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(PROCESSED_DIR / "X_train_merged.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test_merged.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train_merged.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test_merged.csv", index=False)
    print(
        "[merge_crm] Saved merged train/test splits to data/processed with "
        f"{len(X_train)} train rows and {len(X_test)} test rows."
    )


if __name__ == "__main__":
    main()
