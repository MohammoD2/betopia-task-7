from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

lead_path = RAW_DIR / "leads.csv"
leads = pd.read_csv(lead_path)

TARGET_COL = "converted"
if TARGET_COL not in leads.columns:
    raise ValueError(f"Expected '{TARGET_COL}' column in {lead_path}")

X = leads.drop(columns=[TARGET_COL])
y = leads[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed datasets
X_train.to_csv(PROCESSED_DIR / "X_train_lead.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test_lead.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train_lead.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test_lead.csv", index=False)
