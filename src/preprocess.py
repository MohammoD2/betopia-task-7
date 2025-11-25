import pandas as pd

# Load raw data
leads = pd.read_csv("data/raw/lead.csv")
outcomes = pd.read_csv("data/raw/outcomes.csv")

# Merge on lead_id (replace with your actual key)
data = pd.merge(leads, outcomes, on='lead_id', how='left')

# Features and target
target = 'converted'
X = data.drop(columns=[target])
y = data[target]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
