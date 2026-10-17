from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project root
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def load_data(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data problems and return a new DataFrame."""
    df = df.copy()
    # TotalCharges is blank for brand-new customers (tenure = 0): treat as 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # Target: 1 = churned, 0 = stayed
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split into train/test features and targets, keeping the churn rate equal."""
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )