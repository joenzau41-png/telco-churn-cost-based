from pathlib import Path

import pandas as pd

# Project Root 
RAW_PATH = Path(__file__).resolve().parents[1] /"data"/"raw"/"telco_churn.csv"

def load_data(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw Telco churn CSV"""
    return pd.read_csv(path)

def clean_data (df : pd.DataFrame) -> pd.DataFrame:
    """Fix known data problems and return a new DataFrame"""
    df = df.copy()
    #TotalCharges is blank for brand new customers (tenure = 0): treat as 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    # # Target: 1 = churned, 0 = stayed
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df