import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation import COST_FP, GAIN_TP

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the saved model and its settings once, then reuse them."""
    model = joblib.load(MODELS_DIR / "churn_pipeline.joblib")
    config = json.loads((MODELS_DIR / "model_config.json").read_text())
    return model, config


def predict_churn(customer: dict) -> dict:
    """Score one customer (a dict of the 19 input fields) and recommend an action."""
    model, config = load_artifacts()
    probability = float(model.predict_proba(pd.DataFrame([customer]))[0, 1])
    expected_gain = probability * GAIN_TP - (1 - probability) * COST_FP
    return {
        "churn_probability": round(probability, 3),
        "contact_customer": probability >= config["threshold"],
        "cutoff_used": config["threshold"],
        "expected_net_gain_if_contacted": round(expected_gain, 2),
    }