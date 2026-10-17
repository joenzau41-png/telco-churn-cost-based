import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data import clean_data, load_data, split_data
from src.evaluation import (CUSTOMER_VALUE, OFFER_COST, SUCCESS_RATE,
                            net_savings, savings_curve)
from src.features import build_preprocessor

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def train():
    """Train the churn model, choose the cost-based cut-off and save both."""
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    # Choose the cut-off from honest out-of-fold predictions on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    curve = savings_curve(y_train, oof)
    threshold = float(curve.loc[curve["net_savings"].idxmax(), "threshold"])

    model.fit(X_train, y_train)
    test_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    test_savings = float(net_savings(y_test, test_pred))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "churn_pipeline.joblib")
    config = {
        "threshold": threshold,
        "offer_cost": OFFER_COST,
        "customer_value": CUSTOMER_VALUE,
        "success_rate": SUCCESS_RATE,
        "test_net_savings": test_savings,
    }
    (MODELS_DIR / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved model. Cut-off = {threshold:.2f}, test net savings = ${test_savings:,.0f}")


if __name__ == "__main__":
    train()