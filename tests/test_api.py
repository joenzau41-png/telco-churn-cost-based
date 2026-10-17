import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.predict import MODELS_DIR

client = TestClient(app)

needs_model = pytest.mark.skipif(
    not (MODELS_DIR / "churn_pipeline.joblib").exists(),
    reason="Model not found: run `python -m src.train` first",
)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@needs_model
def test_predict_returns_a_decision(risky_customer):
    response = client.post("/predict", json=risky_customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["contact_customer"], bool)


def test_predict_rejects_an_unknown_contract(risky_customer):
    risky_customer["Contract"] = "Weekly"
    response = client.post("/predict", json=risky_customer)
    assert response.status_code == 422
