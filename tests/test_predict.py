import pytest

from src.predict import MODELS_DIR, predict_churn

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "churn_pipeline.joblib").exists(),
    reason="Model not found: run `python -m src.train` first",
)


def test_result_has_expected_fields(risky_customer):
    result = predict_churn(risky_customer)
    assert set(result) == {
        "churn_probability", "contact_customer",
        "cutoff_used", "expected_net_gain_if_contacted",
    }
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_new_month_to_month_customer_is_riskier_than_loyal_one(risky_customer, loyal_customer):
    risky = predict_churn(risky_customer)
    loyal = predict_churn(loyal_customer)
    assert risky["churn_probability"] > loyal["churn_probability"]


def test_recommendations_make_business_sense(risky_customer, loyal_customer):
    assert predict_churn(risky_customer)["contact_customer"] is True
    assert predict_churn(loyal_customer)["contact_customer"] is False
