import pytest

RISKY = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
}


@pytest.fixture
def risky_customer():
    """A new, month-to-month, fibre customer: should be high risk."""
    return dict(RISKY)


@pytest.fixture
def loyal_customer():
    """A long-standing two-year DSL customer: should be low risk."""
    return {
        **RISKY,
        "tenure": 60,
        "Contract": "Two year",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    }
