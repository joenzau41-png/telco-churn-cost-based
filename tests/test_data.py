import pandas as pd

from src.data import clean_data, split_data


def make_raw(n=100):
    """A tiny fake version of the raw Telco table (no real data file needed)."""
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "tenure": [0 if i == 0 else 12 for i in range(n)],
        "TotalCharges": [" " if i == 0 else "100.5" for i in range(n)],
        "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
    })


def test_blank_total_charges_become_zero():
    cleaned = clean_data(make_raw())
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned.loc[0, "TotalCharges"] == 0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_churn_is_encoded_as_one_and_zero():
    cleaned = clean_data(make_raw())
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert cleaned.loc[0, "Churn"] == 1      # row 0 was "Yes"


def test_clean_data_does_not_change_its_input():
    raw = make_raw()
    before = raw.copy()
    clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_split_sizes_columns_and_churn_rate():
    X_train, X_test, y_train, y_test = split_data(clean_data(make_raw()))
    assert len(X_train) == 80 and len(X_test) == 20
    assert "customerID" not in X_train.columns      # ID must not be a feature
    assert "Churn" not in X_train.columns           # target must not leak into features
    assert abs(y_train.mean() - y_test.mean()) < 0.02   # stratified split
