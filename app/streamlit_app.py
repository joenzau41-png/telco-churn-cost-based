import sys
from pathlib import Path

import streamlit as st

# Let this script import from src/ no matter where Streamlit is started
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.predict import load_artifacts, predict_churn  # noqa: E402

st.set_page_config(page_title="Telco Churn Decision Tool", page_icon="📞")
st.title("📞 Customer Churn Decision Tool")
st.write(
    "Enter a customer's details to see how likely they are to leave, "
    "and whether a retention offer is worth sending."
)

# ---- Sidebar: the business assumptions behind the recommendation ----
_, config = load_artifacts()
with st.sidebar:
    st.header("Business assumptions")
    st.write(f"Retention offer cost: **${config['offer_cost']}**")
    st.write(f"Value of a saved customer: **${config['customer_value']}**")
    st.write(f"Offer success rate: **{config['success_rate']:.0%}**")
    st.write(f"Decision cut-off: **{config['threshold']:.0%}** churn probability")
    st.caption("These are assumptions, not measured values. Change them in src/evaluation.py.")

# ---- Main inputs ----
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Months as a customer", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
with col2:
    monthly = st.number_input("Monthly charges ($)", 0.0, 150.0, 70.0, step=1.0)
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

with st.expander("More details (services and household)"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"

    addon_labels = {
        "OnlineSecurity": "Online security",
        "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection",
        "TechSupport": "Tech support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies",
    }
    addons = {}
    for field, label in addon_labels.items():
        if internet == "No":
            addons[field] = "No internet service"
        else:
            addons[field] = st.selectbox(label, ["No", "Yes"])

# Approximation: total paid so far = months x monthly charge
total = float(tenure * monthly)
st.caption(f"Total charged so far is estimated as months x monthly charges = ${total:,.2f}")

customer = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

if st.button("Assess customer", type="primary"):
    result = predict_churn(customer)
    st.metric("Churn probability", f"{result['churn_probability']:.0%}")

    if result["contact_customer"]:
        st.success("✅ Recommended: send a retention offer.")
    elif result["expected_net_gain_if_contacted"] > 0:
        st.warning("⚠️ Borderline: a small expected gain, but below the cut-off. Use judgement.")
    else:
        st.info("No offer recommended: it is expected to cost more than it saves.")

    st.caption(
        f"Expected net gain if contacted: ${result['expected_net_gain_if_contacted']:,.2f}"
    )