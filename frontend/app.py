"""Streamlit frontend for the churn prediction API."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 10

RISK_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


def call_predict(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST a customer record to the API and return the parsed response, or None on failure."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error("The prediction request timed out. Please try again.")
        return None

    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(f"API returned an error ({response.status_code}): {detail}")
        return None

    return response.json()


def call_history(limit: int = 20) -> list[dict[str, Any]]:
    """GET recent predictions from the API. Returns an empty list on any failure."""
    try:
        response = requests.get(
            f"{API_URL}/predictions/history", params={"limit": limit}, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []


def render_customer_form() -> dict[str, Any] | None:
    """Render the input form and return the submitted payload, or None if not submitted."""
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

        submitted = st.form_submit_button("Predict Churn Risk")

    if not submitted:
        return None

    return {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }


def render_prediction_result(result: dict[str, Any]) -> None:
    """Display churn probability, risk level, and a SHAP driver bar chart."""
    probability = result["churn_probability"]
    risk_level = result["risk_level"]
    top_drivers = result["top_drivers"]

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.markdown(
        f"### Risk Level: :{RISK_COLORS.get(risk_level, 'gray')}[{risk_level}]"
    )

    st.subheader("Top Drivers (SHAP)")
    drivers_df = pd.DataFrame(
        {"feature": list(top_drivers.keys()), "contribution": list(top_drivers.values())}
    ).sort_values("contribution")
    st.bar_chart(drivers_df.set_index("feature"))


def render_history() -> None:
    """Display a table of recent predictions."""
    st.subheader("Recent Predictions")
    history = call_history(limit=20)
    if not history:
        st.info("No prediction history yet, or the API is unreachable.")
        return
    history_df = pd.DataFrame(history)[["id", "created_at", "churn_probability", "risk_level"]]
    st.dataframe(history_df, use_container_width=True)


def main() -> None:
    """Render the Streamlit page."""
    st.title("📉 Customer Churn Predictor")
    st.caption(f"Connected to API: {API_URL}")

    payload = render_customer_form()
    if payload is not None:
        result = call_predict(payload)
        if result is not None:
            render_prediction_result(result)

    st.divider()
    render_history()


if __name__ == "__main__":
    main()
