"""Unit tests for ml/preprocess.py."""

from __future__ import annotations

import pandas as pd

from ml.preprocess import clean_data, encode_features, engineer_features


def test_clean_data_converts_total_charges_and_fills_blanks() -> None:
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "108.15"], "Churn": ["No", "Yes", "No"]})
    result = clean_data(df)
    assert result["TotalCharges"].tolist() == [29.85, 0.0, 108.15]


def test_clean_data_drops_customer_id() -> None:
    df = pd.DataFrame({"customerID": ["1234-ABCD"], "tenure": [5]})
    result = clean_data(df)
    assert "customerID" not in result.columns


def test_clean_data_maps_churn_to_binary() -> None:
    df = pd.DataFrame({"Churn": ["Yes", "No", "Yes"]})
    result = clean_data(df)
    assert result["Churn"].tolist() == [1, 0, 1]
    assert result["Churn"].dtype.kind in "iu"


def test_clean_data_handles_single_row_without_target_or_id() -> None:
    """Inference-time rows have no customerID or Churn column."""
    df = pd.DataFrame({"tenure": [12], "MonthlyCharges": [70.0], "TotalCharges": [840.0]})
    result = clean_data(df)
    assert "Churn" not in result.columns
    assert result["tenure"].iloc[0] == 12


def test_engineer_features_tenure_bucket_boundaries() -> None:
    df = pd.DataFrame({"tenure": [0, 12, 13, 70]})
    result = engineer_features(df)
    assert result["tenure_bucket"].tolist() == ["0-12", "0-12", "13-24", "61+"]


def test_engineer_features_num_services_counts_only_yes() -> None:
    df = pd.DataFrame(
        {
            "PhoneService": ["Yes"],
            "MultipleLines": ["No phone service"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No internet service"],
            "TechSupport": ["Yes"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
        }
    )
    result = engineer_features(df)
    assert result["num_services"].iloc[0] == 3


def test_engineer_features_missing_service_columns_does_not_crash() -> None:
    df = pd.DataFrame({"tenure": [10]})
    result = engineer_features(df)
    assert "num_services" not in result.columns
    assert "tenure_bucket" in result.columns


def test_encode_features_reference_alignment_adds_missing_and_drops_extra() -> None:
    df = pd.DataFrame({"Contract": ["Month-to-month"], "gender": ["Female"]})
    reference_columns = ["Contract_Month-to-month", "Contract_Two year", "gender_Male"]
    result = encode_features(df, reference_columns=reference_columns)

    assert list(result.columns) == reference_columns
    assert result["Contract_Month-to-month"].iloc[0] == 1
    assert result["Contract_Two year"].iloc[0] == 0
    assert result["gender_Male"].iloc[0] == 0


def test_encode_features_unseen_category_does_not_crash() -> None:
    """A category value never seen at training time should not raise or leak a stray column."""
    df = pd.DataFrame({"Contract": ["Some New Plan"]})
    reference_columns = ["Contract_Month-to-month", "Contract_Two year"]
    result = encode_features(df, reference_columns=reference_columns)

    assert list(result.columns) == reference_columns
    assert result.iloc[0].tolist() == [0, 0]


def test_encode_features_without_reference_columns_returns_dummies() -> None:
    df = pd.DataFrame({"Contract": ["Month-to-month", "Two year"]})
    result = encode_features(df)
    assert "Contract_Month-to-month" in result.columns
    assert "Contract_Two year" in result.columns
