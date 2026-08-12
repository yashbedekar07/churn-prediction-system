"""Tests for ml/predict.py and ml/explain.py against the trained model artifacts."""

from __future__ import annotations

from ml.explain import explain_prediction
from ml.predict import load_feature_columns, predict_churn


def test_predict_churn_returns_probability_in_valid_range(sample_customer: dict) -> None:
    result = predict_churn(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_churn_encoded_row_matches_training_feature_columns(sample_customer: dict) -> None:
    result = predict_churn(sample_customer)
    assert list(result["encoded_row"].columns) == load_feature_columns()
    assert len(result["encoded_row"]) == 1


def test_explain_prediction_returns_requested_top_n(sample_customer: dict) -> None:
    result = predict_churn(sample_customer)
    top_drivers = explain_prediction(result["encoded_row"], top_n=5)
    assert len(top_drivers) == 5
    assert all(isinstance(v, float) for v in top_drivers.values())
    assert set(top_drivers.keys()).issubset(set(load_feature_columns()))


def test_explain_prediction_sorted_by_absolute_contribution(sample_customer: dict) -> None:
    result = predict_churn(sample_customer)
    top_drivers = explain_prediction(result["encoded_row"], top_n=5)
    magnitudes = [abs(v) for v in top_drivers.values()]
    assert magnitudes == sorted(magnitudes, reverse=True)
