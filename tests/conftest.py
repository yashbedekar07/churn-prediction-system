"""Shared pytest fixtures: an isolated SQLite database for API tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest

_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "churn_test_predictions.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_PATH}"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db() -> Iterator[None]:
    """Remove the temp test database file once the whole test session finishes."""
    yield
    from api.database import engine

    engine.dispose()
    if os.path.exists(_TEST_DB_PATH):
        try:
            os.remove(_TEST_DB_PATH)
        except PermissionError:
            pass


@pytest.fixture
def sample_customer() -> dict:
    """A realistic high-churn-risk customer payload matching CustomerInput."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }
