"""API tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    """TestClient as a context manager so FastAPI's lifespan (init_db) runs."""
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(client: TestClient, sample_customer: dict) -> None:
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["top_drivers"]) == 5


def test_predict_missing_required_fields_returns_400(client: TestClient) -> None:
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_wrong_type_returns_400(client: TestClient, sample_customer: dict) -> None:
    bad_payload = dict(sample_customer)
    bad_payload["tenure"] = "not-a-number"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400


def test_predict_invalid_category_returns_400(client: TestClient, sample_customer: dict) -> None:
    bad_payload = dict(sample_customer)
    bad_payload["Contract"] = "Lifetime"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400


def test_predict_negative_charges_returns_400(client: TestClient, sample_customer: dict) -> None:
    bad_payload = dict(sample_customer)
    bad_payload["MonthlyCharges"] = -10.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400


def test_history_reflects_logged_prediction(client: TestClient, sample_customer: dict) -> None:
    client.post("/predict", json=sample_customer)
    response = client.get("/predictions/history", params={"limit": 5})
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert "churn_probability" in body[0]
    assert "risk_level" in body[0]


def test_history_rejects_invalid_limit(client: TestClient) -> None:
    response = client.get("/predictions/history", params={"limit": 0})
    assert response.status_code == 400
