"""FastAPI application exposing the churn prediction model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.database import get_db, init_db
from api.models import PredictionLog
from api.schemas import CustomerInput, HealthResponse, PredictionHistoryItem, PredictionResponse
from ml.explain import explain_prediction
from ml.predict import predict_churn

logger = logging.getLogger(__name__)

LOW_RISK_THRESHOLD = 0.4
HIGH_RISK_THRESHOLD = 0.7
TOP_N_DRIVERS = 5


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the prediction_logs table exists before serving traffic."""
    init_db()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serves an XGBoost churn model with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 400 with a readable message instead of the default 422."""
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request payload", "errors": exc.errors()},
    )


def compute_risk_level(probability: float) -> str:
    """Bucket a churn probability into Low / Medium / High risk."""
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability <= HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, db: Session = Depends(get_db)) -> PredictionResponse:
    """Score a customer, explain the prediction, and log it to the database."""
    customer_dict = customer.model_dump()

    try:
        result = predict_churn(customer_dict)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run `python -m ml.train` before serving predictions.",
        ) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not process customer input: {exc}") from exc

    probability = result["churn_probability"]
    risk_level = compute_risk_level(probability)
    top_drivers = explain_prediction(result["encoded_row"], top_n=TOP_N_DRIVERS)

    log_entry = PredictionLog(
        churn_probability=probability,
        risk_level=risk_level,
        customer_data=customer_dict,
    )
    db.add(log_entry)
    db.commit()

    return PredictionResponse(
        churn_probability=probability, risk_level=risk_level, top_drivers=top_drivers
    )


@app.get("/predictions/history", response_model=list[PredictionHistoryItem])
def prediction_history(limit: int = 20, db: Session = Depends(get_db)) -> list[PredictionLog]:
    """Return the most recent predictions, newest first."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return (
        db.query(PredictionLog)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
        .all()
    )
