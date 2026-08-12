"""Inference-time helpers: load the trained model and score a single customer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from ml.preprocess import clean_data, encode_features, engineer_features
from ml.train import FEATURES_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> xgb.XGBClassifier:
    """Load the trained XGBoost model from disk (cached after first call)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m ml.train` first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_feature_columns(path: Path = FEATURES_PATH) -> list[str]:
    """Load the ordered training-time feature column list (cached after first call)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Feature list not found at {path}. Run `python -m ml.train` first."
        )
    return joblib.load(path)


def preprocess_single(customer: dict[str, Any]) -> pd.DataFrame:
    """Run a single customer record through the same pipeline used in training."""
    df = pd.DataFrame([customer])
    df = clean_data(df)
    df = engineer_features(df)
    reference_columns = load_feature_columns()
    return encode_features(df, reference_columns=reference_columns)


def predict_churn(customer: dict[str, Any]) -> dict[str, Any]:
    """Predict churn probability for a single customer.

    Returns a dict with the churn probability (0-1) and the encoded feature
    row, so callers (e.g. the API layer) can reuse the row for SHAP
    explanations without recomputing preprocessing.
    """
    model = load_model()
    encoded_row = preprocess_single(customer)
    probability = float(model.predict_proba(encoded_row)[0, 1])
    return {"churn_probability": probability, "encoded_row": encoded_row}
