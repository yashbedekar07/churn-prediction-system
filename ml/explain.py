"""SHAP-based explainability for individual churn predictions."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import shap
import xgboost as xgb

from ml.predict import load_model


@lru_cache(maxsize=1)
def get_explainer() -> shap.TreeExplainer:
    """Build (and cache) a SHAP TreeExplainer around the trained model."""
    model: xgb.XGBClassifier = load_model()
    return shap.TreeExplainer(model)


def explain_prediction(encoded_row: pd.DataFrame, top_n: int = 5) -> dict[str, float]:
    """Return the top-N SHAP feature contributions for a single encoded row.

    ``encoded_row`` must already be aligned to the model's training-time
    feature columns (see ``ml.predict.preprocess_single``). Positive values
    push the prediction toward churn, negative values push away from churn.
    """
    explainer = get_explainer()
    shap_values = explainer.shap_values(encoded_row)

    row_values = shap_values[0]
    contributions = dict(zip(encoded_row.columns, row_values))
    top_features = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return {name: float(value) for name, value in top_features}
