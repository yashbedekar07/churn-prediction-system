"""Data cleaning, feature engineering, and encoding for the churn pipeline.

Every function accepts either a full training dataframe (many rows) or a
single-row inference dataframe and must behave consistently in both cases,
since api/main.py reuses this module at prediction time.
"""

from __future__ import annotations

import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BUCKET_BINS = [-1, 12, 24, 48, 60, 1000]
TENURE_BUCKET_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, drop identifiers, and normalize the target column.

    - ``TotalCharges`` arrives as a string in the raw CSV (some rows contain
      blank strings for brand-new customers); coerce to numeric and fill
      missing values with 0.
    - ``customerID`` is a unique identifier with no predictive value.
    - ``Churn`` (Yes/No) is mapped to a binary integer target when present.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0.0)

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: tenure bucket and total active service count."""
    df = df.copy()

    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=TENURE_BUCKET_BINS, labels=TENURE_BUCKET_LABELS
        ).astype(str)

    present_service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
    if present_service_cols:
        df["num_services"] = (df[present_service_cols] == "Yes").sum(axis=1)

    return df


def encode_features(
    df: pd.DataFrame, reference_columns: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode categorical columns.

    When ``reference_columns`` is provided (inference time), the result is
    reindexed to exactly match the training-time feature set: columns absent
    from ``df`` are added as zeros, and any columns not seen during training
    are dropped. This keeps single-row inference frames aligned with the
    frame XGBoost was trained on.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)

    return encoded
