"""Train the churn prediction model.

Pipeline: load -> clean -> engineer -> encode -> stratified train/test split
-> SMOTE (training fold only) -> Optuna hyperparameter search (recall,
5-fold CV) -> final XGBoost fit -> persist model + feature list.

Run standalone with:
    python -m ml.train
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.preprocess import clean_data, encode_features, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURES_PATH = MODEL_DIR / "churn_model_features.pkl"

RANDOM_STATE = 42
N_TRIALS = 25
N_FOLDS = 5
TEST_SIZE = 0.2


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw churn CSV from disk."""
    return pd.read_csv(path)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Run clean -> engineer -> encode and split into features/target.

    Returns the encoded feature frame, the binary target series, and the
    ordered list of feature column names (needed to align inference-time
    single-row frames against this exact training schema).
    """
    df = clean_data(df)
    df = engineer_features(df)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    X = encode_features(X)
    return X, y, X.columns.tolist()


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective: mean 5-fold cross-validated recall for a candidate config.

    SMOTE is applied inside each fold (fit on the training fold only) so
    validation folds never see oversampled/synthetic rows.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_fold, y_train_fold)

        model = xgb.XGBClassifier(**params)
        model.fit(X_res, y_res)
        preds = model.predict(X_val_fold)
        scores.append(recall_score(y_val_fold, preds))

    return float(np.mean(scores))


def train() -> None:
    """Run the full training pipeline and persist the final model artifacts."""
    logger.info("Loading data from %s", DATA_PATH)
    df = load_data()

    X, y, feature_columns = build_training_frame(df)
    logger.info("Feature matrix: %d rows, %d columns", *X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Running Optuna hyperparameter search (%d trials, optimizing recall)...", N_TRIALS)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    logger.info("Best CV recall: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    logger.info("Applying SMOTE to the full training split and fitting the final model...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    best_params = dict(study.best_params)
    best_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss"})
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION (held-out test set)")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC: {auc:.4f}")
    print("=" * 60 + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved feature list (%d columns) to %s", len(feature_columns), FEATURES_PATH)


if __name__ == "__main__":
    train()
