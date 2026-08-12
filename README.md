# Customer Churn Prediction System

A full-stack, production-shaped app that predicts telecom customer churn and
explains *why* a customer is at risk — combining a data science pipeline
(XGBoost + SHAP) with a proper engineering stack (FastAPI, Streamlit,
PostgreSQL/SQLite, Docker, CI).

## Business problem

Telecom providers lose recurring revenue every time a customer churns, and
acquiring a replacement customer costs far more than retaining an existing
one. Retention teams need two things to act early: **a risk score per
customer**, and **the specific reasons driving that score**, so an offer or
intervention can be targeted at the actual cause (e.g. a month-to-month
contract, no tech support, a high monthly bill) rather than a blanket
discount. This system serves both as a live API and a self-serve UI so a
retention analyst can score a customer and immediately see the top drivers
behind the prediction.

## Architecture

```
                    ┌────────────────────┐
                    │   Streamlit UI      │
                    │  (frontend/app.py)  │
                    │  :8501               │
                    └──────────┬──────────┘
                               │ HTTP (API_URL)
                               ▼
                    ┌────────────────────┐        ┌──────────────────┐
                    │   FastAPI backend   │───────▶│  SQLite/Postgres  │
                    │   (api/main.py)     │        │  prediction_logs   │
                    │   :8000              │◀───────│                    │
                    └──────────┬──────────┘        └──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  ml/predict.py       │
                    │  ml/preprocess.py    │──▶ ml/explain.py (SHAP)
                    │  models/*.pkl         │
                    │  (XGBoost, trained    │
                    │   via ml/train.py)    │
                    └───────────────────────┘
```

- **frontend/** — Streamlit form that posts to the API and renders the
  probability, risk level, a SHAP driver bar chart, and prediction history.
- **api/** — FastAPI service: request validation, inference, SHAP
  explanation, and prediction logging.
- **ml/** — the model itself: preprocessing shared by training *and*
  inference, the training pipeline, and SHAP explainability.
- **data/ / models/** — the dataset and the trained model artifacts (both
  committed so the repo runs immediately without a training step).

## Data science approach and results

**Dataset**: the real [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(downloaded from a public GitHub mirror at build time — 7,043 customers, 21
columns, ~26.5% churn rate). No synthetic fallback was needed.

**Pipeline** (`ml/train.py`):
1. Clean: fix `TotalCharges` dtype, drop `customerID`, map `Churn` to binary.
2. Engineer: tenure buckets, `num_services` (count of active add-on services).
3. Encode: one-hot encoding, with the resulting column list persisted
   (`models/churn_model_features.pkl`) so inference-time single-row frames
   are aligned to the exact training schema, including graceful handling of
   categories the model never saw.
4. Stratified 80/20 train/test split.
5. **SMOTE** applied to the training fold only (never to validation or test
   data) to correct the ~3:1 class imbalance.
6. **Optuna** hyperparameter search — 25 trials, 5-fold stratified CV,
   optimizing **recall** (missing a churner is costlier than a false alarm
   in a retention setting), with SMOTE applied inside each CV fold.
7. Final XGBoost model refit on the full SMOTE-resampled training split
   with the best-found hyperparameters.

**Results on the held-out test set** (from the actual training run):

```
              precision    recall  f1-score   support

    No Churn       0.86      0.82      0.84      1035
       Churn       0.57      0.64      0.60       374

    accuracy                           0.78      1409
   macro avg       0.72      0.73      0.72      1409
weighted avg       0.78      0.78      0.78      1409

ROC-AUC: 0.8397
```

Best CV recall during the Optuna search: **0.6682**. The model is tuned to
catch churners (recall-optimized) at some cost to precision on the churn
class, which is the right trade-off for a retention use case — a missed
churner is a lost customer, a false positive is just an unnecessary offer.

**Explainability**: every prediction is accompanied by the top 5 SHAP
feature contributions (`ml/explain.py`, `shap.TreeExplainer`), so a
retention analyst sees *why* a customer is flagged, not just a number.

## Engineering components

- **FastAPI** backend (`api/`) — Pydantic-validated `/predict`,
  `/predictions/history`, `/health`; auto-generated OpenAPI docs at `/docs`;
  malformed requests return `400` with a readable message instead of a
  stack trace.
- **SQLAlchemy** persistence — `prediction_logs` table, SQLite by default,
  swaps to Postgres by changing `DATABASE_URL` only (no code changes).
- **Streamlit** frontend (`frontend/`) — customer form, SHAP bar chart,
  prediction history table, graceful error handling if the API is down.
- **pytest** suite (`tests/`) — 22 tests covering preprocessing edge cases
  (missing values, unseen categories), inference/SHAP, and the API's happy
  path plus 400 handling for missing/invalid/malformed input.
- **Docker** — a single image runs either the API or the Streamlit app
  (selected via the `command:` in `docker-compose.yml`); `docker compose up`
  runs both, frontend waiting on the API's health check.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — installs
  dependencies, runs pytest, and builds the Docker image on every push/PR
  to `main`.

## Tech stack

| Layer          | Tech |
|----------------|------|
| ML              | XGBoost, SHAP, Optuna, imbalanced-learn (SMOTE), scikit-learn |
| API             | FastAPI, Pydantic v2, Uvicorn |
| Persistence     | SQLAlchemy, SQLite (dev) / PostgreSQL (prod) |
| Frontend        | Streamlit |
| Testing         | pytest, httpx |
| Infra           | Docker, docker-compose, GitHub Actions |

## Running locally

### Option A: Docker Compose (recommended)

```bash
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:8501

```bash
docker compose down
```

### Option B: manual (venv)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 1. Train the model (skip if models/*.pkl are already present)
python -m ml.train

# 2. Start the API
uvicorn api.main:app --reload

# 3. In a second terminal, start the frontend
streamlit run frontend/app.py
```

By default the frontend calls `http://localhost:8000`; override with the
`API_URL` env var if the API runs elsewhere. By default the API uses a
local SQLite file; set `DATABASE_URL` to a Postgres DSN
(`postgresql://user:pass@host:5432/churn`) to use Postgres instead.

### Running the tests

```bash
pytest tests/ -v
```

## Repository layout

```
churn-prediction-system/
├── api/            # FastAPI app: main.py, schemas.py, database.py, models.py
├── ml/             # preprocess.py, train.py, predict.py, explain.py
├── frontend/        # Streamlit app
├── tests/           # pytest suite
├── data/             # telco_churn.csv
├── models/           # churn_model.pkl, churn_model_features.pkl
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
