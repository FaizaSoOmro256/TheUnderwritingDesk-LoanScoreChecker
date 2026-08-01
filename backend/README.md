# Backend — Credit Scoring API

FastAPI service that serves the trained `HistGradientBoostingClassifier` model (ROC-AUC 0.944).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Endpoints

### `GET /`
Health check.

### `GET /model-info`
Returns model metadata, evaluation metrics, and available decision thresholds.

### `POST /predict`
Scores a single applicant.

**Request body:**

```json
{
  "person_age": 30,
  "person_income": 55000,
  "person_emp_length": 4,
  "loan_grade": "B",
  "loan_amnt": 9000,
  "loan_int_rate": 11.5,
  "cb_person_cred_hist_length": 6,
  "cb_person_default_on_file": false,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "threshold_mode": "cost"
}
```

`threshold_mode` accepts `"default"` (0.50), `"recall"` (0.366, tuned for maximum defaulter recall), or `"cost"` (0.30, tuned to minimize total dollar cost).

**Response:**

```json
{
  "probability_of_default": 0.0821,
  "credit_score": 761,
  "risk_band": "Low risk",
  "verdict": "APPROVE",
  "threshold_used": 0.30,
  "threshold_mode": "cost",
  "top_contributions": [
    { "feature": "loan_income_ratio", "label": "Loan-to-income ratio", "contribution": -0.041 }
  ],
  "model_name": "HistGradientBoostingClassifier",
  "model_roc_auc": 0.944
}
```

### `POST /predict-batch`
Scores an entire uploaded CSV of applicants at once.

**Request**: multipart form upload with a `file` field (`.csv`) and an optional `threshold_mode` query parameter (`default` | `recall` | `cost`, defaults to `cost`).

Required columns: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `cb_person_default_on_file`, `cb_person_cred_hist_length`.

If the file also includes a `loan_status` column (0 = repaid, 1 = defaulted), the response includes accuracy/precision/recall/F1 computed against those real outcomes — useful for validating the model against a labeled holdout set.

```bash
curl -X POST "http://localhost:8000/predict-batch?threshold_mode=cost" \
  -F "file=@../sample_data/sample_applicants.csv"
```

**Response:**

```json
{
  "count": 25,
  "threshold_used": 0.30,
  "threshold_mode": "cost",
  "approve_count": 15,
  "review_count": 3,
  "decline_count": 7,
  "results": [
    { "row": 0, "probability_of_default": 0.91, "credit_score": 303, "risk_band": "High risk", "verdict": "DECLINE" }
  ],
  "evaluation": {
    "accuracy": 0.72, "precision": 0.65, "recall": 0.7, "f1": 0.67
  }
}
```

`evaluation` is omitted entirely if the uploaded file had no `loan_status` column.

## Explainability method

`top_contributions` uses a marginal-effect approximation: each feature is swapped to its population median (holding all other values fixed), and the resulting change in predicted default probability is recorded. This is computed directly from the trained model, not a placeholder.

For SHAP-based attribution instead, install `shap` and replace `local_contributions()` in `main.py`:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(scaled_row)
```

## Retraining

```bash
python train.py
```

Reads `data/credit_risk_dataset.csv`, retrains the model, and overwrites `model_bundle.pkl`. Replace the CSV with new data to retrain on an updated dataset.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI application |
| `train.py` | Training script |
| `model_bundle.pkl` | Trained model, scaler, and feature medians |
| `data/credit_risk_dataset.csv` | Training dataset |
| `requirements.txt` | Python dependencies |

See also `sample_data/sample_applicants.csv` (in the project root) — 25 real rows for testing `/predict-batch`.

## Deployment

Any Python host works (Render, Railway, Fly.io, a VM).

1. Push this `backend/` folder to a Git repository.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Update `API_BASE_URL` in `frontend/src/App.jsx` to point to the deployed URL.
5. Restrict `allow_origins` in `main.py` to the frontend's domain instead of `"*"` before going live.
