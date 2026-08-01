"""
Credit Scoring API — serves the real trained HistGradientBoosting model
(ROC-AUC 0.944 on held-out test set), not the browser-side approximation.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import os
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
medians = bundle["medians"]
grade_map = bundle["grade_map"]

# Real thresholds found during evaluation (see notebook, sections 12 & 13)
THRESHOLDS = {
    "default": 0.50,
    "recall": 0.366,   # maximizes defaulter recall (85%)
    "cost": 0.30,      # minimizes total dollar cost (10% recovery rate assumption)
}

app = FastAPI(
    title="Credit Scoring API",
    description="Serves a HistGradientBoosting model trained on 32,416 real loan records.",
    version="1.0.0",
)

# Allow the frontend (hosted elsewhere) to call this API.
# For production, replace "*" with your actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Applicant(BaseModel):
    person_age: int = Field(..., ge=18, le=100, example=30)
    person_income: float = Field(..., gt=0, example=55000)
    person_emp_length: float = Field(..., ge=0, le=60, example=4)
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., example="B")
    loan_amnt: float = Field(..., gt=0, example=9000)
    loan_int_rate: float = Field(..., gt=0, le=40, example=11.5)
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, example=6)
    cb_person_default_on_file: bool = Field(..., example=False)
    person_home_ownership: Literal["MORTGAGE", "RENT", "OWN", "OTHER"] = Field(..., example="RENT")
    loan_intent: Literal[
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"
    ] = Field(..., example="PERSONAL")
    threshold_mode: Literal["default", "recall", "cost"] = Field("cost", example="cost")


class Contribution(BaseModel):
    feature: str
    label: str
    contribution: float  # change in default probability vs. a median baseline applicant


class ScoreResponse(BaseModel):
    probability_of_default: float
    credit_score: int
    risk_band: str
    verdict: str
    threshold_used: float
    threshold_mode: str
    top_contributions: list[Contribution]
    model_name: str = "HistGradientBoostingClassifier"
    model_roc_auc: float = 0.944


LABELS = {
    "person_age": "Applicant age", "person_income": "Annual income",
    "person_emp_length": "Employment length", "loan_grade": "Lender risk grade",
    "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan % of income", "cb_person_default_on_file": "Prior default on file",
    "cb_person_cred_hist_length": "Credit history length", "loan_income_ratio": "Loan-to-income ratio",
    "person_home_ownership_OTHER": "Home: other", "person_home_ownership_OWN": "Home: own",
    "person_home_ownership_RENT": "Home: rent", "loan_intent_EDUCATION": "Purpose: education",
    "loan_intent_HOMEIMPROVEMENT": "Purpose: home improvement", "loan_intent_MEDICAL": "Purpose: medical",
    "loan_intent_PERSONAL": "Purpose: personal", "loan_intent_VENTURE": "Purpose: venture",
}


def build_feature_row(a: Applicant) -> dict:
    loan_percent_income = min(a.loan_amnt / a.person_income, 1.0)
    return {
        "person_age": a.person_age,
        "person_income": a.person_income,
        "person_emp_length": a.person_emp_length,
        "loan_grade": grade_map[a.loan_grade],
        "loan_amnt": a.loan_amnt,
        "loan_int_rate": a.loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": int(a.cb_person_default_on_file),
        "cb_person_cred_hist_length": a.cb_person_cred_hist_length,
        "loan_income_ratio": loan_percent_income,
        "person_home_ownership_OTHER": int(a.person_home_ownership == "OTHER"),
        "person_home_ownership_OWN": int(a.person_home_ownership == "OWN"),
        "person_home_ownership_RENT": int(a.person_home_ownership == "RENT"),
        "loan_intent_EDUCATION": int(a.loan_intent == "EDUCATION"),
        "loan_intent_HOMEIMPROVEMENT": int(a.loan_intent == "HOMEIMPROVEMENT"),
        "loan_intent_MEDICAL": int(a.loan_intent == "MEDICAL"),
        "loan_intent_PERSONAL": int(a.loan_intent == "PERSONAL"),
        "loan_intent_VENTURE": int(a.loan_intent == "VENTURE"),
    }


def predict_proba_row(row: dict) -> float:
    df = pd.DataFrame([row])[feature_names]
    scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    return float(model.predict_proba(scaled)[0, 1])


def local_contributions(row: dict, top_n: int = 7) -> list[Contribution]:
    """
    Approximate per-applicant explanation: for each feature, swap it to the
    population median (holding everything else fixed) and measure how much
    the predicted default probability changes. This is a real, computed
    marginal-effect explanation — not SHAP, but not fabricated either.
    For production-grade per-feature attribution, install `shap` and use
    shap.TreeExplainer(model) instead (commented example in the notebook).
    """
    base_prob = predict_proba_row(row)
    contributions = []
    for f in feature_names:
        if row[f] == medians.get(f, row[f]):
            continue
        swapped = dict(row)
        swapped[f] = medians.get(f, row[f])
        swapped_prob = predict_proba_row(swapped)
        # positive contribution == this feature's actual value raises risk
        # relative to a median applicant
        contrib = base_prob - swapped_prob
        contributions.append(Contribution(feature=f, label=LABELS.get(f, f), contribution=round(contrib, 4)))
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contributions[:top_n]


def score_from_prob(prob: float, threshold_mode: str):
    score = int(round(max(300, min(850, 850 - prob * 550))))
    threshold = THRESHOLDS[threshold_mode]
    band = 0.06
    if prob < threshold - band:
        risk_band, verdict = "Low risk", "APPROVE"
    elif prob < threshold + band:
        risk_band, verdict = "Moderate risk", "REVIEW"
    else:
        risk_band, verdict = "High risk", "DECLINE"
    return score, threshold, risk_band, verdict


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Credit Scoring API is running. See /docs for interactive documentation.",
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "HistGradientBoostingClassifier",
        "trained_on_records": 32416,
        "test_set_size": 6484,
        "metrics": {
            "accuracy": 0.923, "precision": 0.847, "recall": 0.791,
            "f1": 0.818, "roc_auc": 0.944,
        },
        "thresholds": THRESHOLDS,
    }


@app.post("/predict", response_model=ScoreResponse)
def predict(applicant: Applicant):
    try:
        row = build_feature_row(applicant)
        prob = predict_proba_row(row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    prob = round(prob, 4)
    score, threshold, risk_band, verdict = score_from_prob(prob, applicant.threshold_mode)

    return ScoreResponse(
        probability_of_default=prob,
        credit_score=score,
        risk_band=risk_band,
        verdict=verdict,
        threshold_used=threshold,
        threshold_mode=applicant.threshold_mode,
        top_contributions=local_contributions(row),
    )


class BatchResultRow(BaseModel):
    row: int
    probability_of_default: float
    credit_score: int
    risk_band: str
    verdict: str


class BatchResponse(BaseModel):
    count: int
    threshold_used: float
    threshold_mode: str
    approve_count: int
    review_count: int
    decline_count: int
    results: list[BatchResultRow]
    evaluation: dict | None = None  # only present if the uploaded file included a loan_status column


REQUIRED_BATCH_COLUMNS = {
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
}


def preprocess_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same cleaning, feature engineering, and encoding used in
    training (see backend/train.py) to an arbitrary uploaded dataset, then
    aligns the result to the exact feature columns the model expects —
    filling in any one-hot category missing from this particular file with 0.
    """
    df = df_raw.copy()
    df["person_emp_length"] = pd.to_numeric(df["person_emp_length"], errors="coerce")
    df["loan_int_rate"] = pd.to_numeric(df["loan_int_rate"], errors="coerce")
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    df["loan_income_ratio"] = df["loan_amnt"] / df["person_income"]
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    # Normalize casing/whitespace on every categorical column BEFORE encoding.
    # Without this, a file that writes "rent" instead of "RENT" (or "b" instead
    # of "B") would either fail confusingly or, worse, silently produce a
    # one-hot column the model has never seen — which get_dummies + the
    # feature-alignment step below would silently zero out, quietly treating
    # that applicant as the baseline category instead of erroring.
    df["loan_grade"] = df["loan_grade"].astype(str).str.strip().str.upper().map(grade_map)
    if df["loan_grade"].isna().any():
        raise ValueError("loan_grade must be one of A-G")

    df["cb_person_default_on_file"] = (
        df["cb_person_default_on_file"].astype(str).str.strip().str.upper().map(
            {"N": 0, "Y": 1, "FALSE": 0, "TRUE": 1, "0": 0, "1": 1}
        )
    )
    if df["cb_person_default_on_file"].isna().any():
        raise ValueError("cb_person_default_on_file must be Y/N (or true/false, 0/1)")

    df["person_home_ownership"] = df["person_home_ownership"].astype(str).str.strip().str.upper()
    df["loan_intent"] = df["loan_intent"].astype(str).str.strip().str.upper()

    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"])

    for f in feature_names:
        if f not in df.columns:
            df[f] = 0

    return df[feature_names]


@app.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(
    file: UploadFile = File(...),
    threshold_mode: Literal["default", "recall", "cost"] = Query("cost"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    content = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    missing = REQUIRED_BATCH_COLUMNS - set(df_raw.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {sorted(missing)}. "
                    f"Expected columns: {sorted(REQUIRED_BATCH_COLUMNS)}",
        )

    has_labels = "loan_status" in df_raw.columns
    y_true = df_raw["loan_status"].values if has_labels else None

    try:
        aligned = preprocess_batch(df_raw.drop(columns=["loan_status"], errors="ignore"))
        scaled = pd.DataFrame(scaler.transform(aligned), columns=feature_names)
        probs = model.predict_proba(scaled)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not score file: {e}")

    threshold = THRESHOLDS[threshold_mode]
    results = []
    approve_count = review_count = decline_count = 0
    for i, prob in enumerate(probs):
        score, _, risk_band, verdict = score_from_prob(float(prob), threshold_mode)
        if verdict == "APPROVE":
            approve_count += 1
        elif verdict == "REVIEW":
            review_count += 1
        else:
            decline_count += 1
        results.append(BatchResultRow(
            row=i, probability_of_default=round(float(prob), 4),
            credit_score=score, risk_band=risk_band, verdict=verdict,
        ))

    evaluation = None
    if has_labels:
        preds_binary = (probs >= threshold).astype(int)
        evaluation = {
            "accuracy": round(float(accuracy_score(y_true, preds_binary)), 4),
            "precision": round(float(precision_score(y_true, preds_binary, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds_binary, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds_binary, zero_division=0)), 4),
        }

    return BatchResponse(
        count=len(results),
        threshold_used=threshold,
        threshold_mode=threshold_mode,
        approve_count=approve_count,
        review_count=review_count,
        decline_count=decline_count,
        results=results,
        evaluation=evaluation,
    )
