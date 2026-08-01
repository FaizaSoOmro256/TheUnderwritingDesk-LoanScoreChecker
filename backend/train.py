"""
Trains the credit scoring model from credit_risk_dataset.csv and exports
model_bundle.pkl for the FastAPI backend to serve.

This is the exact pipeline used to build the model_bundle.pkl already
included in this repo — run it yourself to verify the numbers, or re-run
it after updating data/credit_risk_dataset.csv with fresh data.

Usage:
    cd backend
    pip install -r requirements.txt
    python train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model_bundle.pkl")

GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}


def main():
    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # ---- Clean ----
    df["person_emp_length"] = pd.to_numeric(df["person_emp_length"], errors="coerce")
    df["loan_int_rate"] = pd.to_numeric(df["loan_int_rate"], errors="coerce")
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped {before - df.shape[0]} duplicate rows -> {df.shape[0]} rows remain")

    # ---- Feature engineering (done BEFORE the split, so it's actually used) ----
    df["loan_income_ratio"] = df["loan_amnt"] / df["person_income"]

    # ---- Encoding ----
    df["loan_grade"] = df["loan_grade"].map(GRADE_MAP)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"N": 0, "Y": 1})
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    # ---- Split ----
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    feature_names = list(X.columns)
    print(f"Feature count: {len(feature_names)}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Scale ----
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    # ---- Train ----
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---- Export bundle ----
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "medians": X_train.median().to_dict(),
        "grade_map": GRADE_MAP,
    }
    joblib.dump(bundle, OUTPUT_PATH)
    print(f"\nSaved model bundle to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
