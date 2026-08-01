# Credit Scoring System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning system that predicts loan default risk from applicant and loan data, with a cost-sensitive decision framework and a full-stack web application for real-time scoring.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Encoding](#3-encoding)
  - [4. Train-Test Split & Scaling](#4-train-test-split--scaling)
  - [5. Models Trained](#5-models-trained)
  - [6. Model Comparison](#6-model-comparison)
  - [7. Decision Threshold Tuning](#7-decision-threshold-tuning)
  - [8. Cost-Sensitive Evaluation](#8-cost-sensitive-evaluation)
  - [9. Explainability](#9-explainability)
- [Final Model](#final-model)
- [Web Application](#web-application)
- [User Guide](#user-guide)
- [Setup Instructions](#setup-instructions)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

The goal of this project is to predict whether a loan applicant is likely to default, using historical loan data. Beyond model accuracy, the project evaluates model choices in terms of real dollar cost, since in lending, false negatives (approving a bad loan) and false positives (rejecting a good customer) carry very different financial consequences.

The project has two parts:
1. **Model development** (`notebook/`) — data cleaning, feature engineering, model training, evaluation, and cost analysis.
2. **Web application** (`backend/`, `frontend/`) — a FastAPI service serving the trained model, and a React dashboard for scoring applicants interactively.

## Features

- Predicts loan default risk using Machine Learning
- Cost-sensitive decision analysis for lending
- FastAPI REST API for real-time predictions
- Modern React dashboard with Tailwind CSS
- Batch applicant scoring using CSV upload
- Interactive credit score gauge
- Model comparison and performance metrics
- Feature importance and explainability
- Configurable decision thresholds
- Client-side fallback prediction when the backend is unavailable

## Dataset

- **Source**: Credit Risk Dataset (32,581 rows, 12 columns)
- **Target variable**: `loan_status` (0 = repaid, 1 = defaulted)
- **Features**: applicant demographics (age, income, employment length, home ownership), loan details (amount, interest rate, grade, purpose), and credit history (prior defaults, credit history length)

## Project Structure

```
credit-scoring-app/
├── notebook/
│   └── Credit_Scoring_Model_Enhanced.ipynb   # full training and evaluation pipeline
├── docs/
│   └── USER_GUIDE.md          # plain-language guide to every field, button, and result
├── backend/
│   ├── main.py                # FastAPI app (/predict, /predict-batch, /model-info)
│   ├── train.py                # training script — regenerates model_bundle.pkl
│   ├── data/
│   │   └── credit_risk_dataset.csv
│   ├── model_bundle.pkl       # trained model + scaler + feature medians
│   ├── requirements.txt
│   └── README.md              # backend API reference
├── sample_data/
│   └── sample_applicants.csv  # 25 real rows for testing the CSV upload feature
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.jsx            # scoring dashboard
│       ├── main.jsx
│       └── index.css
└── README.md
```

## Methodology

### 1. Data Cleaning

- Converted `person_emp_length` and `loan_int_rate` to numeric, coercing invalid entries to `NaN`.
- Filled missing values in both columns with the column median.
- Removed 165 duplicate rows (32,581 → 32,416 rows).

### 2. Feature Engineering

Added `loan_income_ratio = loan_amnt / person_income`, computed before the train-test split so it is available to every model. This turned out to be the single most predictive feature in the dataset (see [Explainability](#9-explainability)).

### 3. Encoding

- **Ordinal encoding** for `loan_grade` (A–G → 0–6).
- **Binary mapping** for `cb_person_default_on_file` (Y/N → 1/0).
- **One-hot encoding** for `person_home_ownership` and `loan_intent`, dropping the first category of each to avoid multicollinearity.

### 4. Train-Test Split & Scaling

- 80/20 train-test split, stratified on the target variable, `random_state=42`.
- Features standardized with `StandardScaler`, fit on the training set only and applied to both sets to avoid data leakage.

### 5. Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear baseline |
| Random Forest | Ensemble of trees |
| Random Forest (class-weight balanced) | Tests whether reweighting the minority class improves recall |
| HistGradientBoosting (class-weight balanced) | Gradient-boosted trees, final model |

### 6. Model Comparison

Evaluated on the held-out test set (6,484 records):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 85.9% | 75.7% | 52.5% | 62.0% | 87.0% |
| Decision Tree | 89.1% | 73.9% | 77.7% | 75.7% | 85.0% |
| Random Forest | 93.6% | 97.1% | 72.8% | 83.2% | 93.3% |
| Random Forest (balanced) | 93.5% | 98.1% | 71.7% | 82.8% | 93.4% |
| **HistGradientBoosting (balanced)** | **92.3%** | **84.7%** | **79.1%** | **81.8%** | **94.4%** |

Class-weight balancing on Random Forest did not meaningfully improve recall. HistGradientBoosting was selected as the final model for its higher ROC-AUC and better recall.

### 7. Decision Threshold Tuning

The default 0.5 classification threshold is not necessarily the best cutoff. Three thresholds were evaluated on the test set:

| Threshold | Optimized for | Recall | Precision |
|---|---|---|---|
| 0.50 | Default | 79.1% | 84.7% |
| 0.366 | Maximum recall (≥85%) | 85.0% | 66.2% |
| 0.30 | Minimum dollar cost | — | — |

### 8. Cost-Sensitive Evaluation

Classification metrics alone don't capture financial impact. Using each applicant's real loan amount and interest rate:

- **Cost of a missed defaulter** = loan principal × (1 − recovery rate), recovery rate assumed at 10%.
- **Cost of a wrongly rejected good customer** = loan amount × interest rate × 1 year (foregone interest income).

| Scenario | Total cost (test set) |
|---|---|
| Approve everyone (no model) | $13,749,458 |
| Reject everyone | $5,039,935 |
| Model at default threshold (0.50) | $2,774,378 |
| **Model at cost-optimal threshold (0.30)** | **$2,450,492** |

Using the model reduces cost by approximately $11.3M versus not screening applicants at all. Tuning the threshold from 0.50 to 0.30 saves a further ~$324K (11.7%).

A sensitivity check across different recovery-rate assumptions (0%, 10%, 25%, 40%) confirmed the optimal threshold stays in the 0.25–0.30 range, indicating the result is not overly sensitive to that assumption.

### 9. Explainability

Permutation importance on the final model identified the strongest predictors of default:

1. Loan-to-income ratio
2. Lender-assigned loan grade
3. Applicant income
4. Home ownership status (rent vs. own)
5. Loan percent of income

## Final Model

**HistGradientBoostingClassifier**, trained with class-weight balancing.

- Accuracy: 92.3%
- Precision: 84.7%
- Recall: 79.1%
- F1: 81.8%
- ROC-AUC: 94.4%

## Web Application

- **Backend**: FastAPI service exposing `/predict` and `/model-info`, serving the trained model directly.
- **Frontend**: React + Vite + Tailwind dashboard with:
  - A live scoring form with a credit score gauge (300–850 scale)
  - A per-applicant score breakdown showing which factors raised or lowered the score
  - A model comparison and feature importance view
  - A selectable decision threshold (default / maximum recall / minimum cost)
  - A batch upload tab — upload a CSV of applicants and score all of them at once, with a downloadable results file and automatic accuracy metrics if the file includes real outcomes

If the backend is unreachable, the frontend falls back to a client-side Logistic Regression approximation so the interface remains functional without a server.

## User Guide

For a plain-language walkthrough of every field, button, and result in the app — see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Setup Instructions

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

To retrain the model from the dataset:

```bash
python train.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`).

Start the backend before the frontend so the app connects to the real model on first load.

## Tech Stack

### Machine Learning & Data Science

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### Backend

- FastAPI
- Uvicorn
- Pydantic
- Joblib

### Frontend

- React
- Vite
- Tailwind CSS
- Recharts
- Lucide React
- PapaParse

### Development Tools

- Git
- GitHub
- VS Code
- npm
- PostCSS
- Autoprefixer

## Future Improvements

- SHAP explainability
- Docker support
- Cloud deployment
- User authentication
- Database integration
- Loan approval recommendation engine
- Multi-model ensemble learning

# 👩‍💻 Author

**Faiza Soomro**

Full Stack Web Developer & AI/ML Engineer

Passionate about building intelligent applications using Machine Learning, Artificial Intelligence, Python, and modern web technologies.

GitHub:  
https://github.com/FaizaSoOmro256

LinkedIn:  
https://www.linkedin.com/in/faiza-shafi-muhammad-a1a92325b/

## ⭐ Support This Project

If you found this project useful, innovative, or helpful, please consider giving it a ⭐ star on GitHub.

Your support motivates further improvements, new features, and more open-source AI projects.

⭐ **Rate this project:**  
https://github.com/FaizaSoOmro256/CodeAlpha_LanguageTranslationToolAI

Thank you for your support! 🚀

# 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and improve this project for personal, educational, and commercial purposes, provided that the original author is credited.

See the [LICENSE](LICENSE) file for more details.
