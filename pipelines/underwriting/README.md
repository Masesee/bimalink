# Alternative-Data Insurance Underwriting System (ML Core)

This repository contains the machine learning core for an alternative-data insurance underwriting system designed for informal sector workers (boda boda riders, market traders, etc.) in Kenya who lack formal credit history.

> [!IMPORTANT]
> **Synthetic Data Disclosure**: All data used, generated, and processed within this repository is strictly **synthetic** and designed for hackathon demonstration purposes. No real transactions, credit, or mobile money history are used.

---

## Architecture Overview

The system consists of three independent, standalone testable components:
1. **Synthetic Profile Generator (`src/generate_synthetic_profiles.py`)**: Simulates 3,000 user profiles with non-traditional behavioral, historical, and session-level features.
2. **Risk Model Trainer (`src/train_risk_model.py`)**: Trains an XGBoost classifier with early stopping, generates SHAP explainers, and packages model artifacts.
3. **FastAPI Scoring Service (`src/scoring_service.py`)**: Serves model predictions, risk tiers, premium quotes, and top SHAP explanations for live USSD integration.

---

## Current Model Performance

* **Test ROC-AUC Score**: **0.8288**
* **What this means**: An ROC-AUC of ~0.83 indicates a highly robust model that has successfully learned key directional associations (such as SACCO membership, transaction regularity, and airtime top-up frequency) without suffering from data leakage (which would result in an unrealistically high AUC > 0.97) or random noise classification (which would result in an AUC < 0.65). It represents a strong, realistic model suitable for commercial evaluation.

---

## How to Run

### 1. Installation
Install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
Run the profile generator to create the CSV datasets:
```bash
python src/generate_synthetic_profiles.py
```
This generates:
* `data/synthetic_profiles.csv` (3,000 rows, gitignored)
* `data/sample_10_rows.csv` (10 rows, committed for schema reference)

### 3. Train the Underwriting Model
Train the model, evaluate it, and save the binary model and explainer artifacts:
```bash
python src/train_risk_model.py
```
This saves:
* `models/model.joblib`
* `models/explainer.joblib`
* `models/encoder_config.json`

### 4. Run the scoring service
Start the FastAPI server using Uvicorn:
```bash
python -m uvicorn src.scoring_service:app --reload
```

---

## Example API / Curl Calls

Once the server is running on `http://127.0.0.1:8000`, verify using the following commands:

### GET `/health`
Check if the API is active and the model artifacts are successfully loaded:
```bash
curl -s http://127.0.0.1:8000/health
```
**Response**:
```json
{"status":"ok","model_loaded":true,"data_mode":"synthetic"}
```

### GET `/score/example`
Triggers scoring on a pre-defined test profile (useful for quick USSD callback validation):
```bash
curl -s http://127.0.0.1:8000/score/example
```
**Response**:
```json
{
  "risk_tier": "Low",
  "premium_quote_kes": 150,
  "default_probability": 0.04055299237370491,
  "shap_top_factors": [
    {
      "feature": "sacco_contribution_flag",
      "direction": "decreases_risk",
      "plain_language": "Active SACCO membership indicates strong savings discipline and creditworthiness."
    },
    {
      "feature": "momo_txn_regularity_score",
      "direction": "decreases_risk",
      "plain_language": "Regular mobile money activity indicates stable cash flow."
    }
  ],
  "data_disclosure": "Score based on synthetic demo data, not real transaction history."
}
```

### POST `/score`
Send a live profile payload for real-time underwriting risk tiering and premium quoting:
```bash
curl -s -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number_hash": "a1b2c3d4e5f6g7h8i9j0",
    "self_reported": {
      "occupation": "other",
      "avg_daily_income_band": "under_500",
      "years_active": "under_1"
    },
    "synthetic_historical": {
      "momo_txn_frequency": 2.0,
      "momo_txn_regularity_score": 0.21,
      "airtime_topup_cadence": 12.0,
      "sacco_contribution_flag": false
    },
    "live_behavioral": {
      "ussd_session_duration_sec": 120.0,
      "menu_completion_rate": 0.45,
      "session_hour_of_day": 23,
      "retry_count": 3
    }
  }'
```
**Response**:
```json
{
  "risk_tier": "High",
  "premium_quote_kes": 500,
  "default_probability": 0.7419211864471436,
  "shap_top_factors": [
    {
      "feature": "airtime_topup_cadence",
      "direction": "increases_risk",
      "plain_language": "Longer gaps between airtime top-ups suggest potential cash flow constraints."
    },
    {
      "feature": "momo_txn_regularity_score",
      "direction": "increases_risk",
      "plain_language": "Irregular mobile money activity suggests inconsistent income."
    }
  ],
  "data_disclosure": "Score based on synthetic demo data, not real transaction history."
}
```

---

## Running Unit & Integration Tests

We enforce 100% test coverage using `pytest`. Execute the test suites via:
```bash
python -m pytest pipelines/underwriting/
```
All tests verify integrity, class distributions, model overfitting, error handling (HTTP 422 validations), and latency targets (< 1.0s).
