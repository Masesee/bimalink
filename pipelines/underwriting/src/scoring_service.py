import sys
import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.contracts import (  # noqa: E402
    UserProfile, SelfReported, SyntheticHistorical, LiveBehavioral,
    ShapFactor, RiskScoreResponse
)
from src.train_risk_model import (  # noqa: E402
    preprocess_features, map_default_probability_to_tier
)

# Global variables for model artifacts
model = None
explainer = None
encoder_config = None

SHAP_TEMPLATES = {
    "momo_txn_regularity_score": {
        "decreases_risk": "Regular mobile money activity indicates stable cash flow.",
        "increases_risk": "Irregular mobile money activity suggests inconsistent income."
    },
    "momo_txn_frequency": {
        "decreases_risk": "Frequent mobile money transactions show active business operations.",
        "increases_risk": "Low mobile money transaction frequency suggests limited business volume."
    },
    "airtime_topup_cadence": {
        "decreases_risk": "Frequent airtime top-ups indicate consistent connectivity and activity.",
        "increases_risk": "Longer gaps between airtime top-ups suggest potential cash flow constraints."
    },
    "sacco_contribution_flag": {
        "decreases_risk": "Active SACCO membership indicates strong savings discipline and creditworthiness.",
        "increases_risk": "Lack of cooperative/SACCO membership limits group-based security."
    },
    "occupation_boda_rider": {
        "decreases_risk": "Steady daily cash flows from boda rider operations reduce risk.",
        "increases_risk": "Boda rider occupation is associated with higher operational safety hazards."
    },
    "occupation_market_trader": {
        "decreases_risk": "Fixed market trading operations suggest a stable business location.",
        "increases_risk": "Market trader occupation is exposed to daily market demand volatility."
    },
    "years_active_over_3": {
        "decreases_risk": "Business operational maturity (over 3 years) lowers default risk.",
        "increases_risk": "Operational history of under 3 years increases risk of business instability."
    },
    "avg_daily_income_band_over_1500": {
        "decreases_risk": "Higher average daily income (over KES 1,500) reduces credit risk.",
        "increases_risk": "Average daily income under KES 1,500 reduces premium repayment buffers."
    },
    "avg_daily_income_band_under_500": {
        "increases_risk": "Low average daily income (under KES 500) increases repayment risk.",
        "decreases_risk": "Average daily income exceeds KES 500, lowering default probability."
    },
    "years_active_under_1": {
        "increases_risk": "New operational history (under 1 year) increases credit uncertainty.",
        "decreases_risk": "Operational history exceeds 1 year, lowering credit risk."
    }
}


def get_shap_explanation(feature_name: str, shap_val: float) -> ShapFactor:
    """
    Translates a raw SHAP value and feature name into a user-friendly plain-language explanation.
    """
    direction = "increases_risk" if shap_val > 0 else "decreases_risk"

    # Try exact match
    if feature_name in SHAP_TEMPLATES:
        plain_language = SHAP_TEMPLATES[feature_name][direction]
    else:
        # Try substring match
        matched_key = None
        for key in SHAP_TEMPLATES:
            if key in feature_name:
                matched_key = key
                break

        if matched_key:
            plain_language = SHAP_TEMPLATES[matched_key][direction]
        else:
            # Fallback
            clean_name = feature_name.replace("_", " ").title()
            if direction == "decreases_risk":
                plain_language = f"Favorable status for {clean_name} reduces risk score."
            else:
                plain_language = f"Unfavorable status for {clean_name} increases risk score."

    return ShapFactor(
        feature=feature_name,
        direction=direction,
        plain_language=plain_language
    )


def run_scoring(user_profile: UserProfile) -> RiskScoreResponse:
    """
    Scores a single UserProfile using the loaded model, explainer, and encoding rules.
    """
    if model is None or explainer is None or encoder_config is None:
        raise HTTPException(
            status_code=503,
            detail="Scoring service model artifacts are not loaded yet. Please try again."
        )

    # 1. Flatten UserProfile to match training schema
    profile_dict = user_profile.model_dump()
    flat_profile = {
        "phone_number_hash": profile_dict["phone_number_hash"],
        "occupation": profile_dict["self_reported"]["occupation"],
        "avg_daily_income_band": profile_dict["self_reported"]["avg_daily_income_band"],
        "years_active": profile_dict["self_reported"]["years_active"],
        "momo_txn_frequency": profile_dict["synthetic_historical"]["momo_txn_frequency"],
        "momo_txn_regularity_score": profile_dict["synthetic_historical"]["momo_txn_regularity_score"],
        "airtime_topup_cadence": profile_dict["synthetic_historical"]["airtime_topup_cadence"],
        "sacco_contribution_flag": int(profile_dict["synthetic_historical"]["sacco_contribution_flag"]),
        "ussd_session_duration_sec": profile_dict["live_behavioral"]["ussd_session_duration_sec"],
        "menu_completion_rate": profile_dict["live_behavioral"]["menu_completion_rate"],
        "session_hour_of_day": profile_dict["live_behavioral"]["session_hour_of_day"],
        "retry_count": profile_dict["live_behavioral"]["retry_count"]
    }

    df_single = pd.DataFrame([flat_profile])

    # 2. Encode features using saved config
    df_aligned = preprocess_features(df_single, encoder_config)

    # 3. Predict probability of default
    prob = float(model.predict_proba(df_aligned)[0, 1])

    # 4. Map probability to risk tier and premium quote
    risk_tier, premium_quote = map_default_probability_to_tier(prob)

    # 5. Extract SHAP explanations
    shap_vals = explainer.shap_values(df_aligned)
    instance_shap = shap_vals[0]
    feature_names = encoder_config["feature_names"]

    # Pair features with their SHAP values
    shap_pairs = list(zip(feature_names, instance_shap))
    # Sort by absolute SHAP value descending
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Map the top 2 features to ShapFactor objects
    top_factors = []
    for name, val in shap_pairs[:2]:
        # Filter out features with virtually no SHAP impact to keep explanations meaningful
        if abs(val) > 1e-4:
            top_factors.append(get_shap_explanation(name, val))

    # 6. Build and return the response
    return RiskScoreResponse(
        risk_tier=risk_tier,
        premium_quote_kes=premium_quote,
        default_probability=prob,
        shap_top_factors=top_factors,
        data_disclosure="Score based on synthetic demo data, not real transaction history."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, encoder_config
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "models", "model.joblib")
    explainer_path = os.path.join(root_dir, "models", "explainer.joblib")
    encoder_path = os.path.join(root_dir, "models", "encoder_config.json")

    print("[LOG] Starting scoring service and loading synthetic model artifacts...")
    model = joblib.load(model_path)
    explainer = joblib.load(explainer_path)
    with open(encoder_path, "r") as f:
        encoder_config = json.load(f)
    print("[LOG] All synthetic model artifacts successfully loaded.")
    yield
    print("[LOG] Scoring service shutting down.")


app = FastAPI(
    title="Alternative-Data Insurance Underwriting API (Synthetic Demo)",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    model_loaded = (model is not None and explainer is not None and encoder_config is not None)
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "data_mode": "synthetic"
    }


@app.post("/score", response_model=RiskScoreResponse)
def score_profile(user_profile: UserProfile):
    return run_scoring(user_profile)


@app.get("/score/example", response_model=RiskScoreResponse)
def score_example():
    # Hardcoded profile of a typical boda boda rider for verification
    example_profile = UserProfile(
        phone_number_hash="8f9468bc7f94119d67b2d56c703bdf854e60bf7d5fdf1966a4bc2a44e594df51",
        self_reported=SelfReported(
            occupation="boda_rider",
            avg_daily_income_band="500_to_1500",
            years_active="1_to_3"
        ),
        synthetic_historical=SyntheticHistorical(
            momo_txn_frequency=15.5,
            momo_txn_regularity_score=0.75,
            airtime_topup_cadence=3.2,
            sacco_contribution_flag=True
        ),
        live_behavioral=LiveBehavioral(
            ussd_session_duration_sec=42.5,
            menu_completion_rate=0.88,
            session_hour_of_day=14,
            retry_count=1
        )
    )
    return run_scoring(example_profile)
