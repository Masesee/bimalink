from typing import Literal
from pydantic import BaseModel, Field

class SelfReported(BaseModel):
    occupation: Literal["boda_rider", "market_trader", "other"]
    avg_daily_income_band: Literal["under_500", "500_to_1500", "over_1500"]
    years_active: Literal["under_1", "1_to_3", "over_3"]

class SyntheticHistorical(BaseModel):
    momo_txn_frequency: float       # transactions per week
    momo_txn_regularity_score: float # 0-1, higher = more regular intervals
    airtime_topup_cadence: float     # days between top-ups, avg
    sacco_contribution_flag: bool

class LiveBehavioral(BaseModel):
    ussd_session_duration_sec: float
    menu_completion_rate: float      # 0-1
    session_hour_of_day: int         # 0-23
    retry_count: int

class UserProfile(BaseModel):
    phone_number_hash: str
    self_reported: SelfReported
    synthetic_historical: SyntheticHistorical
    live_behavioral: LiveBehavioral

class ShapFactor(BaseModel):
    feature: str
    direction: Literal["increases_risk", "decreases_risk"]
    plain_language: str   # e.g. "Regular M-Pesa activity (main factor)"

class RiskScoreResponse(BaseModel):
    risk_tier: Literal["Low", "Medium", "High"]
    premium_quote_kes: int
    default_probability: float
    shap_top_factors: list[ShapFactor]
    data_disclosure: str
