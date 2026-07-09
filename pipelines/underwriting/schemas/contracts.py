from typing import Literal
from pydantic import BaseModel

ProvenanceType = Literal["self_reported", "statement_verified"]


class SelfReported(BaseModel):
    occupation: Literal["boda_rider", "market_trader", "other"]
    avg_daily_income_band: Literal["under_500", "500_to_1500", "over_1500"]
    years_active: Literal["under_1", "1_to_3", "over_3"]


class SyntheticHistorical(BaseModel):
    # transactions per week
    momo_txn_frequency: float
    # 0-1, higher = more regular intervals.
    # NOTE: When data_provenance == "self_reported", momo_txn_regularity_score
    # is a proxy value derived from years_active, not a directly measured figure.
    # Any SHAP plain-language template must reflect this (e.g. refer to work tenure,
    # not actual transactional regularity).
    momo_txn_regularity_score: float
    sacco_contribution_flag: bool


class LiveBehavioral(BaseModel):
    ussd_session_duration_sec: float
    # 0-1
    menu_completion_rate: float
    # 0-23
    session_hour_of_day: int
    retry_count: int


class UserProfile(BaseModel):
    phone_number_hash: str
    data_provenance: ProvenanceType
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
    data_provenance: ProvenanceType
    shap_top_factors: list[ShapFactor]
    data_disclosure: str
