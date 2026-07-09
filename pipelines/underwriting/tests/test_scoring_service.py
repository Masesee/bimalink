import sys
import os
import time
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring_service import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["data_mode"] == "synthetic"


def test_score_valid(client):
    payload = {
        "phone_number_hash": "8f9468bc7f94119d67b2d56c703bdf854e60bf7d5fdf1966a4bc2a44e594df51",
        "data_provenance": "self_reported",
        "self_reported": {
            "occupation": "market_trader",
            "avg_daily_income_band": "over_1500",
            "years_active": "over_3"
        },
        "synthetic_historical": {
            "momo_txn_frequency": 25.0,
            "momo_txn_regularity_score": 0.85,
            "sacco_contribution_flag": True
        },
        "live_behavioral": {
            "ussd_session_duration_sec": 30.0,
            "menu_completion_rate": 0.95,
            "session_hour_of_day": 10,
            "retry_count": 0
        }
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert isinstance(data["premium_quote_kes"], int)
    assert data["premium_quote_kes"] in [150, 300, 500]
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["data_provenance"] == "self_reported"

    assert isinstance(data["shap_top_factors"], list)
    assert len(data["shap_top_factors"]) >= 1

    # Confirm shape top factors schema
    for factor in data["shap_top_factors"]:
        assert "feature" in factor
        assert factor["direction"] in ["increases_risk", "decreases_risk"]
        assert "plain_language" in factor

    assert "self-reported" in data["data_disclosure"].lower()


def test_score_malformed(client):
    # Missing 'self_reported' block entirely
    payload = {
        "phone_number_hash": "8f9468bc7f94119d67b2d56c703bdf854e60bf7d5fdf1966a4bc2a44e594df51",
        "data_provenance": "self_reported",
        "synthetic_historical": {
            "momo_txn_frequency": 25.0,
            "momo_txn_regularity_score": 0.85,
            "sacco_contribution_flag": True
        },
        "live_behavioral": {
            "ussd_session_duration_sec": 30.0,
            "menu_completion_rate": 0.95,
            "session_hour_of_day": 10,
            "retry_count": 0
        }
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 422  # Enforces Pydantic model validation


def test_score_example_latency(client):
    start_time = time.time()
    response = client.get("/score/example")
    elapsed = time.time() - start_time

    assert response.status_code == 200
    assert elapsed < 1.0, f"Latency gate failed: request took {elapsed:.4f} seconds"

    data = response.json()
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert "self-reported" in data["data_disclosure"].lower()


def test_provenance_and_shap_branching(client):
    """Confirm SHAP templates branch correctly and output distinct text by data_provenance."""
    # We will construct a profile with highly regular transactions (regularity = 0.95)
    # under self_reported and statement_verified and verify the descriptions differ
    base_payload = {
        "phone_number_hash": "8f9468bc7f94119d67b2d56c703bdf854e60bf7d5fdf1966a4bc2a44e594df51",
        "self_reported": {
            "occupation": "market_trader",
            "avg_daily_income_band": "over_1500",
            "years_active": "over_3"
        },
        "synthetic_historical": {
            "momo_txn_frequency": 1.0,  # low frequency
            "momo_txn_regularity_score": 0.95,  # High regularity score
            "sacco_contribution_flag": True
        },
        "live_behavioral": {
            "ussd_session_duration_sec": 10.0,
            "menu_completion_rate": 1.0,
            "session_hour_of_day": 12,
            "retry_count": 0
        }
    }

    # 1. Test Self-Reported request
    payload_self = dict(base_payload)
    payload_self["data_provenance"] = "self_reported"
    r_self = client.post("/score", json=payload_self)
    assert r_self.status_code == 200
    res_self = r_self.json()
    assert res_self["data_provenance"] == "self_reported"

    # Find the regularity score explanation
    regularity_factor_self = next(
        (f for f in res_self["shap_top_factors"] if "momo_txn_regularity_score" in f["feature"]),
        None
    )

    # 2. Test Statement-Verified request
    payload_verified = dict(base_payload)
    payload_verified["data_provenance"] = "statement_verified"
    r_verified = client.post("/score", json=payload_verified)
    assert r_verified.status_code == 200
    res_verified = r_verified.json()
    assert res_verified["data_provenance"] == "statement_verified"

    regularity_factor_verified = next(
        (f for f in res_verified["shap_top_factors"] if "momo_txn_regularity_score" in f["feature"]),
        None
    )

    # If regularity score is a top factor in both, they must have different explanations.
    if regularity_factor_self and regularity_factor_verified:
        assert regularity_factor_self["plain_language"] != regularity_factor_verified["plain_language"]
        self_lang = regularity_factor_self["plain_language"].lower()
        verified_lang = regularity_factor_verified["plain_language"].lower()
        assert "work history" in self_lang or "tenure" in self_lang
        assert "transaction" in verified_lang or "mobile money" in verified_lang
