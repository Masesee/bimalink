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
        "self_reported": {
            "occupation": "market_trader",
            "avg_daily_income_band": "over_1500",
            "years_active": "over_3"
        },
        "synthetic_historical": {
            "momo_txn_frequency": 25.0,
            "momo_txn_regularity_score": 0.85,
            "airtime_topup_cadence": 2.0,
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

    assert isinstance(data["shap_top_factors"], list)
    assert len(data["shap_top_factors"]) >= 1

    # Confirm shape top factors schema
    for factor in data["shap_top_factors"]:
        assert "feature" in factor
        assert factor["direction"] in ["increases_risk", "decreases_risk"]
        assert "plain_language" in factor

    assert "synthetic" in data["data_disclosure"].lower()


def test_score_malformed(client):
    # Missing 'self_reported' block entirely
    payload = {
        "phone_number_hash": "8f9468bc7f94119d67b2d56c703bdf854e60bf7d5fdf1966a4bc2a44e594df51",
        "synthetic_historical": {
            "momo_txn_frequency": 25.0,
            "momo_txn_regularity_score": 0.85,
            "airtime_topup_cadence": 2.0,
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
    assert "synthetic" in data["data_disclosure"].lower()
