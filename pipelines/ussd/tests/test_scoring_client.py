import sys
import os
import pytest
import requests
import responses

# Add pipelines/ussd/src and pipelines/underwriting to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "ussd"))
sys.path.append(os.path.join(root_dir, "underwriting"))

from src.scoring_client import get_risk_score, ScoringServiceUnavailable  # noqa: E402
from schemas.contracts import UserProfile, SelfReported, SyntheticHistorical, LiveBehavioral  # noqa: E402


@pytest.fixture
def dummy_profile():
    return UserProfile(
        phone_number_hash="dummy_hash_123",
        self_reported=SelfReported(
            occupation="boda_rider",
            avg_daily_income_band="500_to_1500",
            years_active="1_to_3"
        ),
        synthetic_historical=SyntheticHistorical(
            momo_txn_frequency=5.5,
            momo_txn_regularity_score=0.8,
            airtime_topup_cadence=2.1,
            sacco_contribution_flag=True
        ),
        live_behavioral=LiveBehavioral(
            ussd_session_duration_sec=15.2,
            menu_completion_rate=1.0,
            session_hour_of_day=14,
            retry_count=0
        )
    )


@responses.activate
def test_get_risk_score_success(dummy_profile):
    """Tests successful scoring service response parsing."""
    url = "http://localhost:8000/score"
    mock_response = {
        "risk_probability": 0.125,
        "risk_tier": "Low",
        "premium_quote_kes": 150,
        "default_probability": 0.125,
        "shap_top_factors": [
            {
                "feature": "momo_txn_regularity_score",
                "direction": "decreases_risk",
                "plain_language": "Highly regular transaction pattern"
            },
            {
                "feature": "sacco_contribution_flag",
                "direction": "decreases_risk",
                "plain_language": "Active membership in a financial cooperative"
            }
        ],
        "data_disclosure": "Consent details"
    }

    responses.add(
        responses.POST,
        url,
        json=mock_response,
        status=200
    )

    res = get_risk_score(dummy_profile)
    assert res.risk_tier == "Low"
    assert res.premium_quote_kes == 150
    assert len(res.shap_top_factors) == 2
    assert res.shap_top_factors[0].feature == "momo_txn_regularity_score"


@responses.activate
def test_get_risk_score_timeout(dummy_profile):
    """Tests timeout handling leading to ScoringServiceUnavailable exception."""
    url = "http://localhost:8000/score"

    responses.add(
        responses.POST,
        url,
        body=requests.exceptions.Timeout("Connection timed out"),
        status=500  # Will be ignored since error is thrown
    )

    with pytest.raises(ScoringServiceUnavailable) as excinfo:
        get_risk_score(dummy_profile)

    assert "timed out" in str(excinfo.value)


@responses.activate
def test_get_risk_score_server_error(dummy_profile):
    """Tests non-200 server response handling."""
    url = "http://localhost:8000/score"

    responses.add(
        responses.POST,
        url,
        body="Internal Server Error",
        status=500
    )

    with pytest.raises(ScoringServiceUnavailable) as excinfo:
        get_risk_score(dummy_profile)

    assert "non-200 status code: 500" in str(excinfo.value)
