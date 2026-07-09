import sys
import os
import unittest.mock as mock
import pytest

# Add pipelines/ussd/src and pipelines/underwriting to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "ussd"))
sys.path.append(os.path.join(root_dir, "underwriting"))

from src.ussd_handler import app, sessions  # noqa: E402
from src.scoring_client import ScoringServiceUnavailable  # noqa: E402
from schemas.contracts import RiskScoreResponse, ShapFactor  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
    # Clear sessions after each test
    sessions.clear()


def test_health_endpoint(client):
    """Tests GET /health response."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json == {"status": "ok", "app": "bimalink_ussd"}


def test_ussd_callback_session_lifecycle(client):
    """Tests full step-by-step user traversal and menu presentation."""
    # 1. Start Session (Empty input)
    payload = {
        "sessionId": "lifecycle_session_id",
        "phoneNumber": "+254712345678",
        "serviceCode": "*384*5#",
        "text": ""
    }
    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert res.text.startswith("CON Welcome to BimaLink. Choose Language:")
    assert "lifecycle_session_id" in sessions

    # 2. Select Language
    payload["text"] = "1"
    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert res.text.startswith("CON Select your occupation:")
    assert sessions["lifecycle_session_id"]["validated_step_count"] == 1

    # 3. Enter Invalid Input (attempting step 1: occupation)
    payload["text"] = "1*9"
    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert "Invalid option" in res.text
    # Step remains at 1, retries incremented
    assert sessions["lifecycle_session_id"]["validated_step_count"] == 1
    assert sessions["lifecycle_session_id"]["retry_count"] == 1

    # 4. Correct the input (select Boda Boda)
    payload["text"] = "1*9*1"
    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert res.text.startswith("CON Select your average daily income band:")
    assert sessions["lifecycle_session_id"]["validated_step_count"] == 2


@mock.patch("src.ussd_handler.get_risk_score")
def test_ussd_callback_completion_success(mock_get_score, client):
    """Tests happy-path quote completion and session cache cleanup."""
    # Mock successful response
    mock_get_score.return_value = RiskScoreResponse(
        risk_probability=0.25,
        risk_tier="Medium",
        premium_quote_kes=300,
        default_probability=0.25,
        shap_top_factors=[
            ShapFactor(
                feature="momo_txn_regularity_score",
                direction="decreases_risk",
                plain_language="Consistent money usage"
            ),
            ShapFactor(
                feature="occupation",
                direction="increases_risk",
                plain_language="Boda rider risk tier"
            )
        ],
        data_disclosure="Consent details"
    )

    # Initialize session in cache (simulating steps 0, 1, 2 already validated)
    session_id = "success_session_id"
    sessions[session_id] = {
        "session_start_time": 1000.0,
        "validated_step_count": 3,
        "processed_input_count": 3,
        "collected_answers": {
            "language": "en",
            "occupation": "boda_rider",
            "avg_daily_income_band": "500_to_1500"
        },
        "retry_count": 0,
        "last_error": None
    }

    # Submit final step choice ("2" for 1_to_3 years)
    payload = {
        "sessionId": session_id,
        "phoneNumber": "+254712345678",
        "serviceCode": "*384*5#",
        "text": "1*1*2*2"
    }

    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert res.text.startswith("END Your tier: Medium. Premium: KES 300/month.")
    assert "Consistent money usage" in res.text
    assert "Reply *384*5# to enroll." in res.text

    # Verify session was cleaned up
    assert session_id not in sessions


@mock.patch("src.ussd_handler.get_risk_score")
def test_ussd_callback_completion_outage_cleanup(mock_get_score, client):
    """Tests backend outage handling and confirms session cleanup is executed on failure branch."""
    # Mock outage
    mock_get_score.side_effect = ScoringServiceUnavailable("Backend down")

    session_id = "fail_session_id"
    sessions[session_id] = {
        "session_start_time": 1000.0,
        "validated_step_count": 3,
        "processed_input_count": 3,
        "collected_answers": {
            "language": "en",
            "occupation": "boda_rider",
            "avg_daily_income_band": "500_to_1500"
        },
        "retry_count": 0,
        "last_error": None
    }

    payload = {
        "sessionId": session_id,
        "phoneNumber": "+254712345678",
        "serviceCode": "*384*5#",
        "text": "1*1*2*2"
    }

    res = client.post("/ussd", data=payload)
    assert res.status_code == 200
    assert "Service temporarily unavailable" in res.text

    # Verify session was cleaned up despite backend failure
    assert session_id not in sessions
