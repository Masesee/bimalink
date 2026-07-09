import sys
import os
import time
import hashlib

# Add pipelines/ussd/src and underwriting to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "ussd"))
sys.path.append(os.path.join(root_dir, "underwriting"))

from src.session_state import (  # noqa: E402
    parse_session_state, build_user_profile,
    map_momo_volume_bucket_to_frequency,
    map_years_active_to_regularity_proxy
)
from schemas.contracts import UserProfile  # noqa: E402


def test_mapping_functions():
    """Direct unit tests for both proxy mapping functions with all valid inputs checked."""
    # 1. Weekly volume bucket mapping to frequency
    assert map_momo_volume_bucket_to_frequency("under_2000") == 3.0
    assert map_momo_volume_bucket_to_frequency("2000_to_10000") == 8.0
    assert map_momo_volume_bucket_to_frequency("over_10000") == 15.0

    # 2. Years active mapping to cash flow regularity proxy
    assert map_years_active_to_regularity_proxy("under_1") == 0.30
    assert map_years_active_to_regularity_proxy("1_to_3") == 0.55
    assert map_years_active_to_regularity_proxy("over_3") == 0.75


def test_parse_session_state_happy_path():
    """Tests normal step-by-step progress through the new 5-step USSD questions."""
    # Startup (no inputs)
    state = parse_session_state(
        text="",
        validated_step_count=0,
        processed_input_count=0,
        collected_answers={},
        retry_count=0
    )
    assert state.current_step == 0
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {}

    # Step 0 (Language choice: 1 -> en)
    state = parse_session_state(
        text="1",
        validated_step_count=0,
        processed_input_count=0,
        collected_answers={},
        retry_count=0
    )
    assert state.current_step == 1
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {"language": "en"}
    assert state.processed_input_count == 1

    # Step 1 (Occupation choice: 2 -> market_trader)
    state = parse_session_state(
        text="1*2",
        validated_step_count=1,
        processed_input_count=1,
        collected_answers={"language": "en"},
        retry_count=0
    )
    assert state.current_step == 2
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {"language": "en", "occupation": "market_trader"}
    assert state.processed_input_count == 2

    # Step 2 (Income band: 1 -> under_500)
    state = parse_session_state(
        text="1*2*1",
        validated_step_count=2,
        processed_input_count=2,
        collected_answers={"language": "en", "occupation": "market_trader"},
        retry_count=0
    )
    assert state.current_step == 3
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500"
    }
    assert state.processed_input_count == 3

    # Step 3 (Years active: 3 -> over_3)
    state = parse_session_state(
        text="1*2*1*3",
        validated_step_count=3,
        processed_input_count=3,
        collected_answers={
            "language": "en",
            "occupation": "market_trader",
            "avg_daily_income_band": "under_500"
        },
        retry_count=0
    )
    assert state.current_step == 4
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3"
    }
    assert state.processed_input_count == 4

    # Step 4 (SACCO/chama member: 1 -> Yes / True)
    state = parse_session_state(
        text="1*2*1*3*1",
        validated_step_count=4,
        processed_input_count=4,
        collected_answers={
            "language": "en",
            "occupation": "market_trader",
            "avg_daily_income_band": "under_500",
            "years_active": "over_3"
        },
        retry_count=0
    )
    assert state.current_step == 5
    assert not state.is_complete
    assert state.error is None
    assert state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3",
        "sacco_contribution_flag": True
    }
    assert state.processed_input_count == 5

    # Step 5 (Weekly volume bucket: 2 -> 2000_to_10000)
    state = parse_session_state(
        text="1*2*1*3*1*2",
        validated_step_count=5,
        processed_input_count=5,
        collected_answers={
            "language": "en",
            "occupation": "market_trader",
            "avg_daily_income_band": "under_500",
            "years_active": "over_3",
            "sacco_contribution_flag": True
        },
        retry_count=0
    )
    assert state.current_step == 6
    assert state.is_complete
    assert state.error is None
    assert state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3",
        "sacco_contribution_flag": True,
        "momo_volume_bucket": "2000_to_10000"
    }
    assert state.processed_input_count == 6


def test_parse_session_state_invalid_input_and_recovery():
    """Tests that a bad input sets an error but doesn't block recovery on the next turn."""
    # Steps 0, 1, 2, 3 successfully validated
    cached_step = 4
    processed_count = 4
    answers = {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3"
    }
    retries = 0

    # User enters "9" (invalid SACCO choice)
    bad_state = parse_session_state(
        text="1*2*1*3*9",
        validated_step_count=cached_step,
        processed_input_count=processed_count,
        collected_answers=answers,
        retry_count=retries
    )
    assert bad_state.current_step == 4
    assert bad_state.error is not None
    assert bad_state.retry_count == 1
    assert bad_state.processed_input_count == 5
    assert not bad_state.is_complete

    # User retries with a valid choice "1" (Yes)
    recovery_state = parse_session_state(
        text="1*2*1*3*9*1",
        validated_step_count=bad_state.current_step,
        processed_input_count=bad_state.processed_input_count,
        collected_answers=answers,
        retry_count=bad_state.retry_count
    )
    assert recovery_state.current_step == 5
    assert recovery_state.error is None
    assert recovery_state.retry_count == 1
    assert recovery_state.processed_input_count == 6
    assert recovery_state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3",
        "sacco_contribution_flag": True
    }


def test_build_user_profile():
    """Tests constructing the UserProfile model and validates the deterministic seed mapping."""
    phone = "+254700000000"
    answers = {
        "language": "en",
        "occupation": "boda_rider",
        "avg_daily_income_band": "500_to_1500",
        "years_active": "over_3",
        "sacco_contribution_flag": True,
        "momo_volume_bucket": "2000_to_10000"
    }
    start_time = time.time() - 30.0

    profile = build_user_profile(
        collected_answers=answers,
        phone_number=phone,
        session_start_time=start_time,
        retry_count=1
    )

    assert isinstance(profile, UserProfile)
    assert profile.phone_number_hash == hashlib.sha256(phone.encode()).hexdigest()
    assert profile.data_provenance == "self_reported"
    assert profile.self_reported.occupation == "boda_rider"
    assert profile.self_reported.avg_daily_income_band == "500_to_1500"
    assert profile.self_reported.years_active == "over_3"

    # Verify derived proxy scores
    assert profile.synthetic_historical.momo_txn_frequency == 8.0  # From "2000_to_10000"
    assert profile.synthetic_historical.momo_txn_regularity_score == 0.75  # From "over_3" years_active
    assert profile.synthetic_historical.sacco_contribution_flag is True

    # Verify airtime feature is absent from the resulting model
    assert not hasattr(profile.synthetic_historical, "airtime_topup_cadence")

    # Verify live behavioral metrics
    assert profile.live_behavioral.retry_count == 1
    assert profile.live_behavioral.ussd_session_duration_sec >= 30.0
    # completion rate: 6 validated steps / (6 + 1 retry) = 6/7 = 0.857...
    assert abs(profile.live_behavioral.menu_completion_rate - (6.0 / 7.0)) < 1e-4
