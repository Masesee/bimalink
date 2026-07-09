import sys
import os
import time
import hashlib

# Add pipelines/ussd/src to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "ussd"))
sys.path.append(os.path.join(root_dir, "underwriting"))

from src.session_state import parse_session_state, build_user_profile  # noqa: E402
from schemas.contracts import UserProfile  # noqa: E402


def test_parse_session_state_happy_path():
    """Tests normal step-by-step progress through the USSD questions."""
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
    assert state.is_complete
    assert state.error is None
    assert state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500",
        "years_active": "over_3"
    }
    assert state.processed_input_count == 4


def test_parse_session_state_invalid_input_and_recovery():
    """Tests that a bad input sets an error but doesn't block recovery on the next turn."""
    # Step 0 (Language) and Step 1 (Occupation) successfully validated
    cached_step = 2
    processed_count = 2
    answers = {"language": "en", "occupation": "market_trader"}
    retries = 0

    # User enters "9" (invalid daily income choice)
    bad_state = parse_session_state(
        text="1*2*9",
        validated_step_count=cached_step,
        processed_input_count=processed_count,
        collected_answers=answers,
        retry_count=retries
    )
    assert bad_state.current_step == 2
    assert bad_state.error is not None
    assert bad_state.retry_count == 1
    assert bad_state.processed_input_count == 3
    assert not bad_state.is_complete

    # User retries with a valid choice "1"
    recovery_state = parse_session_state(
        text="1*2*9*1",
        validated_step_count=bad_state.current_step,
        processed_input_count=bad_state.processed_input_count,
        collected_answers=answers,
        retry_count=bad_state.retry_count
    )
    assert recovery_state.current_step == 3
    assert recovery_state.error is None
    assert recovery_state.retry_count == 1
    assert recovery_state.processed_input_count == 4
    assert recovery_state.collected_answers == {
        "language": "en",
        "occupation": "market_trader",
        "avg_daily_income_band": "under_500"
    }


def test_build_user_profile():
    """Tests constructing the UserProfile model and validates the deterministic seed mapping."""
    phone = "+254700000000"
    answers = {
        "language": "en",
        "occupation": "boda_rider",
        "avg_daily_income_band": "500_to_1500",
        "years_active": "1_to_3"
    }
    start_time = time.time() - 30.0

    profile1 = build_user_profile(
        collected_answers=answers,
        phone_number=phone,
        session_start_time=start_time,
        retry_count=1
    )

    assert isinstance(profile1, UserProfile)
    assert profile1.phone_number_hash == hashlib.sha256(phone.encode()).hexdigest()
    assert profile1.self_reported.occupation == "boda_rider"
    assert profile1.self_reported.avg_daily_income_band == "500_to_1500"
    assert profile1.self_reported.years_active == "1_to_3"

    # Verify live behavioral metrics
    assert profile1.live_behavioral.retry_count == 1
    assert profile1.live_behavioral.ussd_session_duration_sec >= 30.0
    # completion rate: 4 validated steps / (4 + 1 retry) = 0.8
    assert profile1.live_behavioral.menu_completion_rate == 0.8

    # Verify deterministic synthetic history on matching phone number
    profile2 = build_user_profile(
        collected_answers=answers,
        phone_number=phone,
        session_start_time=start_time,
        retry_count=1
    )
    hist1 = profile1.synthetic_historical
    hist2 = profile2.synthetic_historical
    assert hist1.momo_txn_frequency == hist2.momo_txn_frequency
    assert hist1.momo_txn_regularity_score == hist2.momo_txn_regularity_score
    assert hist1.sacco_contribution_flag == hist2.sacco_contribution_flag

    # Verify different phone number produces a different historical profile
    other_phone = "+254711111111"
    profile_other = build_user_profile(
        collected_answers=answers,
        phone_number=other_phone,
        session_start_time=start_time,
        retry_count=1
    )
    # The odds of two random seeds yielding exact floating point equality across multiple features is practically zero
    assert profile1.synthetic_historical.momo_txn_frequency != profile_other.synthetic_historical.momo_txn_frequency
