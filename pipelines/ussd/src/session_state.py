"""
DiaBima USSD Session State Module.

This module parses incoming accumulated USSD input paths (separated by "*") using
incremental validation backed by an in-memory session cache. It separates real session
behavioral metrics (live_behavioral) from synthetic historical features (synthetic_historical).

Telemetry:
- live_behavioral fields (duration, completions, retries, time of day) are REAL metrics.
- synthetic_historical fields are synthetic, but generated DETERMINISTICALLY using a
  stable SHA-256 phone number hash seed to ensure consistency across process restarts.
"""

import sys
import os
import time
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Add pipelines/underwriting to python path to allow importing its contracts and generator
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
underwriting_dir = os.path.join(root_dir, "underwriting")
sys.path.append(underwriting_dir)

from schemas.contracts import UserProfile, SelfReported, SyntheticHistorical, LiveBehavioral  # noqa: E402
from src.generate_synthetic_profiles import generate_single_synthetic_profile  # noqa: E402

STEP_NAMES = ["language", "occupation", "avg_daily_income_band", "years_active"]

STEP_OPTIONS = [
    {"1": "en", "2": "sw"},
    {"1": "boda_rider", "2": "market_trader", "3": "other"},
    {"1": "under_500", "2": "500_to_1500", "3": "over_1500"},
    {"1": "under_1", "2": "1_to_3", "3": "over_3"}
]

STEP_PROMPTS = [
    "Welcome to BimaLink. Choose Language:\n1. English\n2. Kiswahili",
    "Select your occupation:\n1. Boda Boda Rider\n2. Market Trader\n3. Other Informal Worker",
    "Select your average daily income band:\n1. Under KES 500\n2. KES 500 to 1,500\n3. Over KES 1,500",
    "How many years have you been active in this occupation:\n1. Under 1 year\n2. 1 to 3 years\n3. Over 3 years"
]


@dataclass
class SessionState:
    current_step: int
    is_complete: bool
    collected_answers: dict
    processed_input_count: int
    retry_count: int
    error: Optional[str] = None


def parse_session_state(
    text: str,
    validated_step_count: int,
    processed_input_count: int,
    collected_answers: dict,
    retry_count: int
) -> SessionState:
    """
    Parses accumulated USSD inputs (separated by "*") and returns the updated state.
    Validates only unconsumed inputs that have arrived since processed_input_count.
    """
    # Split text on "*" and filter empty entries
    inputs = [x for x in text.split("*") if x != ""]

    if len(inputs) > processed_input_count:
        new_inputs = inputs[processed_input_count:]

        updated_answers = dict(collected_answers)
        updated_step = validated_step_count
        updated_processed = processed_input_count
        updated_retries = retry_count
        error = None

        for new_input in new_inputs:
            if updated_step >= 4:
                updated_processed += 1
                continue

            step_name = STEP_NAMES[updated_step]
            step_options = STEP_OPTIONS[updated_step]

            if new_input in step_options:
                updated_answers[step_name] = step_options[new_input]
                updated_step += 1
                error = None
            else:
                error = "Invalid option. Please enter a valid number."
                updated_retries += 1
                updated_processed += 1
                break

            updated_processed += 1

        return SessionState(
            current_step=updated_step,
            is_complete=(updated_step >= 4),
            collected_answers=updated_answers,
            processed_input_count=updated_processed,
            retry_count=updated_retries,
            error=error
        )
    else:
        # No new inputs (initial screen or duplicate gateway request)
        return SessionState(
            current_step=validated_step_count,
            is_complete=(validated_step_count >= 4),
            collected_answers=collected_answers,
            processed_input_count=processed_input_count,
            retry_count=retry_count,
            error=None
        )


def get_prompt_for_state(state: SessionState) -> str:
    """
    Generates the USSD response string with CON prefix based on current state.
    Re-shows prompt if validation error occurred.
    """
    if state.error:
        base_prompt = STEP_PROMPTS[state.current_step]
        return f"CON {state.error}\n{base_prompt}"

    if state.current_step < 4:
        return f"CON {STEP_PROMPTS[state.current_step]}"

    return "END Processing your quote..."


def build_user_profile(
    collected_answers: dict,
    phone_number: str,
    session_start_time: float,
    retry_count: int
) -> UserProfile:
    """
    Builds the UserProfile object.
    Uses SHA-256 for deterministic synthetic historical profiling
    and time-based metrics for real session behavioral tracking.
    """
    # 1. Build SelfReported section, ignoring presentation-only language selection
    self_reported = SelfReported(
        occupation=collected_answers["occupation"],
        avg_daily_income_band=collected_answers["avg_daily_income_band"],
        years_active=collected_answers["years_active"]
    )

    # 2. Build SyntheticHistorical section using stable SHA-256 seed from phone number
    hash_hex = hashlib.sha256(phone_number.encode()).hexdigest()
    seed = int(hash_hex, 16) % (2**32)
    rng = np.random.default_rng(seed)

    hist_data = generate_single_synthetic_profile(rng)

    synthetic_historical = SyntheticHistorical(
        momo_txn_frequency=hist_data["momo_txn_frequency"],
        momo_txn_regularity_score=hist_data["momo_txn_regularity_score"],
        airtime_topup_cadence=hist_data["airtime_topup_cadence"],
        sacco_contribution_flag=hist_data["sacco_contribution_flag"]
    )

    # 3. Build LiveBehavioral section using real telemetry metrics
    duration = max(float(time.time() - session_start_time), 0.1)

    # Calculate completion rate: validated steps (including language = 4 steps) / total attempts processed
    # Avoid zero division
    total_attempts = 4 + retry_count
    completion_rate = float(4.0 / total_attempts)

    # Hour of day from current local system wall clock
    hour_of_day = int(time.localtime().tm_hour)

    live_behavioral = LiveBehavioral(
        ussd_session_duration_sec=duration,
        menu_completion_rate=completion_rate,
        session_hour_of_day=hour_of_day,
        retry_count=retry_count
    )

    return UserProfile(
        phone_number_hash=hash_hex,
        self_reported=self_reported,
        synthetic_historical=synthetic_historical,
        live_behavioral=live_behavioral
    )
