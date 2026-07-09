import os
import sys
import time
from flask import Flask, request

# Add pipelines/ussd/src and pipelines/underwriting to python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "ussd"))
sys.path.append(os.path.join(root_dir, "underwriting"))

from src.session_state import parse_session_state, get_prompt_for_state, build_user_profile  # noqa: E402
from src.scoring_client import get_risk_score, ScoringServiceUnavailable  # noqa: E402

app = Flask(__name__)

# In-memory session cache.
# KEY: sessionId (str)
# VALUE: dict of:
#   - session_start_time: float (epoch time)
#   - validated_step_count: int (0 to 4)
#   - processed_input_count: int (total valid + invalid inputs processed)
#   - collected_answers: dict (step_name -> validated literal string)
#   - retry_count: int (total number of validation failures)
#   - last_error: Optional[str] (stored error message from latest invalid input)
# LIMITATION: This session cache is in-memory only. Re-starting the process will wipe out active
# session states. This is acceptable for hackathon demos but must not be used in production.
sessions: dict[str, dict] = {}


@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "app": "bimalink_ussd"}


@app.route("/ussd", methods=["POST"])
def ussd_callback():
    # Extract form variables matching Africa's Talking USSD protocol
    session_id = request.form.get("sessionId")
    phone_number = request.form.get("phoneNumber")
    service_code = request.form.get("serviceCode", "*384*5#")
    text = request.form.get("text", "")

    if not session_id or not phone_number:
        return "END Bad Request: Missing sessionId or phoneNumber", 400

    # 1. Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            "session_start_time": time.time(),
            "validated_step_count": 0,
            "processed_input_count": 0,
            "collected_answers": {},
            "retry_count": 0,
            "last_error": None
        }

    session = sessions[session_id]

    # 2. Parse the session state based on accumulated text and cache records
    state = parse_session_state(
        text=text,
        validated_step_count=session["validated_step_count"],
        processed_input_count=session["processed_input_count"],
        collected_answers=session["collected_answers"],
        retry_count=session["retry_count"]
    )

    # 3. Update the session cache values
    session["processed_input_count"] = state.processed_input_count

    if state.error:
        session["retry_count"] = state.retry_count
        session["last_error"] = state.error
    else:
        session["validated_step_count"] = state.current_step
        session["collected_answers"] = state.collected_answers
        session["last_error"] = None

    # 4. Handle menu flow branching
    if not state.is_complete:
        # Prompt user for the next input, incorporating any error message
        response_text = get_prompt_for_state(state)
        return response_text, 200
    else:
        # On completion:
        # A. Build the UserProfile (Telemetry calculations are performed here)
        user_profile = build_user_profile(
            collected_answers=session["collected_answers"],
            phone_number=phone_number,
            session_start_time=session["session_start_time"],
            retry_count=session["retry_count"]
        )

        try:
            # B. Call the scoring microservice
            response = get_risk_score(user_profile)

            # Format up to 2 SHAP explanation factors in plain language
            shap_1 = (
                response.shap_top_factors[0].plain_language
                if len(response.shap_top_factors) > 0
                else "Profile attributes are within normal limits."
            )
            shap_2 = (
                response.shap_top_factors[1].plain_language
                if len(response.shap_top_factors) > 1
                else "No other significant risk drivers detected."
            )

            response_text = (
                f"END Your tier: {response.risk_tier}. Premium: KES {response.premium_quote_kes}/month.\n"
                f"Why:\n"
                f"- {shap_1}\n"
                f"- {shap_2}\n"
                f"Reply {service_code} to enroll."
            )
        except ScoringServiceUnavailable as e:
            # Handle backend outages gracefully, returning an END message rather than a raw crash
            print(f"[ERROR] Scoring service exception: {e}")
            response_text = "END Service temporarily unavailable, please try again shortly."
        finally:
            # C. Clean up the session cache entry in both success and error terminal branches to prevent memory leaks
            if session_id in sessions:
                del sessions[session_id]

        return response_text, 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
