# DiaBima USSD Onboarding Service

The `ussd/` service is a lightweight, stateless Flask application that acts as the client-facing gateway for informal sector workers using basic feature phones. It provides a guided onboarding menu, validates answers in real-time, builds user profiles, and communicates with the core underwriting scoring service to fetch micro-insurance quotes.

## Architectural Boundaries

This layer is decoupled from the underwriting system. It does not import any model or pipeline files directly; instead, it communicates with the FastAPI scoring service over HTTP.

```
       [ Africa's Talking USSD Gateway ]
                     │ (HTTP POST)
                     ▼
       [ DiaBima USSD Layer (Flask:5000) ]
                     │ (HTTP POST JSON payload)
                     ▼
  [ Underwriting Scoring Service (FastAPI:8000) ]
```

## Running the Services

### 1. Run Underwriting Scoring Backend
Ensure the FastAPI scoring service is running on port 8000:
```bash
python pipelines/underwriting/src/scoring_service.py
```

### 2. Run USSD Gateway Callback
Start the USSD gateway Flask application on port 5000:
```bash
python pipelines/ussd/src/ussd_handler.py
```

### 3. Run the Session Simulator CLI
Simulate a live USSD session from your terminal:
```bash
python pipelines/ussd/src/simulate_session.py --phone +254712345678
```

---

## Technical Details

### 1. Incremental Suffix Parser
USSD gateways pass the full accumulated path of inputs in the `text` field (e.g. `"1*2*9*1"`). To prevent invalid inputs from bricking a session permanently, our parser uses the session cache to track how many inputs have been processed. On each request, it isolates and validates only the newly appended suffix.

### 2. Live Behavioral Telemetry vs Synthetic History
* **Live Behavioral Telemetry (Real)**: Session duration, retry counts, completion rates, and wall-clock hour of day are computed dynamically from the user's actual live session inputs.
* **Synthetic History (Deterministic)**: First-time callers have no mobile money transactional history. A deterministic profile is generated using a stable SHA-256 phone number hash as a seed. This ensures that the same phone number always yields identical historical metrics across restarts.

### 3. Graceful Outage Handling
If the underwriting scoring backend is offline or timed out, the USSD layer catches `ScoringServiceUnavailable` and returns a friendly `"END Service temporarily unavailable..."` message to the user, cleaning up the session cache immediately to prevent memory leaks.

---

## Testing

Run unit and integration tests:
```bash
python -m pytest pipelines/ussd/
```
