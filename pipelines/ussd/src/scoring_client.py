import os
import sys
import requests

# Add pipelines/underwriting to python path to allow importing contracts
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, "underwriting"))

from schemas.contracts import UserProfile, RiskScoreResponse  # noqa: E402


class ScoringServiceUnavailable(Exception):
    """Raised when the backend scoring service is unreachable, timed out, or returns an error."""
    pass


def get_risk_score(profile: UserProfile) -> RiskScoreResponse:
    """
    Sends the serialized UserProfile to the scoring service over HTTP.
    Raises ScoringServiceUnavailable on timeouts, connection failures, or bad responses.
    """
    scoring_service_url = os.getenv("SCORING_SERVICE_URL", "http://localhost:8000/score")

    try:
        # Pydantic v2 serialization to dict, then POST as JSON
        payload = profile.model_dump()

        response = requests.post(
            scoring_service_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=2.0
        )

        # Check for non-200 responses
        if response.status_code != 200:
            raise ScoringServiceUnavailable(
                f"Scoring service returned non-200 status code: {response.status_code}"
            )

        # Parse directly into the boundary contract Pydantic model
        data = response.json()
        return RiskScoreResponse.model_validate(data)

    except requests.Timeout as e:
        raise ScoringServiceUnavailable("Scoring service request timed out.") from e
    except requests.RequestException as e:
        raise ScoringServiceUnavailable("Scoring service is unreachable.") from e
    except Exception as e:
        raise ScoringServiceUnavailable(f"Error parsing scoring service response: {e}") from e
