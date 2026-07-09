import sys
import argparse
import requests


def run_simulator(phone_number, server_url):
    print("====================================================")
    print("        DiaBima USSD Onboarding CLI Simulator       ")
    print("====================================================")
    print(f"Target Server : {server_url}")
    print(f"Phone Number  : {phone_number}")
    print("----------------------------------------------------\n")

    session_id = "sim_session_hash_987654"
    text = ""

    # Session loop
    while True:
        payload = {
            "sessionId": session_id,
            "phoneNumber": phone_number,
            "serviceCode": "*384*5#",
            "text": text
        }

        try:
            response = requests.post(server_url, data=payload, timeout=5.0)
        except requests.ConnectionError:
            print(f"[ERROR] Could not connect to USSD server at {server_url}.")
            print("Please ensure your ussd_handler.py application is running on port 5000.")
            sys.exit(1)
        except requests.RequestException as e:
            print(f"[ERROR] HTTP request failed: {e}")
            sys.exit(1)

        if response.status_code != 200:
            print(f"[ERROR] Server returned HTTP {response.status_code}: {response.text}")
            sys.exit(1)

        raw_content = response.text
        print("--- USSD SCREEN ---")
        print(raw_content)
        print("-------------------")

        # Check if session is ended by the callback
        if raw_content.startswith("END"):
            print("\n[LOG] USSD Session ended by server.")
            break

        # Read user input
        try:
            user_choice = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[LOG] Simulator session aborted.")
            break

        # Accumulate input
        if text == "":
            text = user_choice
        else:
            text = f"{text}*{user_choice}"
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulates a live USSD session.")
    parser.add_argument(
        "--phone",
        default="+254712345678",
        help="Phone number to send in USSD callback (default: +254712345678)"
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000/ussd",
        help="USSD handler endpoint URL (default: http://127.0.0.1:5000/ussd)"
    )
    args = parser.parse_args()

    run_simulator(args.phone, args.url)
