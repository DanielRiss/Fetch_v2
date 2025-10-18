import requests

SAMPLES = [
    "What is 1+1?",
    "What is 2+3?",
    "What is 5×6?",
    "What is 10−4?",
    "Compute 7+8.",
]

for port in (8004, 8005):
    print(f"\n— Verifier on port {port} —")
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/predict",
            json={"texts": SAMPLES}
        )
        resp.raise_for_status()
        vals = resp.json()["values"]
        print("Raw values:", vals)
    except Exception as e:
        print("Error:", e)
