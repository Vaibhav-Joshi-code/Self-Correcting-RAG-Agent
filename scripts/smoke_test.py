"""Live corpus benchmark against a running API. Makes real provider calls."""
import argparse
import json
import os
from pathlib import Path
import httpx

CASES = [
    {"name": "annual_leave", "question": "How many annual leave days do employees receive?",
     "expected_accepted": True, "fact_groups": [["24", "twenty-four"]]},
    {"name": "carry_forward", "question": "How many leave days can be carried forward, and by when must they be used?",
     "expected_accepted": True, "fact_groups": [["five", "5"], ["march 31", "31 march"]]},
    {"name": "travel", "question": "What are the domestic travel meal and hotel limits, and when must receipts be submitted?",
     "expected_accepted": True, "fact_groups": [["800"], ["3,000", "3000"], ["30", "thirty"]]},
    {"name": "security", "question": "How soon must I report a lost company laptop, and to whom?",
     "expected_accepted": True, "fact_groups": [["one hour", "1 hour"], ["it security", "it-security"]]},
    {"name": "unsupported_revenue", "question": "What was ChaiTech's revenue in 2025?",
     "expected_accepted": False, "fact_groups": []},
    {"name": "unsupported_international", "question": "What is the hotel reimbursement limit for international travel?",
     "expected_accepted": False, "fact_groups": []},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/live_results.json"))
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    rows = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=args.timeout,
                      headers={"X-API-Key": os.getenv("API_KEY", "")}) as client:
        for case in CASES:
            try:
                response = client.post("/ask", json={"question": case["question"]})
                response.raise_for_status()
                result = response.json()
                facts = all(any(word in result["answer"].lower() for word in group) for group in case["fact_groups"])
                passed = result["accepted"] == case["expected_accepted"] and facts
                rows.append({"case": case, "passed": passed, "result": result})
                print(f"{case['name']}: {'PASS' if passed else 'FAIL'}; attempts={result['attempts']}; status={result['status']}")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                rows.append({"case": case, "passed": False, "error_type": type(exc).__name__})
                print(f"{case['name']}: ERROR ({type(exc).__name__})")
    successes = sum(row["passed"] for row in rows)
    valid = [row["result"] for row in rows if "result" in row]
    report = {"mode": "live_api", "cases": len(rows), "passed": successes,
              "pass_rate": successes / len(rows),
              "mean_attempts": sum(r["attempts"] for r in valid) / len(valid) if valid else None,
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Passed {successes}/{len(rows)}. Evidence: {args.output}")
    raise SystemExit(0 if successes == len(rows) else 1)


if __name__ == "__main__":
    main()
