"""CLI for a running API: python client.py 'Your question'."""
import argparse
import json
import os
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    question = " ".join(args.question).strip() or input("Question: ").strip()
    if not question:
        parser.error("A question is required")
    headers = {"X-API-Key": os.getenv("API_KEY", "")}
    try:
        response = httpx.post(args.url.rstrip("/") + "/ask", json={"question": question},
                              headers=headers, timeout=args.timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"API error {exc.response.status_code}: {exc.response.text}")
        raise SystemExit(1) from None
    except httpx.HTTPError:
        print("Could not reach the API, or the client timeout expired.")
        raise SystemExit(1) from None
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
