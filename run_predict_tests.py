#!/usr/bin/env python3
"""
Run /predict tests for a list of tokens and save Payload + Response to JSON files.
Also runs GET /user/profile/{wallet_address} for each wallet in meme_wallets.txt.

Usage:
  python run_predict_tests.py                          # uses meme_mints.txt, default base URL
  python run_predict_tests.py --test-type meme         # same
  python run_predict_tests.py --test-type perps       # uses perps_symbols.txt
  python run_predict_tests.py --input-file testCases/inputs/meme_mints.txt
  python run_predict_tests.py --base-url http://localhost:8800 --output testCases/results/out.json

Output:
  - testCases/results/predict_<test_type>_<timestamp>.json (predict payload + response per token)
  - testCases/results/profile_meme_<timestamp>.json (profile payload + response per wallet from meme_wallets.txt)
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests

# Default paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent
TEST_CASES_DIR = REPO_ROOT / "testCases"
INPUTS_DIR = TEST_CASES_DIR / "inputs"
RESULTS_DIR = TEST_CASES_DIR / "results"

# Test-type config: (default input file, token_type for /predict)
TEST_TYPE_CONFIG = {
    "meme": ("meme_mints.txt", "meme"),
    "perps": ("perps_symbols.txt", "perps"),
    "meme_wallets": ("meme_wallets.txt", "meme"),  # for future /predict/with-user etc.
}


def load_input_list(path: Path) -> List[str]:
    """Load lines from file; strip whitespace, skip empty and # comments."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


def run_one(base_url: str, token_address: str, token_type: str, timeout: int) -> dict:
    """POST /predict for one token; return { payload, response }."""
    url = f"{base_url.rstrip('/')}/predict"
    payload = {
        "token_address": token_address,
        "token_type": token_type,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        response = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"_raw": r.text}
        if not r.ok:
            response["_status_code"] = r.status_code
            response["_error"] = r.reason or "Request failed"
    except requests.RequestException as e:
        response = {"_error": str(e)}
    except json.JSONDecodeError as e:
        response = {"_error": f"JSON decode: {e}", "_raw": getattr(e, "doc", "")}
    return {"payload": payload, "response": response}


def run_profile_one(
    base_url: str,
    wallet_address: str,
    token_type: str,
    timeout: int,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict:
    """GET /user/profile/{wallet_address} for one wallet; return { payload, response }."""
    url = f"{base_url.rstrip('/')}/user/profile/{wallet_address}"
    payload = {
        "wallet_address": wallet_address,
        "token_type": token_type,
    }
    params = {"token_type": token_type}
    if from_date is not None and to_date is not None:
        params["from_date"] = from_date
        params["to_date"] = to_date
        payload["from_date"] = from_date
        payload["to_date"] = to_date
    try:
        r = requests.get(url, params=params, timeout=timeout)
        response = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"_raw": r.text}
        if not r.ok:
            response["_status_code"] = r.status_code
            response["_error"] = r.reason or "Request failed"
    except requests.RequestException as e:
        response = {"_error": str(e)}
    except json.JSONDecodeError as e:
        response = {"_error": f"JSON decode: {e}", "_raw": getattr(e, "doc", "")}
    return {"payload": payload, "response": response}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run /predict for a list of tokens and save Payload + Response to one JSON file."
    )
    parser.add_argument(
        "--test-type",
        choices=list(TEST_TYPE_CONFIG),
        default="meme",
        help="Test type: meme (default), perps, or meme_wallets (extensible).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input file (one token/symbol per line). Default: testCases/inputs/<test_type>.txt",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8800",
        help="API base URL (default: http://localhost:8800).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: testCases/results/predict_<test_type>_<timestamp>.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120).",
    )
    args = parser.parse_args()

    default_input, token_type = TEST_TYPE_CONFIG[args.test_type]
    input_path = args.input_file or (INPUTS_DIR / default_input)
    input_path = input_path if input_path.is_absolute() else (REPO_ROOT / input_path)

    items = load_input_list(input_path)
    if not items:
        print(f"No items in {input_path}", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {args.test_type}: {item[:20]}...")
        results.append(run_one(args.base_url, item, token_type, args.timeout))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or (RESULTS_DIR / f"predict_{args.test_type}_{timestamp}.json")
    out_path = out_path if out_path.is_absolute() else (REPO_ROOT / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "test_type": args.test_type,
        "base_url": args.base_url,
        "timestamp": timestamp,
        "count": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {out_path}")

    # Also run /user/profile/{wallet_address} tests for meme_wallets.txt
    profile_input_path = INPUTS_DIR / "meme_wallets.txt"
    profile_items = load_input_list(profile_input_path)
    if profile_items:
        print(f"\nRunning profile tests for {len(profile_items)} wallet(s) from {profile_input_path}")
        # Use same 90-day window as server default so profile tests align with manual API calls
        to_dt = datetime.utcnow()
        from_dt = to_dt - timedelta(days=90)
        profile_from = from_dt.strftime("%Y-%m-%dT00:00:00Z")
        profile_to = to_dt.strftime("%Y-%m-%dT23:59:59Z")
        profile_results = []
        for i, wallet in enumerate(profile_items, 1):
            print(f"  [{i}/{len(profile_items)}] profile: {wallet[:20]}...")
            profile_results.append(
                run_profile_one(
                    args.base_url,
                    wallet,
                    "meme",
                    args.timeout,
                    from_date=profile_from,
                    to_date=profile_to,
                )
            )
        profile_out_path = RESULTS_DIR / f"profile_meme_{timestamp}.json"
        profile_out = {
            "test_type": "profile_meme",
            "base_url": args.base_url,
            "timestamp": timestamp,
            "count": len(profile_results),
            "results": profile_results,
        }
        profile_out_path.write_text(json.dumps(profile_out, indent=2), encoding="utf-8")
        print(f"Wrote {len(profile_results)} profile results to {profile_out_path}")
    else:
        print(f"\nNo wallets in {profile_input_path}; skipping profile tests.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
