from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.contracts.webhook_payload import build_success_webhook_payload, validate_webhook_payload  # noqa: E402
from demo_final_ai_output import build_completed_sample  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or POST a sample AI Worker webhook payload.")
    parser.add_argument("--job-id", required=True, help="Existing practice_history job UUID for POST mode.")
    parser.add_argument("--webhook-url", default=None, help="Backend webhook URL.")
    parser.add_argument("--secret", default=None, help="AI webhook secret. Never printed.")
    parser.add_argument("--post", action="store_true", help="Actually POST to backend.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Validate payload without POSTing. Default.")
    return parser.parse_args()


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _webhook_url(args: argparse.Namespace) -> str | None:
    return args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")


def _secret(args: argparse.Namespace) -> str | None:
    return args.secret or os.getenv("AI_WEBHOOK_SECRET")


def _post_payload(webhook_url: str, secret: str, payload: dict[str, object]) -> None:
    try:
        import requests

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"x-ai-webhook-secret": secret},
            timeout=30,
        )
    except Exception as exc:
        print("post_attempted=True")
        print(f"post_error={exc}")
        return

    print("post_attempted=True")
    print(f"response_status={response.status_code}")
    print("response_body=")
    print(response.text)


def main() -> int:
    args = _parse_args()
    webhook_url = _webhook_url(args)
    secret = _secret(args)
    ai_result = build_completed_sample()
    payload = build_success_webhook_payload(args.job_id, ai_result)
    is_valid, issues = validate_webhook_payload(payload)

    print("Warning: classifier confidence is not pronunciation score.")
    print("Warning: heuristic_gop is not real GOP.")
    print("Warning: fallback alignment is approximate.")
    print(f"target_webhook_url={webhook_url or 'not-configured'}")
    print("secret_configured=" + str(bool(secret)))
    print(f"payload_valid={is_valid}")
    if issues:
        print("payload_issues=")
        for issue in issues:
            print(f"- {issue}")

    print("=== webhook_payload ===")
    print(_json(payload))

    if not args.post:
        print("post_attempted=False")
        print("dry_run=True")
        return 0

    if not webhook_url or not secret:
        print("post_attempted=False")
        print("post_skipped_reason=--post requires --webhook-url/NODE_WEBHOOK_URL/AI_WEBHOOK_URL and --secret/AI_WEBHOOK_SECRET.")
        return 2
    if not is_valid:
        print("post_attempted=False")
        print("post_skipped_reason=payload validation failed.")
        return 2

    _post_payload(webhook_url, secret, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
