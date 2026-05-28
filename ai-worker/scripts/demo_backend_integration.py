from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import UUID


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
    parser.add_argument(
        "--check-backend-health",
        metavar="URL",
        default=None,
        help="Optional backend health URL to GET before POST, for example http://localhost:8000/health.",
    )
    parser.add_argument(
        "--expected-status-code",
        type=int,
        default=200,
        help="Expected HTTP status code for POST mode. Default: 200.",
    )
    return parser.parse_args()


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _webhook_url(args: argparse.Namespace) -> str | None:
    return args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")


def _secret(args: argparse.Namespace) -> str | None:
    return args.secret or os.getenv("AI_WEBHOOK_SECRET")


def _is_uuid(value: str) -> bool:
    try:
        UUID(str(value))
    except ValueError:
        return False
    return True


def _print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _check_backend_health(health_url: str) -> bool:
    try:
        import requests

        response = requests.get(health_url, timeout=10)
    except Exception as exc:
        print(f"health_check_url={health_url}")
        print("health_check_ok=False")
        print(f"health_check_error={exc}")
        return False

    print(f"health_check_url={health_url}")
    print(f"health_check_status={response.status_code}")
    print("health_check_ok=" + str(200 <= response.status_code < 300))
    if response.text:
        print("health_check_body=")
        print(response.text)
    return 200 <= response.status_code < 300


def _post_payload(
    webhook_url: str,
    secret: str,
    payload: dict[str, object],
    expected_status_code: int,
) -> bool:
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
        return False

    print("post_attempted=True")
    print(f"response_status={response.status_code}")
    print(f"expected_status_code={expected_status_code}")
    print("response_matches_expected=" + str(response.status_code == expected_status_code))
    print("response_body=")
    print(response.text)
    return response.status_code == expected_status_code


def main() -> int:
    args = _parse_args()
    webhook_url = _webhook_url(args)
    secret = _secret(args)
    ai_result = build_completed_sample()
    payload = build_success_webhook_payload(args.job_id, ai_result)
    is_valid, issues = validate_webhook_payload(payload)
    backend_warnings = []
    if not _is_uuid(args.job_id):
        backend_warnings.append(
            "Backend route model expects job_id to be a UUID for real POST mode."
        )

    _print_section("CONFIG")
    print("Warning: classifier confidence is not pronunciation score.")
    print("Warning: heuristic_gop is not real GOP.")
    print("Warning: fallback alignment is approximate.")
    print(f"job_id={args.job_id}")
    print(f"target_webhook_url={webhook_url or 'not-configured'}")
    print("secret_configured=" + str(bool(secret)))
    print("post_requested=" + str(bool(args.post)))
    print(f"expected_status_code={args.expected_status_code}")

    if args.check_backend_health:
        _print_section("BACKEND HEALTH")
        _check_backend_health(args.check_backend_health)

    _print_section("PAYLOAD VALIDATION")
    print(f"payload_valid={is_valid}")
    if issues:
        print("payload_issues=")
        for issue in issues:
            print(f"- {issue}")
    if backend_warnings:
        print("backend_compatibility_warnings=")
        for warning in backend_warnings:
            print(f"- {warning}")

    _print_section("WEBHOOK PAYLOAD")
    print(_json(payload))

    if not args.post:
        _print_section("POST RESULT")
        print("post_attempted=False")
        print("dry_run=True")
        _print_section("NEXT VERIFY STEPS")
        print("For a real POST, use an existing practice_history UUID with status=processing.")
        print("After a successful POST, verify with GET /practice/<job_id> or inspect the practice_history row.")
        return 0

    _print_section("POST RESULT")
    if not webhook_url or not secret:
        print("post_attempted=False")
        print("post_skipped_reason=--post requires --webhook-url/NODE_WEBHOOK_URL/AI_WEBHOOK_URL and --secret/AI_WEBHOOK_SECRET.")
        return 2
    if not is_valid:
        print("post_attempted=False")
        print("post_skipped_reason=payload validation failed.")
        return 2

    post_ok = _post_payload(webhook_url, secret, payload, args.expected_status_code)

    _print_section("NEXT VERIFY STEPS")
    if post_ok:
        print(f"Check GET /practice/{args.job_id} with a valid user token.")
        print("Or inspect public.practice_history for the same id.")
        print("Expected columns: status=completed, score set, problem_phonemes set, feedback.ai_result present.")
    else:
        print("POST did not match the expected status code. Check backend logs and troubleshooting docs.")

    return 0 if post_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
