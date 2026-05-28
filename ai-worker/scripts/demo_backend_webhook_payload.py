from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.contracts.ai_result_contract import build_failed_ai_result  # noqa: E402
from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.webhook_payload import (  # noqa: E402
    build_failed_webhook_payload,
    build_success_webhook_payload,
    validate_webhook_payload,
)
from demo_final_ai_output import build_completed_sample  # noqa: E402


def _print_payload(label: str, payload: dict[str, object]) -> None:
    is_valid, issues = validate_webhook_payload(payload)
    print(f"=== {label} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"webhook_payload_valid={is_valid}")
    if issues:
        print("webhook_payload_issues=")
        for issue in issues:
            print(f"- {issue}")


def main() -> int:
    print("Warning: classifier confidence is not pronunciation score.")
    print("Warning: heuristic_gop is not real GOP.")
    print("Warning: fallback alignment is approximate.")

    completed_result = build_completed_sample()
    result_is_valid, result_issues = validate_ai_result(completed_result)
    print(f"completed_ai_result_valid={result_is_valid}")
    if result_issues:
        for issue in result_issues:
            print(f"- {issue}")

    success_payload = build_success_webhook_payload(
        "11111111-1111-1111-1111-111111111111",
        completed_result,
    )
    failed_payload = build_failed_webhook_payload(
        "22222222-2222-2222-2222-222222222222",
        "demo failure",
        build_failed_ai_result(error="demo failure"),
    )

    _print_payload("success_webhook_payload", success_payload)
    _print_payload("failed_webhook_payload", failed_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
