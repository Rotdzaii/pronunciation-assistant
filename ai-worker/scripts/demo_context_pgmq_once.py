from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.webhook_payload import (  # noqa: E402
    build_failed_webhook_payload,
    build_success_webhook_payload,
    validate_webhook_payload,
)


SCORER_MODE = "cnn_attention_context"
DEFAULT_QUEUE_NAME = "practice_jobs"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "ai-training"
    / "models"
    / "l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read one PGMQ job and validate context CNN Attention worker flow.")
    parser.add_argument("--post", action="store_true", help="Explicitly POST the webhook payload.")
    parser.add_argument("--archive", action="store_true", help="Explicitly archive the queue message after success.")
    parser.add_argument("--queue-name", default=DEFAULT_QUEUE_NAME, help="PGMQ queue name. Default: practice_jobs.")
    parser.add_argument("--webhook-url", default=None, help="Optional webhook URL for --post.")
    parser.add_argument("--secret", default=None, help="Optional webhook secret for --post.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local context checkpoint path.")
    parser.add_argument("--message-id", type=int, default=None, help="Optional expected message id guard.")
    return parser.parse_args()


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(AI_WORKER_ROOT / ".env")


def configure_context_environment(checkpoint_path: Path) -> None:
    os.environ["SCORER_MODE"] = SCORER_MODE
    os.environ["CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"] = str(checkpoint_path)
    os.environ["CNN_ATTENTION_CONTEXT_MODE"] = "context_0_10"
    os.environ["CNN_ATTENTION_CONTEXT_LEFT_SECONDS"] = "0.10"
    os.environ["CNN_ATTENTION_CONTEXT_RIGHT_SECONDS"] = "0.10"


def missing_supabase_env() -> list[str]:
    return [name for name in ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"] if not os.getenv(name)]


def top_segment(ai_result: dict[str, Any]) -> dict[str, Any]:
    segments = ai_result.get("segments") or []
    if not segments:
        return {}
    return max(segments, key=lambda item: float(item.get("diagnosis_confidence") or 0.0))


def payload_context_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}
    ai_result = feedback.get("ai_result") if isinstance(feedback.get("ai_result"), dict) else {}
    metadata = ai_result.get("metadata") if isinstance(ai_result.get("metadata"), dict) else {}
    return {
        "context_mode": metadata.get("context_mode"),
        "context_used": metadata.get("context_used"),
        "context_left_seconds": metadata.get("context_left_seconds"),
        "context_right_seconds": metadata.get("context_right_seconds"),
        "segment_start_time": metadata.get("segment_start_time"),
        "segment_end_time": metadata.get("segment_end_time"),
        "crop_start_time": metadata.get("crop_start_time"),
        "crop_end_time": metadata.get("crop_end_time"),
    }


def safe_job_for_print(job: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(job)
    if redacted.get("audio_url"):
        redacted["audio_url"] = "<audio-url-redacted>"
    if redacted.get("audio_path"):
        redacted["audio_path"] = "<audio-path-redacted>"
    return redacted


def post_payload(webhook_url: str, secret: str, payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        import requests
    except ImportError as exc:
        return False, f"requests is not installed: {exc}"

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"x-ai-webhook-secret": secret},
            timeout=30,
        )
    except Exception as exc:
        return False, str(exc)

    return 200 <= response.status_code < 300, f"status_code={response.status_code} body={response.text[:500]}"


def main() -> int:
    args = parse_args()
    load_local_env()
    checkpoint_path = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else DEFAULT_CHECKPOINT

    print("CONFIG")
    print(f"queue_name={args.queue_name}")
    print(f"scorer_mode={SCORER_MODE}")
    print("context_mode=context_0_10")
    print(f"post_requested={bool(args.post)}")
    print(f"archive_requested={bool(args.archive)}")
    print("confidence_note=Classifier confidence is not pronunciation correctness.")
    print("score_note=Heuristic score is not real GOP.")
    print("alignment_note=Fallback alignment is approximate.")

    missing_env = missing_supabase_env()
    if missing_env:
        print("QUEUE MESSAGE")
        print("queue_read_attempted=False")
        print("missing_supabase_env=" + ", ".join(missing_env))
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in ai-worker/.env or the shell environment.")
        return 0

    try:
        from supabase import create_client
        import worker
    except Exception as exc:
        print(f"setup_error={exc}")
        return 1

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    try:
        row = worker._read_one_job(client, args.queue_name)  # noqa: SLF001
    except Exception as exc:
        print("QUEUE MESSAGE")
        print(f"queue_read_error={exc}")
        return 1

    print("QUEUE MESSAGE")
    if not row:
        print(f"no_queue_message=True queue_name={args.queue_name}")
        print("No PGMQ message was available. Nothing was posted or archived.")
        return 0

    try:
        msg_id, job = worker._parse_queue_row(row)  # noqa: SLF001
    except Exception as exc:
        print(f"queue_parse_error={exc}")
        return 1

    print(json.dumps({"msg_id": msg_id, "raw_keys": sorted(row.keys())}, indent=2, ensure_ascii=False))
    if args.message_id is not None and msg_id != args.message_id:
        print(f"message_id_guard_skipped=True expected={args.message_id} actual={msg_id}")
        print("No POST or archive was attempted.")
        return 0

    if not checkpoint_path.exists():
        print("SCORER RESULT SUMMARY")
        print("inference_ran=False")
        print("checkpoint_error=Context checkpoint not found.")
        print(f"expected_checkpoint={checkpoint_path}")
        print("Set --checkpoint-path or CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH to an existing local .pt file.")
        return 1

    configure_context_environment(checkpoint_path)

    print("JOB PAYLOAD")
    print(json.dumps(safe_job_for_print(job), indent=2, ensure_ascii=False))

    try:
        ai_result = worker._score(job, SCORER_MODE, float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65")))  # noqa: SLF001
    except Exception as exc:
        print("SCORER RESULT SUMMARY")
        print(f"inference_ran=False error={exc}")
        payload = build_failed_webhook_payload(job["job_id"], str(exc))
        print("WEBHOOK PAYLOAD SUMMARY")
        print(json.dumps({"status": payload.get("status"), "error_message": payload.get("error_message")}, indent=2))
        return 1

    ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
    payload = build_success_webhook_payload(job["job_id"], ai_result)
    payload_valid, payload_issues = validate_webhook_payload(payload)
    top = top_segment(ai_result)

    print("SCORER RESULT SUMMARY")
    print(
        json.dumps(
            {
                "inference_ran": True,
                "predicted_error_type": ai_result.get("predicted_error_type"),
                "class_probabilities": ai_result.get("diagnosis", {}).get("class_probabilities"),
                "diagnosis_confidence": ai_result.get("diagnosis", {}).get("diagnosis_confidence"),
                "confidence_note": ai_result.get("diagnosis", {}).get("confidence_note"),
                "score_note": ai_result.get("score_note"),
                "top_segment": {
                    "phone": top.get("phone"),
                    "word": top.get("word"),
                    "predicted_error_type": top.get("predicted_error_type"),
                    "diagnosis_confidence": top.get("diagnosis_confidence"),
                    "context": top.get("context"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("VALIDATION")
    print(f"ai_result_valid={ai_result_valid}")
    if ai_result_issues:
        print("ai_result_issues=")
        for issue in ai_result_issues:
            print(f"- {issue}")
    print(f"payload_valid={payload_valid}")
    if payload_issues:
        print("payload_issues=")
        for issue in payload_issues:
            print(f"- {issue}")

    print("WEBHOOK PAYLOAD SUMMARY")
    print(
        json.dumps(
            {
                "job_id": payload.get("job_id"),
                "status": payload.get("status"),
                "score": payload.get("score"),
                "score_note": payload.get("score_note"),
                "predicted_error_type": payload.get("predicted_error_type"),
                "problem_phonemes": payload.get("problem_phonemes"),
                "scorer": payload.get("scorer"),
                "context_metadata": payload_context_metadata(payload),
                "feedback_ai_result_present": isinstance(payload.get("feedback", {}).get("ai_result"), dict),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print("POST RESULT")
    post_success = False
    if args.post:
        if not ai_result_valid or not payload_valid:
            print("post_skipped_reason=AI result or webhook payload validation failed.")
            return 1
        webhook_url = args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")
        secret = args.secret or os.getenv("AI_WEBHOOK_SECRET")
        if not webhook_url or not secret:
            print("post_skipped_reason=--webhook-url/NODE_WEBHOOK_URL and --secret/AI_WEBHOOK_SECRET are required.")
            return 1
        post_success, post_message = post_payload(webhook_url, secret, payload)
        print(f"post_success={post_success}")
        print(f"post_result={post_message}")
        if not post_success:
            print("ARCHIVE RESULT")
            print("archive_attempted=False reason=post_failed")
            return 1
    else:
        print("post_attempted=False")

    print("ARCHIVE RESULT")
    if args.archive:
        if not args.post:
            print("archive_attempted=False reason=--archive requires --post")
            return 1
        if not post_success:
            print("archive_attempted=False reason=post_not_successful")
            return 1
        try:
            worker._archive_job(client, args.queue_name, msg_id)  # noqa: SLF001
        except Exception as exc:
            print(f"archive_success=False archive_error={exc}")
            return 1
        print("archive_success=True")
    else:
        print("archive_attempted=False")

    print("NEXT VERIFY STEPS")
    print("- Dry-run: confirm ai_result_valid=True, payload_valid=True, post_attempted=False, archive_attempted=False.")
    print("- With --post: verify practice_history.status=completed and feedback.ai_result.metadata.context_used=true.")
    print("- With --post --archive: verify the PGMQ message was archived only after successful POST.")
    return 0 if ai_result_valid and payload_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
