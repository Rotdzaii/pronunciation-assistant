from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit


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


DEFAULT_QUEUE_NAME = "practice_jobs"
DEFAULT_CHECKPOINT = REPO_ROOT / "ai-training" / "models" / "l2_arctic_cnn_attention_context_0_10.pt"
SENSITIVE_VALUE_MARKERS = (
    "c:\\",
    "/tmp/",
    "appdata\\local\\temp",
    ".textgrid",
    ".wav",
    ".webm",
    ".m4a",
    "x-amz-signature=",
    "token=",
    "sig=",
    "signature=",
)


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one PGMQ job and validate MFA-aligned cnn_attention_context worker flow.",
    )
    parser.add_argument("--queue-name", default=DEFAULT_QUEUE_NAME, help="PGMQ queue name. Default: practice_jobs.")
    parser.add_argument("--post", action="store_true", help="Explicitly POST the webhook payload.")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Explicitly archive the queue message, but only after a successful POST.",
    )
    parser.add_argument("--webhook-url", default=None, help="Optional webhook URL for --post.")
    parser.add_argument("--secret", default=None, help="Optional webhook secret for --post. Never printed.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local context checkpoint override.")
    parser.add_argument("--conda-env", default="mfa", help='MFA conda environment name. Default: "mfa".')
    parser.add_argument(
        "--dictionary-path",
        default="english_us_mfa",
        help='MFA dictionary name/path. Default: "english_us_mfa".',
    )
    parser.add_argument(
        "--acoustic-model-path",
        default="english_mfa",
        help='MFA acoustic model name/path. Default: "english_mfa".',
    )
    parser.add_argument(
        "--allow-fallback",
        type=_parse_bool,
        default=True,
        help="Allow fallback alignment when MFA fails. Default: true.",
    )
    return parser.parse_args()


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(AI_WORKER_ROOT / ".env")


def configure_environment(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = (
        Path(args.checkpoint_path).expanduser()
        if args.checkpoint_path
        else Path(os.getenv("CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT).expanduser()
    )
    config = {
        "SCORER_MODE": "cnn_attention_context",
        "ALIGNMENT_MODE": "mfa",
        "ALLOW_ALIGNMENT_FALLBACK": "true" if args.allow_fallback else "false",
        "MFA_CONDA_ENV": str(args.conda_env or "mfa"),
        "MFA_DICTIONARY_PATH": str(args.dictionary_path or "english_us_mfa"),
        "MFA_ACOUSTIC_MODEL_PATH": str(args.acoustic_model_path or "english_mfa"),
        "CNN_ATTENTION_CONTEXT_MODE": "context_0_10",
        "CNN_ATTENTION_CONTEXT_LEFT_SECONDS": "0.10",
        "CNN_ATTENTION_CONTEXT_RIGHT_SECONDS": "0.10",
        "CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH": str(checkpoint_path),
    }
    for key, value in config.items():
        os.environ[key] = str(value)
    return {
        "checkpoint_path": checkpoint_path,
        "config": config,
    }


def missing_supabase_env() -> list[str]:
    return [name for name in ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"] if not os.getenv(name)]


def safe_webhook_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def safe_job_for_print(job: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(job)
    if redacted.get("audio_url"):
        redacted["audio_url"] = "<audio-url-redacted>"
    if redacted.get("audio_path"):
        redacted["audio_path"] = "<audio-path-redacted>"
    return redacted


def find_sensitive_markers(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            findings.extend(find_sensitive_markers(item, next_path))
        return findings
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            findings.extend(find_sensitive_markers(item, next_path))
        return findings
    if isinstance(value, str):
        normalized = value.lower()
        for marker in SENSITIVE_VALUE_MARKERS:
            if marker in normalized:
                findings.append({"path": path or "<root>", "marker": marker})
    return findings


def payload_context_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}
    ai_result = feedback.get("ai_result") if isinstance(feedback.get("ai_result"), dict) else {}
    metadata = ai_result.get("metadata") if isinstance(ai_result.get("metadata"), dict) else {}
    return {
        "alignment_status": metadata.get("alignment_status"),
        "alignment_method": metadata.get("alignment_method"),
        "requested_alignment_mode": metadata.get("requested_alignment_mode"),
        "is_forced_alignment": metadata.get("is_forced_alignment"),
        "mfa_used": metadata.get("mfa_used"),
        "textgrid_parse_success": metadata.get("textgrid_parse_success"),
        "fallback_alignment": metadata.get("fallback_alignment"),
        "word_segments_count": metadata.get("word_segments_count"),
        "phone_segments_count": metadata.get("phone_segments_count"),
        "segments_count": metadata.get("segments_count"),
        "context_mode": metadata.get("context_mode"),
        "location_reliability": metadata.get("location_reliability"),
    }


def top_segment(ai_result: dict[str, Any]) -> dict[str, Any]:
    segments = ai_result.get("segments") or []
    if not segments:
        return {}
    return max(segments, key=lambda item: float(item.get("diagnosis_confidence") or 0.0))


def _download_audio_to_temp(audio_url: str) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(f"Downloading queue audio_url requires requests: {exc}") from exc

    response = requests.get(audio_url, timeout=60)
    response.raise_for_status()
    suffix = Path(urlparse(audio_url).path).suffix or ".audio"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(response.content)
    finally:
        temp_file.close()
    return temp_path


def prepare_scoring_job(job: dict[str, Any]) -> tuple[dict[str, Any], str]:
    audio_url = str(job.get("audio_url") or "").strip()
    if not audio_url:
        raise RuntimeError("Queue message audio_url is required.")

    parsed = urlparse(audio_url)
    if parsed.scheme in {"http", "https"}:
        temp_audio_path = _download_audio_to_temp(audio_url)
        prepared = dict(job)
        prepared["audio_path"] = str(temp_audio_path)
        prepared["audio_url"] = ""
        return prepared, "downloaded_to_temp"

    local_path = Path(audio_url).expanduser()
    prepared = dict(job)
    prepared["audio_path"] = str(local_path)
    prepared["audio_url"] = ""
    return prepared, "local_path"


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


def print_section(title: str, payload: Any) -> None:
    print(title)
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def main() -> int:
    args = parse_args()
    load_local_env()
    setup = configure_environment(args)
    checkpoint_path = setup["checkpoint_path"]

    config_summary = {
        "queue_name": args.queue_name,
        "scorer_mode": "cnn_attention_context",
        "alignment_mode": "mfa",
        "allow_fallback": args.allow_fallback,
        "mfa_conda_env": os.getenv("MFA_CONDA_ENV"),
        "mfa_dictionary_path": os.getenv("MFA_DICTIONARY_PATH"),
        "mfa_acoustic_model_path": os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
        "context_mode": os.getenv("CNN_ATTENTION_CONTEXT_MODE"),
        "post_requested": bool(args.post),
        "archive_requested": bool(args.archive),
        "checkpoint_path_configured": bool(str(checkpoint_path)),
        "webhook_url_safe": safe_webhook_url(args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")),
        "secret_configured": bool(args.secret or os.getenv("AI_WEBHOOK_SECRET")),
        "confidence_note": "Classifier confidence is not pronunciation correctness.",
        "score_note": "Heuristic score is not real GOP.",
        "alignment_note": "Alignment timing is not pronunciation correctness.",
    }
    print_section("CONFIG", config_summary)

    missing_env = missing_supabase_env()
    if missing_env:
        print_section(
            "QUEUE_MESSAGE",
            {
                "queue_read_attempted": False,
                "missing_supabase_env": missing_env,
                "no_queue_message": None,
            },
        )
        return 0

    try:
        from supabase import create_client
        import worker
    except Exception as exc:
        print_section("SETUP_ERROR", {"error": str(exc)})
        return 1

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    try:
        row = worker._read_one_job(client, args.queue_name)  # noqa: SLF001
    except Exception as exc:
        print_section("QUEUE_MESSAGE", {"queue_read_attempted": True, "queue_read_error": str(exc)})
        return 1

    if not row:
        print_section(
            "QUEUE_MESSAGE",
            {
                "queue_read_attempted": True,
                "queue_name": args.queue_name,
                "no_queue_message": True,
                "post_attempted": False,
                "archive_attempted": False,
            },
        )
        return 0

    try:
        msg_id, job = worker._parse_queue_row(row)  # noqa: SLF001
    except Exception as exc:
        print_section("QUEUE_MESSAGE", {"queue_parse_error": str(exc)})
        return 1

    print_section(
        "QUEUE_MESSAGE",
        {
            "queue_read_attempted": True,
            "queue_name": args.queue_name,
            "no_queue_message": False,
            "msg_id": msg_id,
            "job_id": job.get("job_id"),
            "student_id": job.get("student_id"),
            "target_word": job.get("target_word"),
            "audio_url": "<audio-url-redacted>",
        },
    )
    print_section("JOB_PAYLOAD", safe_job_for_print(job))

    if not checkpoint_path.exists():
        print_section(
            "FINAL_SUMMARY",
            {
                "queue_message_found": True,
                "download_status": "not_attempted",
                "inference_ran": False,
                "checkpoint_missing": True,
                "post_attempted": False,
                "archive_attempted": False,
            },
        )
        return 1

    temp_audio_path: Path | None = None
    post_attempted = False
    archive_attempted = False
    post_success = False
    archive_success = False
    queue_message_found = True
    download_status = "not_attempted"
    inference_ran = False

    try:
        try:
            scoring_job, download_status = prepare_scoring_job(job)
            temp_audio_value = str(scoring_job.get("audio_path") or "").strip()
            temp_audio_path = Path(temp_audio_value) if temp_audio_value and download_status == "downloaded_to_temp" else None
        except Exception as exc:
            print_section(
                "DOWNLOAD_RESULT",
                {
                    "download_success": False,
                    "download_status": "failed",
                    "audio_url": "<audio-url-redacted>",
                    "error": str(exc),
                },
            )
            print_section(
                "FINAL_SUMMARY",
                {
                    "queue_message_found": queue_message_found,
                    "download_status": "failed",
                    "inference_ran": False,
                    "post_attempted": False,
                    "archive_attempted": False,
                },
            )
            return 1

        print_section(
            "DOWNLOAD_RESULT",
            {
                "download_success": True,
                "download_status": download_status,
                "audio_url": "<audio-url-redacted>",
                "local_temp_path_redacted": temp_audio_path is not None,
            },
        )

        try:
            ai_result = worker._score(  # noqa: SLF001
                scoring_job,
                "cnn_attention_context",
                float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65")),
            )
            inference_ran = True
        except Exception as exc:
            payload = build_failed_webhook_payload(job["job_id"], str(exc))
            payload_valid, payload_issues = validate_webhook_payload(payload)
            safety_findings = find_sensitive_markers(payload)
            print_section(
                "AI_RESULT_SUMMARY",
                {
                    "inference_ran": False,
                    "status": "failed",
                    "error": str(exc),
                },
            )
            print_section(
                "WEBHOOK_PAYLOAD_SUMMARY",
                {
                    "job_id": payload.get("job_id"),
                    "status": payload.get("status"),
                    "feedback_ai_result_present": isinstance(payload.get("feedback", {}).get("ai_result"), dict),
                },
            )
            print_section(
                "VALIDATION",
                {
                    "ai_result_valid": False,
                    "ai_result_issues": [str(exc)],
                    "payload_valid": payload_valid,
                    "payload_issues": payload_issues,
                },
            )
            print_section(
                "METADATA_SAFETY_CHECK",
                {
                    "passed": not safety_findings,
                    "findings": safety_findings,
                    "checked_markers": list(SENSITIVE_VALUE_MARKERS),
                },
            )
            print_section(
                "POST_RESULT",
                {
                    "post_requested": bool(args.post),
                    "post_attempted": False,
                    "result": "not_attempted_due_to_inference_failure",
                },
            )
            print_section(
                "ARCHIVE_RESULT",
                {
                    "archive_requested": bool(args.archive),
                    "archive_attempted": False,
                    "result": "not_attempted_due_to_inference_failure",
                },
            )
            print_section(
                "FINAL_SUMMARY",
                {
                    "queue_message_found": queue_message_found,
                    "download_status": download_status,
                    "inference_ran": False,
                    "post_attempted": False,
                    "archive_attempted": False,
                },
            )
            return 1

        ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
        payload = build_success_webhook_payload(job["job_id"], ai_result)
        payload_valid, payload_issues = validate_webhook_payload(payload)
        safety_findings = find_sensitive_markers(payload)
        safety_passed = not safety_findings
        top = top_segment(ai_result)

        print_section(
            "AI_RESULT_SUMMARY",
            {
                "inference_ran": True,
                "status": ai_result.get("status"),
                "score": ai_result.get("score"),
                "score_note": ai_result.get("score_note"),
                "pronunciation_score_source": ai_result.get("metadata", {}).get("pronunciation_score_source"),
                "predicted_error_type": ai_result.get("predicted_error_type"),
                "problem_phonemes": ai_result.get("problem_phonemes"),
                "diagnosis_confidence": ai_result.get("diagnosis", {}).get("diagnosis_confidence"),
                "top_segment": {
                    "phone": top.get("phone"),
                    "word": top.get("word"),
                    "predicted_error_type": top.get("predicted_error_type"),
                    "diagnosis_confidence": top.get("diagnosis_confidence"),
                },
            },
        )
        print_section(
            "WEBHOOK_PAYLOAD_SUMMARY",
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
        )
        print_section(
            "VALIDATION",
            {
                "ai_result_valid": ai_result_valid,
                "ai_result_issues": ai_result_issues,
                "payload_valid": payload_valid,
                "payload_issues": payload_issues,
            },
        )
        print_section(
            "METADATA_SAFETY_CHECK",
            {
                "passed": safety_passed,
                "findings": safety_findings,
                "checked_markers": list(SENSITIVE_VALUE_MARKERS),
            },
        )

        post_result: dict[str, Any] = {
            "post_requested": bool(args.post),
            "post_attempted": False,
            "webhook_url_safe": safe_webhook_url(args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")),
            "result": "not_attempted",
        }
        if args.post:
            webhook_url = args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")
            secret = args.secret or os.getenv("AI_WEBHOOK_SECRET")
            if not ai_result_valid or not payload_valid or not safety_passed:
                post_result["result"] = "skipped_validation_failed"
            elif not webhook_url or not secret:
                post_result["result"] = "skipped_missing_webhook_or_secret"
            else:
                post_attempted = True
                post_success, post_message = post_payload(webhook_url, secret, payload)
                post_result["post_attempted"] = True
                post_result["post_success"] = post_success
                post_result["result"] = post_message
        print_section("POST_RESULT", post_result)

        archive_result: dict[str, Any] = {
            "archive_requested": bool(args.archive),
            "archive_attempted": False,
            "result": "not_attempted",
        }
        if args.archive:
            if not args.post:
                archive_result["result"] = "skipped_requires_post"
            elif not post_success:
                archive_result["result"] = "skipped_post_not_successful"
            else:
                archive_attempted = True
                try:
                    worker._archive_job(client, args.queue_name, msg_id)  # noqa: SLF001
                    archive_success = True
                    archive_result["archive_attempted"] = True
                    archive_result["archive_success"] = True
                    archive_result["result"] = "archived"
                except Exception as exc:
                    archive_result["archive_attempted"] = True
                    archive_result["archive_success"] = False
                    archive_result["result"] = str(exc)
                    print_section("ARCHIVE_RESULT", archive_result)
                    print_section(
                        "FINAL_SUMMARY",
                        {
                            "queue_message_found": queue_message_found,
                            "download_status": download_status,
                            "inference_ran": inference_ran,
                            "post_attempted": post_attempted,
                            "post_success": post_success,
                            "archive_attempted": archive_attempted,
                            "archive_success": archive_success,
                        },
                    )
                    return 1
        print_section("ARCHIVE_RESULT", archive_result)

        print_section(
            "FINAL_SUMMARY",
            {
                "queue_message_found": queue_message_found,
                "download_status": download_status,
                "inference_ran": inference_ran,
                "ai_result_valid": ai_result_valid,
                "payload_valid": payload_valid,
                "metadata_safety_check_passed": safety_passed,
                "post_attempted": post_attempted,
                "post_success": post_success if post_attempted else None,
                "archive_attempted": archive_attempted,
                "archive_success": archive_success if archive_attempted else None,
                "no_queue_message": False,
            },
        )
        return 0 if ai_result_valid and payload_valid and safety_passed and (not post_attempted or post_success) and (not archive_attempted or archive_success) else 1
    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
