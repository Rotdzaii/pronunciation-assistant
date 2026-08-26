from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_contract import build_ai_result  # noqa: E402
from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.webhook_payload import (  # noqa: E402
    build_success_webhook_payload,
    validate_webhook_payload,
)


DEFAULT_TRANSCRIPT = "Architecture"
DEFAULT_JOB_ID = "demo-job-id"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "ai-training" / "models" / "l2_arctic_cnn_attention_context_0_10.pt"
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MFA-aligned CNN Attention context AI result to backend webhook payload conversion.",
    )
    parser.add_argument("--audio-path", default=None, help="Optional local audio file. Real MFA inference runs only when provided.")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT, help=f'Transcript text. Default: "{DEFAULT_TRANSCRIPT}"')
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID, help=f'Job id used in the webhook payload. Default: "{DEFAULT_JOB_ID}"')
    parser.add_argument("--checkpoint-path", default=None, help="Optional local context checkpoint override.")
    parser.add_argument("--alignment-mode", default="mfa", help='Alignment mode. Default: "mfa".')
    parser.add_argument(
        "--allow-fallback",
        type=_parse_bool,
        default=True,
        help="Allow fallback alignment when MFA fails. Default: true.",
    )
    parser.add_argument("--conda-env", default="mfa", help='MFA conda environment name. Default: "mfa".')
    parser.add_argument(
        "--dictionary-path",
        default="english_us_arpa",
        help='MFA dictionary name/path. Default: "english_us_arpa".',
    )
    parser.add_argument(
        "--acoustic-model-path",
        default="english_us_arpa",
        help='MFA acoustic model name/path. Default: "english_us_arpa".',
    )
    parser.add_argument("--post", action="store_true", help="Explicitly POST the payload to the backend webhook.")
    parser.add_argument("--webhook-url", default=None, help="Optional backend webhook URL for --post.")
    parser.add_argument("--secret", default=None, help="Optional webhook secret for --post. Never printed.")
    parser.add_argument("--dry-run", action="store_true", help="Use a representative safe sample AI result instead of real inference.")
    return parser.parse_args()


def _set_env(args: argparse.Namespace) -> dict[str, str]:
    checkpoint_path = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else DEFAULT_CHECKPOINT
    config = {
        "SCORER_MODE": "cnn_attention_context",
        "ALIGNMENT_MODE": str(args.alignment_mode or "mfa"),
        "ALLOW_ALIGNMENT_FALLBACK": "true" if args.allow_fallback else "false",
        "MFA_CONDA_ENV": str(args.conda_env or "mfa"),
        "MFA_DICTIONARY_PATH": str(args.dictionary_path or "english_us_arpa"),
        "MFA_ACOUSTIC_MODEL_PATH": str(args.acoustic_model_path or "english_us_arpa"),
        "CNN_ATTENTION_CONTEXT_MODE": "context_0_10",
        "CNN_ATTENTION_CONTEXT_LEFT_SECONDS": "0.10",
        "CNN_ATTENTION_CONTEXT_RIGHT_SECONDS": "0.10",
        "CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH": str(checkpoint_path),
    }
    for key, value in config.items():
        os.environ[key] = value
    return config


def _safe_webhook_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        normalized = value.lower()
        if any(marker in normalized for marker in SENSITIVE_VALUE_MARKERS):
            return "[redacted-sensitive-value]"
    return value


def _find_sensitive_markers(value: Any, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            findings.extend(_find_sensitive_markers(item, next_path))
        return findings
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            findings.extend(_find_sensitive_markers(item, next_path))
        return findings
    if isinstance(value, str):
        normalized = value.lower()
        for marker in SENSITIVE_VALUE_MARKERS:
            if marker in normalized:
                findings.append({"path": path or "<root>", "marker": marker})
    return findings


def _print_section(title: str, payload: Any) -> None:
    print(title)
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def _build_sample_ai_result(transcript: str) -> dict[str, Any]:
    score_note = "Heuristic/demo score, not production GOP."
    result = build_ai_result(
        score=67.1,
        problem_phonemes=["AA", "K", "T"],
        predicted_error_type="deletion",
        class_probabilities={
            "addition": 0.1124695730,
            "deletion": 0.6575304270,
            "substitution": 0.2299999999,
        },
        diagnosis_confidence=0.6575304269790649,
        scorer={
            "name": "cnn_attention_context",
            "type": "phone_error_classifier",
            "version": "cnn_attention_context_0_10_phase2_candidate",
        },
        score_note=score_note,
        pronunciation_score_source="heuristic_gop",
        metadata={
            "model_output_is_scoring": False,
            "alignment_used": True,
            "alignment_status": "success",
            "alignment_method": "mfa",
            "requested_alignment_mode": "mfa",
            "alignment_note": "MFA forced alignment timing was used for segment localization.",
            "gop_used": False,
            "hybrid_used": True,
            "hybrid_method": "alignment_score_classifier_merge",
            "hybrid_status": "completed",
            "is_forced_alignment": True,
            "mfa_used": True,
            "mfa_attempted": True,
            "mfa_exit_code": 0,
            "textgrid_parse_success": True,
            "fallback_alignment": False,
            "fallback_reason": None,
            "word_segments_count": 1,
            "phone_segments_count": 9,
            "location_reliability": "forced_alignment",
            "is_demo_score": True,
            "score_note": score_note,
            "pronunciation_score_source": "heuristic_gop",
            "scoring_method": "heuristic_gop",
            "scoring_is_heuristic": True,
            "scoring_status": "completed",
            "context_mode": "context_0_10",
            "context_used": True,
            "context_left_seconds": 0.10,
            "context_right_seconds": 0.10,
            "segment_start_time": 0.18,
            "segment_end_time": 0.29,
            "crop_start_time": 0.08,
            "crop_end_time": 0.39,
            "segments_count": 9,
            "limitation": (
                "Alignment timing is not pronunciation correctness. "
                "Classifier confidence is not pronunciation correctness. "
                "Heuristic score is not real GOP."
            ),
            "target_transcript": transcript,
        },
        diagnosis_extra={
            "top_issues": [
                {
                    "phone": "AA",
                    "word": transcript,
                    "predicted_error_type": "deletion",
                    "diagnosis_confidence": 0.6575304269790649,
                    "class_probabilities": {
                        "addition": 0.1124695730,
                        "deletion": 0.6575304270,
                        "substitution": 0.2299999999,
                    },
                }
            ],
            "severity": "moderate",
        },
    )
    result["segments"] = [
        {
            "index": 0,
            "type": "phone",
            "phone": "AA",
            "word": transcript,
            "start": 0.18,
            "end": 0.29,
            "predicted_error_type": "deletion",
            "diagnosis_confidence": 0.6575304269790649,
            "class_probabilities": {
                "addition": 0.1124695730,
                "deletion": 0.6575304270,
                "substitution": 0.2299999999,
            },
            "confidence_note": "Classifier confidence, not pronunciation correctness.",
            "context": {
                "context_mode": "context_0_10",
                "context_used": True,
                "context_left_seconds": 0.10,
                "context_right_seconds": 0.10,
                "segment_start_time": 0.18,
                "segment_end_time": 0.29,
                "crop_start_time": 0.08,
                "crop_end_time": 0.39,
            },
        }
    ]
    result["feedback"]["diagnosis"] = result["diagnosis"]
    result["feedback"]["scorer"] = result["scorer"]
    result["feedback"]["metadata"] = result["metadata"]
    return result


def _run_real_inference(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(str(args.audio_path)).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        from app.scorers.cnn_attention_scorer import score_pronunciation_context
    except Exception as exc:
        raise RuntimeError(f"Unable to import context scorer: {exc}") from exc

    return score_pronunciation_context(
        {
            "job_id": args.job_id,
            "audio_path": str(audio_path),
            "prompt_text": args.transcript,
            "target_text": args.transcript,
            "target_sentence": args.transcript,
            "target_word": args.transcript,
        }
    )


def _ai_result_summary(ai_result: dict[str, Any], mode: str) -> dict[str, Any]:
    metadata = ai_result.get("metadata") if isinstance(ai_result.get("metadata"), dict) else {}
    diagnosis = ai_result.get("diagnosis") if isinstance(ai_result.get("diagnosis"), dict) else {}
    scorer = ai_result.get("scorer") if isinstance(ai_result.get("scorer"), dict) else {}
    return {
        "execution_mode": mode,
        "status": ai_result.get("status"),
        "score": ai_result.get("score"),
        "score_note": ai_result.get("score_note"),
        "pronunciation_score_source": ai_result.get("pronunciation_score_source"),
        "predicted_error_type": ai_result.get("predicted_error_type"),
        "problem_phonemes": ai_result.get("problem_phonemes"),
        "diagnosis_confidence": diagnosis.get("diagnosis_confidence"),
        "scorer_name": scorer.get("name"),
        "context_mode": metadata.get("context_mode"),
        "alignment_status": metadata.get("alignment_status"),
        "alignment_method": metadata.get("alignment_method"),
        "is_forced_alignment": metadata.get("is_forced_alignment"),
        "mfa_used": metadata.get("mfa_used"),
        "textgrid_parse_success": metadata.get("textgrid_parse_success"),
        "fallback_alignment": metadata.get("fallback_alignment"),
        "word_segments_count": metadata.get("word_segments_count"),
        "phone_segments_count": metadata.get("phone_segments_count"),
        "segments_count": metadata.get("segments_count"),
        "location_reliability": metadata.get("location_reliability"),
    }


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    feedback = payload.get("feedback") if isinstance(payload.get("feedback"), dict) else {}
    ai_result = feedback.get("ai_result") if isinstance(feedback.get("ai_result"), dict) else {}
    metadata = ai_result.get("metadata") if isinstance(ai_result.get("metadata"), dict) else {}
    return {
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "score": payload.get("score"),
        "problem_phonemes": payload.get("problem_phonemes"),
        "feedback_summary_present": bool(str(feedback.get("summary") or "").strip()),
        "feedback_tips_count": len(feedback.get("tips") or []),
        "feedback_ai_result_present": isinstance(feedback.get("ai_result"), dict),
        "legacy_fields_present": {
            "job_id": "job_id" in payload,
            "status": "status" in payload,
            "score": "score" in payload,
            "problem_phonemes": "problem_phonemes" in payload,
            "feedback": "feedback" in payload,
        },
        "rich_fields_present": {
            "predicted_error_type": "predicted_error_type" in payload,
            "diagnosis": isinstance(payload.get("diagnosis"), dict),
            "scorer": isinstance(payload.get("scorer"), dict),
            "metadata": isinstance(payload.get("metadata"), dict),
            "ai_result": isinstance(payload.get("ai_result"), dict),
        },
        "preserved_metadata": {
            "alignment_method": metadata.get("alignment_method"),
            "is_forced_alignment": metadata.get("is_forced_alignment"),
            "mfa_used": metadata.get("mfa_used"),
            "textgrid_parse_success": metadata.get("textgrid_parse_success"),
            "fallback_alignment": metadata.get("fallback_alignment"),
            "location_reliability": metadata.get("location_reliability"),
        },
    }


def _post_payload(webhook_url: str, secret: str, payload: dict[str, Any]) -> tuple[bool, str]:
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
    args = _parse_args()
    config = _set_env(args)
    execution_mode = "sample"
    post_attempted = False

    config_summary = {
        "scorer_mode": config["SCORER_MODE"],
        "alignment_mode": config["ALIGNMENT_MODE"],
        "allow_fallback": args.allow_fallback,
        "mfa_conda_env": config["MFA_CONDA_ENV"],
        "mfa_dictionary_path": config["MFA_DICTIONARY_PATH"],
        "mfa_acoustic_model_path": config["MFA_ACOUSTIC_MODEL_PATH"],
        "context_mode": config["CNN_ATTENTION_CONTEXT_MODE"],
        "context_left_seconds": config["CNN_ATTENTION_CONTEXT_LEFT_SECONDS"],
        "context_right_seconds": config["CNN_ATTENTION_CONTEXT_RIGHT_SECONDS"],
        "checkpoint_configured": bool(args.checkpoint_path or DEFAULT_CHECKPOINT),
        "audio_path_provided": bool(args.audio_path),
        "transcript": args.transcript,
        "job_id": args.job_id,
        "dry_run": bool(args.dry_run),
        "post_requested": bool(args.post),
        "webhook_url_configured": bool(args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")),
        "webhook_url_safe": _safe_webhook_url(args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")),
        "secret_configured": bool(args.secret or os.getenv("AI_WEBHOOK_SECRET")),
    }
    _print_section("CONFIG", config_summary)

    try:
        if args.dry_run or not args.audio_path:
            ai_result = _build_sample_ai_result(args.transcript)
            dry_run_reason = "--dry-run requested." if args.dry_run else "No --audio-path provided."
        else:
            execution_mode = "real_mfa_inference"
            dry_run_reason = None
            ai_result = _run_real_inference(args)
    except Exception as exc:
        _print_section(
            "AI_RESULT_SUMMARY",
            {
                "execution_mode": execution_mode,
                "executed": False,
                "error": str(exc),
            },
        )
        _print_section(
            "WEBHOOK_PAYLOAD_SUMMARY",
            {
                "executed": False,
                "note": "Payload build skipped because inference did not produce an AI result.",
            },
        )
        _print_section(
            "VALIDATION",
            {
                "ai_result_valid": False,
                "payload_valid": False,
                "issues": [str(exc)],
            },
        )
        _print_section(
            "METADATA_SAFETY_CHECK",
            {
                "passed": False,
                "issues": ["AI result was not produced, so payload safety could not be confirmed."],
            },
        )
        _print_section(
            "POST_RESULT",
            {
                "post_attempted": False,
                "post_requested": bool(args.post),
                "result": "not_attempted",
            },
        )
        _print_section(
            "NEXT_VERIFY_STEPS",
            {
                "real_local_audio_validation": (
                    ".\\ai-worker\\.venv\\Scripts\\python.exe ai-worker\\scripts\\demo_mfa_backend_payload.py "
                    "--audio-path path\\to\\architecture.wav --transcript \"Architecture\""
                ),
                "optional_backend_post": (
                    ".\\ai-worker\\.venv\\Scripts\\python.exe ai-worker\\scripts\\demo_mfa_backend_payload.py "
                    "--audio-path path\\to\\architecture.wav --transcript \"Architecture\" "
                    "--job-id <existing-practice-history-uuid> --post "
                    "--webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-ai-webhook-secret>"
                ),
            },
        )
        return 1

    ai_result = _safe_value(ai_result)
    payload = build_success_webhook_payload(args.job_id, ai_result)
    payload = _safe_value(payload)

    ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
    payload_valid, payload_issues = validate_webhook_payload(payload)
    safety_findings = _find_sensitive_markers(payload)
    safety_passed = not safety_findings

    ai_summary = _ai_result_summary(ai_result, execution_mode)
    if dry_run_reason:
        ai_summary["sample_reason"] = dry_run_reason
    _print_section("AI_RESULT_SUMMARY", ai_summary)
    _print_section("WEBHOOK_PAYLOAD_SUMMARY", _payload_summary(payload))
    _print_section(
        "VALIDATION",
        {
            "ai_result_valid": ai_result_valid,
            "ai_result_issues": ai_result_issues,
            "payload_valid": payload_valid,
            "payload_issues": payload_issues,
        },
    )
    _print_section(
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
            post_ok, post_message = _post_payload(webhook_url, secret, payload)
            post_result["post_attempted"] = True
            post_result["webhook_url_safe"] = _safe_webhook_url(webhook_url)
            post_result["post_success"] = post_ok
            post_result["result"] = post_message

    _print_section("POST_RESULT", post_result)
    _print_section(
        "NEXT_VERIFY_STEPS",
        {
            "real_local_audio_validation": (
                ".\\ai-worker\\.venv\\Scripts\\python.exe ai-worker\\scripts\\demo_mfa_backend_payload.py "
                "--audio-path path\\to\\architecture.wav --transcript \"Architecture\""
            ),
            "optional_backend_post": (
                ".\\ai-worker\\.venv\\Scripts\\python.exe ai-worker\\scripts\\demo_mfa_backend_payload.py "
                "--audio-path path\\to\\architecture.wav --transcript \"Architecture\" "
                "--job-id <existing-practice-history-uuid> --post "
                "--webhook-url http://localhost:8000/practice/webhook/ai-result --secret <local-ai-webhook-secret>"
            ),
            "backend_expectations": [
                "POST remains opt-in only.",
                "Verify practice_history.status=completed for a real existing job id.",
                "Verify feedback.ai_result preserves normalized scorer, diagnosis, metadata, and segments.",
            ],
        },
    )
    return 0 if ai_result_valid and payload_valid and safety_passed and (not post_attempted or post_result.get("post_success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
