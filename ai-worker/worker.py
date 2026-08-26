from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from supabase import Client, create_client

from app.contracts.ai_result_contract import build_failed_ai_result
from app.audio.storage_resolver import AudioReferenceError, resolve_audio_reference
from app.contracts.webhook_payload import (
    build_failed_webhook_payload,
    build_success_webhook_payload,
    validate_webhook_payload,
)
from scorers.mock_scorer import score_pronunciation as score_mock_pronunciation


DEFAULT_VISIBILITY_TIMEOUT_SECONDS = 60
DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_WORKER_IDLE_BACKOFF_MAX_SECONDS = 10.0
DEFAULT_MODEL_VERSION = "phoenix_v2_stable"
SUPPORTED_SCORER_MODES = ("mock", "wav2vec2", "cnn_attention", "cnn_attention_context")
STANDARD_ERROR_TYPES = {
    "audio_decode_failed",
    "audio_preprocess_failed",
    "audio_quality_rejected",
    "alignment_failed",
    "checkpoint_missing",
    "checkpoint_incompatible",
    "scorer_timeout",
    "scorer_failed",
    "webhook_failed",
    "unknown_error",
}

AUDIO_SNR_REJECT_THRESHOLD_DB = 15.0
AUDIO_MIN_VOICED_DURATION_SECONDS = 0.3
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


class PhoenixWorkerError(RuntimeError):
    """Worker-level failure with a Phoenix v2 output-contract error type."""

    def __init__(self, message: str, error_type: str = "unknown_error") -> None:
        super().__init__(message)
        self.error_type = error_type if error_type in STANDARD_ERROR_TYPES else "unknown_error"


def _load_env() -> dict[str, Any]:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path)

    required_names = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "NODE_WEBHOOK_URL",
        "AI_WEBHOOK_SECRET",
    ]
    missing = [name for name in required_names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    try:
        confidence_threshold = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65"))
    except ValueError as exc:
        raise RuntimeError("MODEL_CONFIDENCE_THRESHOLD must be a number") from exc

    if not 0 <= confidence_threshold <= 1:
        raise RuntimeError("MODEL_CONFIDENCE_THRESHOLD must be between 0 and 1")

    worker_mode = os.getenv("WORKER_MODE", "loop").strip().lower()
    if worker_mode not in {"once", "loop"}:
        raise RuntimeError("WORKER_MODE must be either 'once' or 'loop'")

    try:
        poll_interval_seconds = float(
            os.getenv("WORKER_POLL_INTERVAL_SECONDS", str(DEFAULT_WORKER_POLL_INTERVAL_SECONDS))
        )
    except ValueError as exc:
        raise RuntimeError("WORKER_POLL_INTERVAL_SECONDS must be a number") from exc

    try:
        idle_backoff_max_seconds = float(
            os.getenv("WORKER_IDLE_BACKOFF_MAX_SECONDS", str(DEFAULT_WORKER_IDLE_BACKOFF_MAX_SECONDS))
        )
    except ValueError as exc:
        raise RuntimeError("WORKER_IDLE_BACKOFF_MAX_SECONDS must be a number") from exc

    try:
        visibility_timeout = int(
            os.getenv("QUEUE_VISIBILITY_TIMEOUT_SECONDS", str(DEFAULT_VISIBILITY_TIMEOUT_SECONDS))
        )
    except ValueError as exc:
        raise RuntimeError("QUEUE_VISIBILITY_TIMEOUT_SECONDS must be an integer") from exc

    if visibility_timeout <= 0:
        raise RuntimeError("QUEUE_VISIBILITY_TIMEOUT_SECONDS must be greater than 0")

    if poll_interval_seconds <= 0:
        raise RuntimeError("WORKER_POLL_INTERVAL_SECONDS must be greater than 0")
    if idle_backoff_max_seconds < poll_interval_seconds:
        raise RuntimeError(
            "WORKER_IDLE_BACKOFF_MAX_SECONDS must be greater than or equal to WORKER_POLL_INTERVAL_SECONDS"
        )

    scorer_mode = os.getenv("SCORER_MODE", "mock").strip().lower()
    if scorer_mode not in SUPPORTED_SCORER_MODES:
        raise RuntimeError(
            f"Unsupported SCORER_MODE={scorer_mode!r}. "
            f"Supported scorer modes: {', '.join(SUPPORTED_SCORER_MODES)}"
        )

    return {
        "supabase_url": os.environ["SUPABASE_URL"],
        "supabase_service_role_key": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        "webhook_url": os.environ["NODE_WEBHOOK_URL"],
        "webhook_secret": os.environ["AI_WEBHOOK_SECRET"],
        "queue_name": os.getenv("QUEUE_NAME", "practice_jobs"),
        "practice_audio_bucket": os.getenv("PRACTICE_AUDIO_BUCKET", "practice-audios").strip() or "practice-audios",
        "scorer_mode": scorer_mode,
        "alignment_mode": os.getenv("ALIGNMENT_MODE", "fallback").strip().lower() or "fallback",
        "model_version": os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION).strip() or DEFAULT_MODEL_VERSION,
        "confidence_threshold": confidence_threshold,
        "worker_mode": worker_mode,
        "poll_interval_seconds": poll_interval_seconds,
        "idle_backoff_max_seconds": idle_backoff_max_seconds,
        "visibility_timeout": visibility_timeout,
    }


def _rpc_data(response: Any) -> Any:
    return getattr(response, "data", None)


def _call_rpc(client: Client, rpc_name: str, params: dict[str, Any] | None = None) -> Any:
    return client.rpc(rpc_name, params or {}).execute()


def _first_row(data: Any) -> dict[str, Any] | None:
    if not data:
        return None
    if isinstance(data, list):
        return data[0] if data else None
    if isinstance(data, dict):
        if "msg" in data or "message" in data or "job_id" in data:
            return data
        for value in data.values():
            if isinstance(value, list) and value:
                return value[0]
    return None


def _read_one_job(client: Client, queue_name: str, visibility_timeout: int) -> dict[str, Any] | None:
    rpc_attempts = [
        (
            "read_practice_job",
            {
                "p_queue_name": queue_name,
                "p_vt": visibility_timeout,
                "p_qty": 1,
            },
        ),
        ("read_practice_job", {}),
        (
            "pgmq_read",
            {
                "queue_name": queue_name,
                "vt": visibility_timeout,
                "qty": 1,
            },
        ),
        (
            "pgmq_read",
            {
                "p_queue_name": queue_name,
                "p_vt": visibility_timeout,
                "p_qty": 1,
            },
        ),
    ]

    errors = []
    for rpc_name, params in rpc_attempts:
        try:
            row = _first_row(_rpc_data(_call_rpc(client, rpc_name, params)))
            if row:
                return row
            print(f"[WARN] {rpc_name} returned no messages from {queue_name!r} (queue empty or message within visibility window).")
            return None
        except Exception as exc:
            errors.append(f"{rpc_name}: {exc}")

    raise RuntimeError(
        "Unable to read PGMQ job. Expected an exposed RPC such as "
        "read_practice_job or pgmq_read. Attempts: " + " | ".join(errors)
    )


def _archive_job(client: Client, queue_name: str, msg_id: int) -> None:
    rpc_attempts = [
        ("archive_practice_job", {"p_msg_id": msg_id}),
        ("archive_practice_job", {"msg_id": msg_id}),
        ("pgmq_archive", {"queue_name": queue_name, "msg_id": msg_id}),
        ("pgmq_archive", {"p_queue_name": queue_name, "p_msg_id": msg_id}),
    ]

    errors = []
    for rpc_name, params in rpc_attempts:
        try:
            _call_rpc(client, rpc_name, params)
            return
        except Exception as exc:
            errors.append(f"{rpc_name}: {exc}")

    raise RuntimeError(
        "Webhook succeeded, but the queue message could not be archived. "
        "Attempts: " + " | ".join(errors)
    )


def _parse_queue_row(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    msg_id = row.get("msg_id") or row.get("message_id") or row.get("id")
    message = row.get("message") or row.get("msg") or row.get("payload") or row

    if isinstance(message, str):
        import json

        message = json.loads(message)

    if not isinstance(message, dict):
        raise RuntimeError("Queue message payload must be a JSON object")

    missing = [
        name
        for name in ["job_id", "student_id", "target_word", "audio_url"]
        if not message.get(name)
    ]
    if missing:
        raise RuntimeError(f"Queue message missing fields: {', '.join(missing)}")

    if msg_id is None:
        raise RuntimeError("Queue row is missing msg_id")

    return int(msg_id), message


def _score(job: dict[str, Any], scorer_mode: str, confidence_threshold: float) -> dict[str, Any]:
    if scorer_mode == "mock":
        return score_mock_pronunciation(job, confidence_threshold)
    if scorer_mode == "wav2vec2":
        from scorers.wav2vec2_scorer import score_pronunciation as score_wav2vec2_pronunciation

        return score_wav2vec2_pronunciation(job, confidence_threshold)
    if scorer_mode == "cnn_attention":
        from app.scorers.cnn_attention_scorer import score_pronunciation as score_cnn_attention_pronunciation

        return score_cnn_attention_pronunciation(job, confidence_threshold)
    if scorer_mode == "cnn_attention_context":
        from app.scorers.cnn_attention_scorer import score_pronunciation_context

        return score_pronunciation_context(job, confidence_threshold)

    raise RuntimeError(
        f"Unsupported SCORER_MODE={scorer_mode!r}. "
        f"Supported scorer modes: {', '.join(SUPPORTED_SCORER_MODES)}"
    )


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, PhoenixWorkerError):
        return exc.error_type
    if isinstance(exc, TimeoutError):
        return "scorer_timeout"

    text = str(exc).lower()
    if "checkpoint" in text and any(term in text for term in ("not found", "missing", "no such file")):
        return "checkpoint_missing"
    if any(term in text for term in ("state_dict", "size mismatch", "missing key", "unexpected key")):
        return "checkpoint_incompatible"
    if "checkpoint" in text and any(term in text for term in ("invalid", "incompatible")):
        return "checkpoint_incompatible"
    if any(term in text for term in ("decode", "ffmpeg", "could not be decoded", "unable to decode")):
        return "audio_decode_failed"
    if any(term in text for term in ("preprocess", "feature", "log-mel", "melspectrogram", "librosa", "numpy")):
        return "audio_preprocess_failed"
    if any(term in text for term in ("alignment", "mfa", "textgrid", "forced-alignment")):
        return "alignment_failed"
    if any(term in text for term in ("timeout", "timed out")):
        return "scorer_timeout"
    if any(term in text for term in ("scorer", "inference", "torch", "model")):
        return "scorer_failed"
    return "unknown_error"


def _sanitize_error_text(value: str) -> str:
    sanitized = str(value or "")
    sanitized = re.sub(r"[A-Za-z]:\\[^:;\r\n]+", "[redacted-local-path]", sanitized)
    sanitized = re.sub(r"/tmp/[^\s:;]+", "[redacted-local-path]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"https?://\S+", "[redacted-url]", sanitized, flags=re.IGNORECASE)
    normalized = sanitized.lower()
    if any(marker in normalized for marker in SENSITIVE_VALUE_MARKERS):
        return (
            "AI worker encountered a local-artifact or signed-URL related error. "
            "Inspect local configuration without exposing local paths or secrets."
        )
    return sanitized


def _checkpoint_status() -> dict[str, Any]:
    configured_value = str(os.getenv("CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH") or "").strip()
    checkpoint_exists = False
    if configured_value:
        try:
            checkpoint_exists = Path(configured_value).expanduser().exists()
        except OSError:
            checkpoint_exists = False
    return {
        "checkpoint_configured": bool(configured_value),
        "checkpoint_exists": checkpoint_exists,
    }


def _preflight_scorer_config(config: dict[str, Any]) -> None:
    if config["scorer_mode"] != "cnn_attention_context":
        return

    checkpoint_path = str(os.getenv("CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH") or "").strip()
    if not checkpoint_path:
        raise PhoenixWorkerError(
            "CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH is required for SCORER_MODE=cnn_attention_context.",
            "checkpoint_missing",
        )

    try:
        checkpoint_exists = Path(checkpoint_path).expanduser().exists()
    except OSError as exc:
        raise PhoenixWorkerError(
            "CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH could not be resolved.",
            "checkpoint_missing",
        ) from exc

    if not checkpoint_exists:
        raise PhoenixWorkerError(
            "CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH does not point to an existing local checkpoint.",
            "checkpoint_missing",
        )


def _extract_model_confidence(result: dict[str, Any]) -> Any:
    feedback_confidence = result.get("feedback", {}).get("model_confidence")
    if isinstance(feedback_confidence, dict):
        return feedback_confidence.get("value")
    if feedback_confidence is not None:
        return feedback_confidence
    return result.get("diagnosis", {}).get("diagnosis_confidence")


def _details_from_result(result: dict[str, Any]) -> list[Any]:
    feedback_details = result.get("feedback", {}).get("details")
    if isinstance(feedback_details, list):
        return feedback_details
    diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
    top_issues = diagnosis.get("top_issues")
    if isinstance(top_issues, list):
        return top_issues
    segments = result.get("segments")
    if isinstance(segments, list):
        return segments[:5]
    return []


def _warnings_from_result(result: dict[str, Any]) -> list[str]:
    feedback_warnings = result.get("feedback", {}).get("warnings")
    if isinstance(feedback_warnings, list):
        return [str(warning) for warning in feedback_warnings]
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    warnings = metadata.get("warnings")
    if isinstance(warnings, list):
        return [str(warning) for warning in warnings]
    warning = metadata.get("warning") or metadata.get("alignment_note")
    return [str(warning)] if warning else []


def _normalize_result_for_webhook(
    result: dict[str, Any],
    *,
    config: dict[str, Any],
    error_type: str | None = None,
) -> dict[str, Any]:
    normalized = dict(result)
    status = normalized.get("status")
    if status not in {"completed", "failed"}:
        status = "failed"
        error_type = error_type or "scorer_failed"
    normalized["status"] = status
    normalized["problem_phonemes"] = list(normalized.get("problem_phonemes") or [])

    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    scorer = normalized.get("scorer") if isinstance(normalized.get("scorer"), dict) else {}
    feedback = normalized.get("feedback") if isinstance(normalized.get("feedback"), dict) else {}

    alignment_method = (
        metadata.get("alignment_method")
        or metadata.get("requested_alignment_method")
        or metadata.get("requested_alignment_mode")
        or config["alignment_mode"]
    )
    is_forced_alignment = metadata.get("is_forced_alignment")
    if is_forced_alignment is None:
        is_forced_alignment = bool(metadata.get("mfa_used") and alignment_method == "mfa")

    feedback.setdefault("model_version", config["model_version"])
    feedback.setdefault("scorer_mode", scorer.get("name") or config["scorer_mode"])
    feedback.setdefault("alignment_method", alignment_method)
    feedback.setdefault("is_forced_alignment", is_forced_alignment)
    feedback.setdefault("summary", "Phoenix v2 completed pronunciation scoring.")
    feedback.setdefault("details", _details_from_result(normalized))
    feedback.setdefault("warnings", _warnings_from_result(normalized))

    model_confidence = _extract_model_confidence(normalized)
    if model_confidence is not None:
        feedback.setdefault("model_confidence", model_confidence)

    if status == "failed":
        resolved_error_type = error_type or metadata.get("error_type") or feedback.get("error_type") or "unknown_error"
        if resolved_error_type not in STANDARD_ERROR_TYPES:
            resolved_error_type = "unknown_error"
        normalized["score"] = None
        normalized["score_type"] = "unavailable"
        normalized["problem_phonemes"] = []
        feedback["error_type"] = resolved_error_type
        feedback.setdefault("summary", "Phoenix v2 could not produce a model score for this attempt.")
        feedback.setdefault("details", [])
        feedback.setdefault("warnings", ["No fallback score was generated."])
        metadata["error_type"] = resolved_error_type

    metadata.setdefault("model_version", config["model_version"])
    metadata.setdefault("scorer_mode", config["scorer_mode"])
    metadata.setdefault("alignment_method", alignment_method)
    metadata.setdefault("is_forced_alignment", is_forced_alignment)
    normalized["metadata"] = metadata
    normalized["feedback"] = feedback
    return normalized


def _build_failed_result(
    job: dict[str, Any],
    config: dict[str, Any],
    confidence_threshold: float,
    error: Exception,
) -> dict[str, Any]:
    sanitized_error = _sanitize_error_text(str(error))
    error_type = _classify_exception(error)
    scorer_mode = config["scorer_mode"]
    result = build_failed_ai_result(
        error=sanitized_error,
        scorer={
            "name": scorer_mode,
            "type": "phone_error_classifier" if scorer_mode in {"cnn_attention", "cnn_attention_context"} else scorer_mode,
            "version": (
                "cnn_attention_context_0_10_phase2_candidate"
                if scorer_mode == "cnn_attention_context"
                else "cnn_attention_selected_baseline" if scorer_mode == "cnn_attention" else "unknown"
            ),
        },
        metadata={
            "confidence_threshold": confidence_threshold,
            "target_word": str(job.get("target_word") or ""),
            "error": sanitized_error,
            "error_type": error_type,
            "model_version": config["model_version"],
            "scorer_mode": scorer_mode,
            "alignment_method": config["alignment_mode"],
            "is_forced_alignment": False,
        },
    )
    result["feedback"]["model_version"] = config["model_version"]
    result["feedback"]["scorer_mode"] = scorer_mode
    result["feedback"]["alignment_method"] = config["alignment_mode"]
    result["feedback"]["is_forced_alignment"] = False
    result["feedback"]["error_type"] = error_type
    result["feedback"]["summary"] = "Phoenix v2 could not produce a model score for this attempt."
    result["feedback"]["details"] = []
    result["feedback"]["warnings"] = ["No fallback score was generated."]
    result["feedback"]["diagnosis"] = result["diagnosis"]
    result["feedback"]["scorer"] = result["scorer"]
    result["feedback"]["metadata"] = result["metadata"]
    return _normalize_result_for_webhook(result, config=config, error_type=error_type)


def _build_validation_failure_webhook_payload(job_id: str) -> dict[str, Any]:
    """Return a static terminal payload even when an internal result is invalid.

    Do not reuse the invalid scorer result here: its metadata may be exactly
    what made validation fail.  The minimal failed contract is safe for the
    backend and keeps the queue from leaving the student's job in processing.
    """
    return build_failed_webhook_payload(
        job_id,
        "AI worker could not validate the scoring result. Please retry.",
    )


def _post_webhook(webhook_url: str, webhook_secret: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        webhook_url,
        json=payload,
        headers={"x-ai-webhook-secret": webhook_secret},
        timeout=30,
    )


@contextmanager
def _prepared_audio_for_job(
    job: dict[str, Any],
    *,
    storage_client: Client | None = None,
    practice_audio_bucket: str = "practice-audios",
):
    """Prepare queued audio, supporting stable Storage keys and legacy URLs."""
    from app.alignment.audio_preparation import prepare_audio_for_mfa

    with tempfile.TemporaryDirectory(prefix="worker-prepared-") as directory:
        root = Path(directory)
        try:
            resolved = resolve_audio_reference(
                job,
                storage_client=storage_client,
                practice_audio_bucket=practice_audio_bucket,
                temp_dir=root,
            )
        except AudioReferenceError as exc:
            raise PhoenixWorkerError(str(exc), "audio_preprocess_failed") from exc

        prepared = prepare_audio_for_mfa(resolved.path, root / "prepared.wav")
        yield prepared


def _audio_quality_gate(audio_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Check audio quality before running the scorer.

    Returns a pre-built failed result plus safe diagnostics if quality is below
    threshold, otherwise ``None`` plus diagnostics for downstream metadata.
    Any exception raised here is caught by the caller, which logs a warning
    and falls through to normal scoring rather than failing the job entirely.
    """
    if config.get("scorer_mode") == "mock":
        return None, {"audio_quality_status": "not_checked"}

    try:
        snr_threshold = float(os.getenv("AUDIO_SNR_REJECT_THRESHOLD_DB", str(AUDIO_SNR_REJECT_THRESHOLD_DB)))
    except ValueError:
        snr_threshold = AUDIO_SNR_REJECT_THRESHOLD_DB
    try:
        min_voiced = float(os.getenv("AUDIO_MIN_VOICED_DURATION_SECONDS", str(AUDIO_MIN_VOICED_DURATION_SECONDS)))
    except ValueError:
        min_voiced = AUDIO_MIN_VOICED_DURATION_SECONDS

    from app.audio.preprocessing import estimate_snr

    gate_start = time.monotonic()
    quality = estimate_snr(audio_path)
    gate_elapsed = time.monotonic() - gate_start

    snr_db = quality.get("snr_db")
    voiced_duration = float(quality.get("voiced_duration_seconds") or 0.0)
    voiced_frames_ratio = float(quality.get("voiced_frames_ratio") or 0.0)
    unvoiced_frames_ratio = float(quality.get("unvoiced_frames_ratio") or 0.0)
    mean_voiced_prob = float(quality.get("mean_voiced_prob") or 0.0)
    speech_detection_mode = str(quality.get("speech_detection_mode") or "unknown")
    audio_quality_status = str(quality.get("audio_quality_status") or "invalid")
    largest_unvoiced_run = float(quality.get("largest_unvoiced_run_seconds") or 0.0)

    # Primary gate: VAD-derived speech duration. SNR remains diagnostic only.
    rejection_reason: str | None = None
    if audio_quality_status == "invalid":
        if speech_detection_mode == "no_voiced_anchor" and unvoiced_frames_ratio > 0:
            rejection_reason = "insufficient_sustained_unvoiced_activity"
        elif speech_detection_mode == "no_voiced_anchor":
            rejection_reason = "no_voiced_anchor"
        else:
            rejection_reason = "insufficient_detected_speech"
    elif speech_detection_mode != "pitch_degraded_unvoiced_only" and voiced_duration < min_voiced:
        rejection_reason = "insufficient_detected_speech"

    print(
        f"audio_quality_check snr_db={snr_db} "
        f"file_duration={float(quality.get('file_duration_seconds') or 0.0):.3f}s "
        f"voiced_anchor_duration={float(quality.get('voiced_anchor_duration_seconds') or 0.0):.3f}s "
        f"detected_speech_duration={voiced_duration:.3f}s "
        f"voiced_frames_ratio={voiced_frames_ratio:.3f} "
        f"unvoiced_frames_ratio={unvoiced_frames_ratio:.3f} "
        f"mean_voiced_prob={mean_voiced_prob:.3f} "
        f"finite_f0_frames={int(quality.get('finite_f0_frames') or 0)} "
        f"finite_f0_ratio={float(quality.get('finite_f0_ratio') or 0.0):.3f} "
        f"f0_min_hz={quality.get('f0_min_hz')} f0_max_hz={quality.get('f0_max_hz')} "
        f"pyin_voiced_flag_ratio={float(quality.get('pyin_voiced_flag_ratio') or 0.0):.3f} "
        f"voiced_candidate_count={int(quality.get('voiced_candidate_count') or 0)} "
        f"unvoiced_candidate_count={int(quality.get('unvoiced_candidate_count') or 0)} "
        f"largest_unvoiced_run_seconds={largest_unvoiced_run:.3f} "
        f"speech_detection_mode={speech_detection_mode} audio_quality_status={audio_quality_status} "
        f"gate_elapsed_s={gate_elapsed:.2f} "
        f"min_voiced={min_voiced}s snr_threshold_diag={snr_threshold} "
        f"rejected={rejection_reason is not None} reason={rejection_reason}"
    )

    if rejection_reason is None:
        quality["gate_elapsed_seconds"] = round(gate_elapsed, 3)
        return None, quality

    scorer_mode = config.get("scorer_mode", "unknown")
    from app.contracts.ai_result_contract import build_failed_ai_result as _build_failed
    failed = _build_failed(
        error=f"audio_quality_rejected:{rejection_reason}",
        scorer={
            "name": scorer_mode,
            "type": "phone_error_classifier",
            "version": "audio_quality_gate_v3",
        },
        metadata={
            "error_type": "audio_quality_rejected",
            "rejection_reason": rejection_reason,
            "snr_db": snr_db,
            **quality,
            "snr_threshold_db": snr_threshold,
            "min_voiced_duration_seconds": min_voiced,
            "gate_elapsed_seconds": round(gate_elapsed, 3),
            "model_version": config.get("model_version", "unknown"),
            "scorer_mode": scorer_mode,
            "alignment_method": config.get("alignment_mode", "unknown"),
            "is_forced_alignment": False,
        },
    )
    failed["feedback"]["error_type"] = "audio_quality_rejected"
    failed["feedback"]["summary"] = "Âm thanh không rõ, vui lòng thu âm lại ở nơi yên tĩnh hơn."
    failed["feedback"]["tips"] = [
        "Di chuyển đến nơi yên tĩnh hơn và thử lại.",
        "Giữ micro gần miệng và nói rõ ràng hơn.",
        "Đảm bảo không có tiếng ồn lớn xung quanh khi ghi âm.",
    ]
    return failed, quality


def _process_one_job(client: Client, config: dict[str, Any]) -> bool:
    row = _read_one_job(client, config["queue_name"], config["visibility_timeout"])
    if not row:
        print(f"No job found in {config['queue_name']} queue.")
        return False

    msg_id, job = _parse_queue_row(row)
    print(f"msg_id={msg_id}")
    print(f"job_id={job['job_id']}")
    print(f"target_word={job['target_word']}")
    print(f"scorer_mode={config['scorer_mode']}")
    print(f"alignment_mode={config['alignment_mode']}")
    print(f"model_version={config['model_version']}")

    try:
        _preflight_scorer_config(config)
        quality_rejection = None
        scored_result: dict[str, Any] | None = None
        if config["scorer_mode"] == "mock":
            quality_rejection, _ = _audio_quality_gate(Path(), config)
        else:
            with _prepared_audio_for_job(
                job,
                storage_client=client,
                practice_audio_bucket=config["practice_audio_bucket"],
            ) as prepared_audio:
                scoring_job = {
                    **job,
                    "audio_path": str(prepared_audio.path),
                    "_prepared_audio": prepared_audio,
                }
                try:
                    quality_rejection, audio_quality = _audio_quality_gate(prepared_audio.path, config)
                except Exception as qe:
                    print(f"[WARN] Audio quality gate failed, proceeding with scoring. error={_sanitize_error_text(str(qe))}")
                    audio_quality = {"audio_quality_status": "not_checked"}
                if quality_rejection is None:
                    scoring_job["_audio_quality_warnings"] = [audio_quality["warning"]] if audio_quality.get("warning") else []
                    scored_result = _score(
                        scoring_job,
                        config["scorer_mode"],
                        config["confidence_threshold"],
                    )
        if quality_rejection is not None:
            result = _normalize_result_for_webhook(
                quality_rejection, config=config, error_type="audio_quality_rejected"
            )
        elif scored_result is not None:
            result = _normalize_result_for_webhook(scored_result, config=config)
        else:
            result = _normalize_result_for_webhook(
                _score(job, config["scorer_mode"], config["confidence_threshold"]),
                config=config,
            )
    except Exception as exc:
        sanitized_error = _sanitize_error_text(str(exc))
        error_type = _classify_exception(exc)
        print(
            f"Scoring failed for job_id={job['job_id']} "
            f"scorer_mode={config['scorer_mode']} error_type={error_type}: {sanitized_error}"
        )
        if config["scorer_mode"] == "cnn_attention_context":
            checkpoint_status = _checkpoint_status()
            print(f"checkpoint_configured={checkpoint_status['checkpoint_configured']}")
            print(f"checkpoint_exists={checkpoint_status['checkpoint_exists']}")
            print(f"checkpoint_error={sanitized_error}")
        result = _build_failed_result(
            job,
            config,
            config["confidence_threshold"],
            PhoenixWorkerError(sanitized_error, error_type),
        )

    confidence = _extract_model_confidence(result)
    print(f"model_confidence={confidence if confidence is not None else 'unavailable'}")
    print(f"result_status={result.get('status')}")
    if result.get("status") == "failed":
        print(f"error_type={result.get('feedback', {}).get('error_type', 'unknown_error')}")

    if result.get("status") == "failed":
        webhook_payload = build_failed_webhook_payload(
            job["job_id"],
            _sanitize_error_text(
                str(
                    result.get("metadata", {}).get("error")
                    or "AI scoring failed."
                )
            ),
            result,
        )
    else:
        webhook_payload = build_success_webhook_payload(
            job["job_id"],
            result,
        )

    result_metadata = (
        result.get("metadata")
        if isinstance(result.get("metadata"), dict)
        else {}
    )
    webhook_feedback = (
        webhook_payload.get("feedback")
        if isinstance(webhook_payload.get("feedback"), dict)
        else {}
    )
    webhook_ai_result = (
        webhook_feedback.get("ai_result")
        if isinstance(webhook_feedback.get("ai_result"), dict)
        else {}
    )

    print(
        "score_trace "
        f"job_id={job['job_id']} "
        f"result_status={result.get('status')!r} "
        f"result_score={result.get('score')!r} "
        f"result_score_type={result.get('score_type')!r} "
        f"webhook_status={webhook_payload.get('status')!r} "
        f"webhook_score={webhook_payload.get('score')!r} "
        f"feedback_ai_result_score={webhook_ai_result.get('score')!r} "
        f"pronunciation_score_source="
        f"{result.get('pronunciation_score_source') or result_metadata.get('pronunciation_score_source')!r} "
        f"model_confidence={_extract_model_confidence(result)!r}"
    )

    payload_is_valid, payload_issues = validate_webhook_payload(
        webhook_payload
    )
    if not payload_is_valid:
        print(
            "Webhook payload validation failed: "
            + " | ".join(payload_issues)
        )
        print("webhook_payload_valid=false")
        print(
            "reason=payload_validation_failed; "
            "sending terminal safety payload"
        )
        webhook_payload = _build_validation_failure_webhook_payload(
            job["job_id"]
        )
        safety_payload_is_valid, safety_payload_issues = (
            validate_webhook_payload(webhook_payload)
        )
        print(
            "safety_webhook_payload_valid="
            f"{'true' if safety_payload_is_valid else 'false'}"
        )
        if not safety_payload_is_valid:
            print(
                "Safety payload validation unexpectedly failed: "
                + " | ".join(safety_payload_issues)
            )
    else:
        print("webhook_payload_valid=true")

    try:
        response = _post_webhook(
            config["webhook_url"],
            config["webhook_secret"],
            webhook_payload,
        )
    except requests.RequestException as exc:
        sanitized_error = _sanitize_error_text(str(exc))
        print(
            f"Webhook request failed. Message {msg_id} was not archived. "
            f"error_type=webhook_failed error={sanitized_error}"
        )
        return False

    print(f"webhook_status_code={response.status_code}")
    response_preview = _sanitize_error_text(
        str(response.text or "")[:500]
    )
    print(f"webhook_response_body={response_preview}")

    if not 200 <= response.status_code < 300:
        print(
            f"Webhook failed. Message {msg_id} was not archived. "
            "error_type=webhook_failed"
        )
        print(response_preview)
        return False

    try:
        _archive_job(client, config["queue_name"], msg_id)
    except Exception as exc:
        print(f"archive_success=false msg_id={msg_id} error={_sanitize_error_text(str(exc))}")
        return False
    print(f"archive_success=true msg_id={msg_id}")
    print(f"Processed job {job['job_id']} and archived message {msg_id}.")
    return True


def main() -> int:
    config = _load_env()
    client = create_client(config["supabase_url"], config["supabase_service_role_key"])
    print(f"Supported scorer modes: {', '.join(SUPPORTED_SCORER_MODES)}")
    print(f"worker_mode={config['worker_mode']}")
    print(f"scorer_mode={config['scorer_mode']}")
    print(f"alignment_mode={config['alignment_mode']}")
    print(f"model_version={config['model_version']}")

    if config["worker_mode"] == "once":
        try:
            _process_one_job(client, config)
        except Exception as exc:
            print(f"Worker once job failed without archive. error={_sanitize_error_text(str(exc))}")
        return 0

    print(
        "Worker loop started "
        f"queue={config['queue_name']} "
        f"poll_interval_seconds={config['poll_interval_seconds']} "
        f"idle_backoff_max_seconds={config['idle_backoff_max_seconds']}"
    )

    idle_sleep_seconds = config["poll_interval_seconds"]
    while True:
        try:
            processed = _process_one_job(client, config)
        except Exception as exc:
            print(f"Worker loop job failed without archive. error={_sanitize_error_text(str(exc))}")
            processed = False
        if processed:
            idle_sleep_seconds = config["poll_interval_seconds"]
            continue

        time.sleep(idle_sleep_seconds)
        idle_sleep_seconds = min(
            config["idle_backoff_max_seconds"],
            idle_sleep_seconds + config["poll_interval_seconds"],
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Worker stopped by Ctrl+C.")
        raise SystemExit(0)
    except Exception as exc:
        print(f"Worker failed: {exc}", file=sys.stderr)
        raise SystemExit(1)