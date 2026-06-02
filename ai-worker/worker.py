from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from supabase import Client, create_client

from app.contracts.ai_result_contract import build_failed_ai_result
from app.contracts.webhook_payload import (
    build_failed_webhook_payload,
    build_success_webhook_payload,
    validate_webhook_payload,
)
from scorers.mock_scorer import score_pronunciation as score_mock_pronunciation


DEFAULT_VISIBILITY_TIMEOUT_SECONDS = 60
DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_WORKER_IDLE_BACKOFF_MAX_SECONDS = 10.0
SUPPORTED_SCORER_MODES = ("mock", "wav2vec2", "cnn_attention", "cnn_attention_context")


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
        "scorer_mode": scorer_mode,
        "confidence_threshold": confidence_threshold,
        "worker_mode": worker_mode,
        "poll_interval_seconds": poll_interval_seconds,
        "idle_backoff_max_seconds": idle_backoff_max_seconds,
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


def _read_one_job(client: Client, queue_name: str) -> dict[str, Any] | None:
    rpc_attempts = [
        (
            "read_practice_job",
            {
                "p_queue_name": queue_name,
                "p_vt": DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
                "p_qty": 1,
            },
        ),
        ("read_practice_job", {}),
        (
            "pgmq_read",
            {
                "queue_name": queue_name,
                "vt": DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
                "qty": 1,
            },
        ),
        (
            "pgmq_read",
            {
                "p_queue_name": queue_name,
                "p_vt": DEFAULT_VISIBILITY_TIMEOUT_SECONDS,
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


def _build_failed_result(
    job: dict[str, Any],
    scorer_mode: str,
    confidence_threshold: float,
    error: Exception,
) -> dict[str, Any]:
    result = build_failed_ai_result(
        error=str(error),
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
        },
    )
    result["feedback"]["diagnosis"] = result["diagnosis"]
    result["feedback"]["scorer"] = result["scorer"]
    result["feedback"]["metadata"] = result["metadata"]
    return result


def _post_webhook(webhook_url: str, webhook_secret: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        webhook_url,
        json=payload,
        headers={"x-ai-webhook-secret": webhook_secret},
        timeout=30,
    )


def _process_one_job(client: Client, config: dict[str, Any]) -> bool:
    row = _read_one_job(client, config["queue_name"])
    if not row:
        print(f"No job found in {config['queue_name']} queue.")
        return False

    msg_id, job = _parse_queue_row(row)
    print(f"msg_id={msg_id}")
    print(f"job_id={job['job_id']}")
    print(f"target_word={job['target_word']}")
    print(f"scorer_mode={config['scorer_mode']}")

    try:
        result = _score(job, config["scorer_mode"], config["confidence_threshold"])
    except Exception as exc:
        print(f"Scoring failed for job_id={job['job_id']} scorer_mode={config['scorer_mode']}: {exc}")
        result = _build_failed_result(
            job,
            config["scorer_mode"],
            config["confidence_threshold"],
            exc,
        )

    confidence = (
        result.get("feedback", {}).get("model_confidence", {}).get("value")
        or result.get("diagnosis", {}).get("diagnosis_confidence")
    )
    print(f"model_confidence={confidence if confidence is not None else 'unavailable'}")

    if result.get("status") == "failed":
        webhook_payload = build_failed_webhook_payload(
            job["job_id"],
            str(result.get("metadata", {}).get("error") or "AI scoring failed."),
            result,
        )
    else:
        webhook_payload = build_success_webhook_payload(job["job_id"], result)

    payload_is_valid, payload_issues = validate_webhook_payload(webhook_payload)
    if not payload_is_valid:
        print("Webhook payload validation failed: " + " | ".join(payload_issues))
        webhook_payload = build_failed_webhook_payload(
            job["job_id"],
            "AI worker produced an invalid webhook payload: " + " | ".join(payload_issues),
            result,
        )

    try:
        response = _post_webhook(
            config["webhook_url"],
            config["webhook_secret"],
            webhook_payload,
        )
    except requests.RequestException as exc:
        print(f"Webhook request failed. Message {msg_id} was not archived. error={exc}")
        return False

    print(f"webhook_status_code={response.status_code}")

    if not 200 <= response.status_code < 300:
        print(f"Webhook failed. Message {msg_id} was not archived.")
        print(response.text)
        return False

    _archive_job(client, config["queue_name"], msg_id)
    print(f"Processed job {job['job_id']} and archived message {msg_id}.")
    return True


def main() -> int:
    config = _load_env()
    client = create_client(config["supabase_url"], config["supabase_service_role_key"])
    print(f"Supported scorer modes: {', '.join(SUPPORTED_SCORER_MODES)}")
    print(f"worker_mode={config['worker_mode']}")
    print(f"scorer_mode={config['scorer_mode']}")

    if config["worker_mode"] == "once":
        _process_one_job(client, config)
        return 0

    print(
        "Worker loop started "
        f"queue={config['queue_name']} "
        f"poll_interval_seconds={config['poll_interval_seconds']} "
        f"idle_backoff_max_seconds={config['idle_backoff_max_seconds']}"
    )

    idle_sleep_seconds = config["poll_interval_seconds"]
    while True:
        processed = _process_one_job(client, config)
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
