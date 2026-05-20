from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from supabase import Client, create_client

from scorers.mock_scorer import score_pronunciation
from scorers.wav2vec2_scorer import score_pronunciation as score_wav2vec2_pronunciation


DEFAULT_VISIBILITY_TIMEOUT_SECONDS = 60


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

    try:
        audio_download_timeout_seconds = float(
            os.getenv("AUDIO_DOWNLOAD_TIMEOUT_SECONDS", "30")
        )
        poll_interval_seconds = float(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "1"))
        idle_backoff_max_seconds = float(
            os.getenv("WORKER_IDLE_BACKOFF_MAX_SECONDS", "10")
        )
        batch_size = int(os.getenv("WORKER_BATCH_SIZE", "1"))
        max_jobs_per_run = int(os.getenv("WORKER_MAX_JOBS_PER_RUN", "0"))
    except ValueError as exc:
        raise RuntimeError("Worker numeric env vars must be valid numbers") from exc

    worker_mode = os.getenv("WORKER_MODE", "once").strip().lower()
    if worker_mode not in {"once", "loop"}:
        raise RuntimeError("WORKER_MODE must be 'once' or 'loop'")
    if poll_interval_seconds <= 0:
        raise RuntimeError("WORKER_POLL_INTERVAL_SECONDS must be greater than 0")
    if idle_backoff_max_seconds < poll_interval_seconds:
        raise RuntimeError(
            "WORKER_IDLE_BACKOFF_MAX_SECONDS must be greater than or equal to "
            "WORKER_POLL_INTERVAL_SECONDS"
        )
    if batch_size < 1:
        raise RuntimeError("WORKER_BATCH_SIZE must be at least 1")
    if max_jobs_per_run < 0:
        raise RuntimeError("WORKER_MAX_JOBS_PER_RUN must be 0 or greater")

    return {
        "supabase_url": os.environ["SUPABASE_URL"],
        "supabase_service_role_key": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        "webhook_url": os.environ["NODE_WEBHOOK_URL"],
        "webhook_secret": os.environ["AI_WEBHOOK_SECRET"],
        "queue_name": os.getenv("QUEUE_NAME", "practice_jobs"),
        "scorer_mode": os.getenv("SCORER_MODE", "mock"),
        "confidence_threshold": confidence_threshold,
        "wav2vec2_model_name": os.getenv(
            "WAV2VEC2_MODEL_NAME",
            "facebook/wav2vec2-base-960h",
        ),
        "audio_download_timeout_seconds": audio_download_timeout_seconds,
        "worker_mode": worker_mode,
        "poll_interval_seconds": poll_interval_seconds,
        "idle_backoff_max_seconds": idle_backoff_max_seconds,
        "batch_size": batch_size,
        "max_jobs_per_run": max_jobs_per_run,
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


def _score(job: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    scorer_mode = config["scorer_mode"]
    if scorer_mode == "mock":
        return score_pronunciation(job, config["confidence_threshold"])
    if scorer_mode == "wav2vec2":
        return score_wav2vec2_pronunciation(
            job,
            confidence_threshold=config["confidence_threshold"],
            model_name=config["wav2vec2_model_name"],
            audio_download_timeout_seconds=config["audio_download_timeout_seconds"],
        )

    raise RuntimeError(
        f"Unsupported SCORER_MODE={scorer_mode!r}. Use 'mock' or 'wav2vec2'."
    )


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
        return False

    msg_id, job = _parse_queue_row(row)
    print(f"msg_id={msg_id}")
    print(f"job_id={job['job_id']}")
    print(f"target_word={job['target_word']}")
    print(f"scorer_mode={config['scorer_mode']}")
    if config["scorer_mode"] == "wav2vec2":
        print(f"wav2vec2_model_name={config['wav2vec2_model_name']}")

    result = _score(job, config)
    confidence = result.get("feedback", {}).get("model_confidence", {}).get("value")
    print(f"result_status={result.get('status')}")
    print(f"model_confidence={confidence}")

    webhook_payload = {"job_id": job["job_id"], **result}
    response = _post_webhook(
        config["webhook_url"],
        config["webhook_secret"],
        webhook_payload,
    )
    print(f"webhook_status_code={response.status_code}")

    if not 200 <= response.status_code < 300:
        print(f"Webhook failed. Message {msg_id} was not archived.")
        print(response.text)
        raise RuntimeError(f"Webhook failed with status {response.status_code}")

    _archive_job(client, config["queue_name"], msg_id)
    print(f"Processed job {job['job_id']} and archived message {msg_id}.")
    return True


def _run_once(client: Client, config: dict[str, Any]) -> int:
    processed = _process_one_job(client, config)
    if not processed:
        print(f"No job found in {config['queue_name']} queue.")
    return 0


def _run_loop(client: Client, config: dict[str, Any]) -> int:
    processed_jobs = 0
    idle_sleep_seconds = config["poll_interval_seconds"]
    last_idle_log_time = 0.0

    print(
        "Worker loop started "
        f"queue={config['queue_name']} "
        f"poll_interval={config['poll_interval_seconds']}s "
        f"max_idle_backoff={config['idle_backoff_max_seconds']}s "
        f"batch_size={config['batch_size']} "
        f"max_jobs_per_run={config['max_jobs_per_run']}"
    )

    while True:
        processed_in_batch = 0
        for _ in range(config["batch_size"]):
            if (
                config["max_jobs_per_run"]
                and processed_jobs >= config["max_jobs_per_run"]
            ):
                print(f"Reached WORKER_MAX_JOBS_PER_RUN={processed_jobs}.")
                return 0

            try:
                if not _process_one_job(client, config):
                    break
            except Exception as exc:
                print(f"Job processing failed without archiving message: {exc}")
                break

            processed_jobs += 1
            processed_in_batch += 1

        if processed_in_batch:
            idle_sleep_seconds = config["poll_interval_seconds"]
            continue

        now = time.monotonic()
        if now - last_idle_log_time >= 30 or last_idle_log_time == 0:
            print(
                f"No job found in {config['queue_name']} queue. "
                f"Sleeping {idle_sleep_seconds:g}s."
            )
            last_idle_log_time = now

        time.sleep(idle_sleep_seconds)
        idle_sleep_seconds = min(
            idle_sleep_seconds * 2,
            config["idle_backoff_max_seconds"],
        )


def main() -> int:
    config = _load_env()
    client = create_client(config["supabase_url"], config["supabase_service_role_key"])

    if config["worker_mode"] == "loop":
        return _run_loop(client, config)

    return _run_once(client, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Worker stopped by Ctrl+C.")
        raise SystemExit(0)
    except Exception as exc:
        print(f"Worker failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
