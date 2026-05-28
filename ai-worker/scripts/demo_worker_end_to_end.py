from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_contract import build_ai_result, build_failed_ai_result  # noqa: E402
from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.webhook_payload import (  # noqa: E402
    build_failed_webhook_payload,
    build_success_webhook_payload,
    validate_webhook_payload,
)


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end AI Worker demo without queue access.")
    parser.add_argument("--audio-path", default=None, help="Optional local audio path.")
    parser.add_argument("--target-text", default=None, help="Optional prompt text.")
    parser.add_argument("--target-word", default=None, help="Optional target word.")
    parser.add_argument("--job-id", default=None, help="Optional UUID job id.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Build payload without POSTing. Default.")
    parser.add_argument("--post", action="store_true", help="POST webhook payload to configured backend.")
    return parser.parse_args()


def _write_temp_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"".join(struct.pack("<h", 0) for _ in range(16000)))


def _scorer_mode() -> str:
    return os.getenv("SCORER_MODE", "mock").strip().lower() or "mock"


def _confidence_threshold() -> float:
    try:
        return float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.65"))
    except ValueError:
        return 0.65


def _score_with_worker_path(job: dict[str, Any], scorer_mode: str, confidence_threshold: float) -> dict[str, Any]:
    try:
        import worker

        return worker._score(job, scorer_mode, confidence_threshold)  # noqa: SLF001
    except Exception as exc:
        try:
            if scorer_mode == "mock":
                from scorers.mock_scorer import score_pronunciation as score_mock_pronunciation

                return score_mock_pronunciation(job, confidence_threshold)
            if scorer_mode == "cnn_attention":
                from app.scorers.cnn_attention_scorer import score_pronunciation as score_cnn_attention_pronunciation

                return score_cnn_attention_pronunciation(job, confidence_threshold)
        except Exception as scorer_exc:
            exc = scorer_exc

        return build_failed_ai_result(
            error=str(exc),
            scorer={
                "name": scorer_mode,
                "type": "phone_error_classifier" if scorer_mode == "cnn_attention" else scorer_mode,
                "version": "cnn_attention_selected_baseline" if scorer_mode == "cnn_attention" else "unknown",
            },
            metadata={
                "scorer_mode": scorer_mode,
                "demo_failure": True,
            },
        )


def _normalized_result(result: dict[str, Any], scorer_mode: str) -> dict[str, Any]:
    if {"diagnosis", "scorer", "metadata"}.issubset(result.keys()):
        return result

    feedback = result.get("feedback") if isinstance(result.get("feedback"), dict) else {}
    confidence = None
    model_confidence = feedback.get("model_confidence") if isinstance(feedback.get("model_confidence"), dict) else {}
    if isinstance(model_confidence, dict):
        confidence = model_confidence.get("value")

    return build_ai_result(
        score=result.get("score"),
        score_note="Demo/mock score. Classifier confidence is not pronunciation score.",
        pronunciation_score_source="mock_demo",
        problem_phonemes=result.get("problem_phonemes") if isinstance(result.get("problem_phonemes"), list) else [],
        predicted_error_type="unknown",
        diagnosis_confidence=confidence,
        feedback={
            "summary": str(feedback.get("summary") or ""),
            "tips": list(feedback.get("tips") or []),
        },
        scorer={
            "name": scorer_mode,
            "type": scorer_mode,
            "version": "demo",
        },
        metadata={
            "model_output_is_scoring": False,
            "alignment_used": False,
            "gop_used": False,
            "hybrid_used": False,
            "is_demo_score": True,
            "score_note": "Demo/mock score. Classifier confidence is not pronunciation score.",
        },
    )


def _build_job(args: argparse.Namespace, audio_path: Path) -> dict[str, Any]:
    target_text = args.target_text or args.target_word or "example"
    target_word = args.target_word or target_text.split()[0]
    return {
        "job_id": args.job_id or str(uuid.uuid4()),
        "student_id": "demo-student",
        "target_word": target_word,
        "target_text": target_text,
        "prompt_text": target_text,
        "audio_path": str(audio_path),
        "audio_url": str(audio_path),
    }


def _build_webhook_payload(job_id: str, ai_result: dict[str, Any]) -> dict[str, Any]:
    if ai_result.get("status") == "failed":
        return build_failed_webhook_payload(
            job_id,
            str((ai_result.get("metadata") or {}).get("error") or "AI scoring failed."),
            ai_result,
        )
    payload = build_success_webhook_payload(job_id, ai_result)
    is_valid, issues = validate_webhook_payload(payload)
    if is_valid:
        return payload
    return build_failed_webhook_payload(
        job_id,
        "AI worker produced an invalid webhook payload: " + " | ".join(issues),
        ai_result,
    )


def _post_if_requested(payload: dict[str, Any], post: bool) -> None:
    if not post:
        print("post_attempted=False")
        return

    webhook_url = os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")
    webhook_secret = os.getenv("AI_WEBHOOK_SECRET")
    if not webhook_url or not webhook_secret:
        print("post_attempted=False")
        print("post_skipped_reason=NODE_WEBHOOK_URL/AI_WEBHOOK_URL and AI_WEBHOOK_SECRET are required.")
        return

    try:
        import requests

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"x-ai-webhook-secret": webhook_secret},
            timeout=30,
        )
    except Exception as exc:
        print("post_attempted=True")
        print(f"post_error={exc}")
        return

    print("post_attempted=True")
    print(f"post_status_code={response.status_code}")
    if not 200 <= response.status_code < 300:
        print(response.text)


def _print_validation(label: str, is_valid: bool, issues: list[str]) -> None:
    print(f"{label}={is_valid}")
    if issues:
        print(f"{label}_issues=")
        for issue in issues:
            print(f"- {issue}")


def _run(args: argparse.Namespace, audio_path: Path) -> None:
    scorer_mode = _scorer_mode()
    confidence_threshold = _confidence_threshold()
    job = _build_job(args, audio_path)

    print("Warning: classifier confidence is not pronunciation score.")
    print("Warning: fallback alignment is approximate.")
    print("Warning: heuristic_gop is not real GOP.")
    print(f"scorer_mode={scorer_mode}")
    print(f"dry_run={not args.post}")

    ai_result = _normalized_result(
        _score_with_worker_path(job, scorer_mode, confidence_threshold),
        scorer_mode,
    )
    ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
    webhook_payload = _build_webhook_payload(job["job_id"], ai_result)
    webhook_valid, webhook_issues = validate_webhook_payload(webhook_payload)

    print("=== simulated_job ===")
    print(_json({**job, "audio_path": "<temp-or-local-audio>", "audio_url": "<temp-or-local-audio>"}))
    print("=== normalized_ai_result ===")
    print(_json(ai_result))
    _print_validation("ai_result_valid", ai_result_valid, ai_result_issues)
    print("=== webhook_payload ===")
    print(_json(webhook_payload))
    _print_validation("webhook_payload_valid", webhook_valid, webhook_issues)
    _post_if_requested(webhook_payload, args.post)


def main() -> int:
    args = _parse_args()
    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return 2
        _run(args, audio_path)
        return 0

    with tempfile.TemporaryDirectory(prefix="ai-worker-e2e-demo-") as temp_dir:
        audio_path = Path(temp_dir) / "demo.wav"
        _write_temp_wav(audio_path)
        _run(args, audio_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
