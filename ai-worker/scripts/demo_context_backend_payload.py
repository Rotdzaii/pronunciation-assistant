from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.webhook_payload import (  # noqa: E402
    build_success_webhook_payload,
    validate_webhook_payload,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "ai-training"
    / "models"
    / "l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate a context CNN Attention backend payload.")
    parser.add_argument("--job-id", default=None, help="Job id to put in the webhook payload.")
    parser.add_argument("--audio-path", default=None, help="Optional local audio path.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local context checkpoint path.")
    parser.add_argument("--prompt-text", default="example", help="Prompt text used for fallback alignment.")
    parser.add_argument(
        "--phones",
        nargs="+",
        default=["EH", "G", "Z", "AE", "M", "P", "AH", "L"],
        help="Canonical phones used for fallback alignment.",
    )
    parser.add_argument("--dry-run", action="store_true", default=True, help="Build payload without POSTing. Default.")
    parser.add_argument("--post", action="store_true", help="Explicitly POST to the backend webhook.")
    parser.add_argument("--webhook-url", default=None, help="Optional webhook URL for --post.")
    parser.add_argument("--secret", default=None, help="Optional webhook secret for --post.")
    return parser.parse_args()


def write_temp_wav() -> Path:
    sample_rate = 16000
    duration_seconds = 1.2
    sample_count = int(sample_rate * duration_seconds)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = Path(temp_file.name)
    temp_file.close()

    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_count):
            value = int(0.15 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))

    return temp_path


def configure_context_environment(checkpoint_path: Path) -> None:
    os.environ["SCORER_MODE"] = "cnn_attention_context"
    os.environ["CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"] = str(checkpoint_path)
    os.environ["CNN_ATTENTION_CONTEXT_MODE"] = "context_0_10"
    os.environ["CNN_ATTENTION_CONTEXT_LEFT_SECONDS"] = "0.10"
    os.environ["CNN_ATTENTION_CONTEXT_RIGHT_SECONDS"] = "0.10"


def top_segment(ai_result: dict[str, Any]) -> dict[str, Any]:
    segments = ai_result.get("segments") or []
    if not segments:
        return {}
    return max(segments, key=lambda item: float(item.get("diagnosis_confidence") or 0.0))


def context_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
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
    generated_audio = False
    audio_path = Path(args.audio_path) if args.audio_path else write_temp_wav()
    generated_audio = args.audio_path is None
    checkpoint_path = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else DEFAULT_CHECKPOINT
    job_id = args.job_id or str(uuid.uuid4())
    post_requested = bool(args.post)

    try:
        print("Warning: classifier confidence is not pronunciation score.")
        print("Warning: heuristic score is not real GOP.")
        print("Warning: fallback alignment is approximate.")
        print(f"post_attempted={post_requested}")

        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return 2
        if not checkpoint_path.exists():
            print("Context checkpoint not found.")
            print(f"Expected local checkpoint: {checkpoint_path}")
            print("Set --checkpoint-path or CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH to an existing local .pt file.")
            print("Checkpoint files are local artifacts and must not be committed.")
            return 1

        configure_context_environment(checkpoint_path)

        try:
            from app.scorers.cnn_attention_scorer import score_pronunciation_context

            ai_result = score_pronunciation_context(
                {
                    "job_id": job_id,
                    "student_id": "demo-student",
                    "target_word": args.prompt_text,
                    "prompt_text": args.prompt_text,
                    "target_text": args.prompt_text,
                    "canonical_phones": args.phones,
                    "audio_path": str(audio_path),
                }
            )
        except Exception as exc:
            print("Context scorer inference failed.")
            print(str(exc))
            print(f"Checkpoint path: {checkpoint_path}")
            print("If torch is missing, install the AI Worker inference dependencies before retrying.")
            return 1

        ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
        payload = build_success_webhook_payload(job_id, ai_result)
        payload_valid, payload_issues = validate_webhook_payload(payload)
        top = top_segment(ai_result)

        print("=== scorer_result_summary ===")
        print(
            json.dumps(
                {
                    "job_id": job_id,
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

        print("=== context_metadata_in_feedback_ai_result ===")
        print(json.dumps(context_metadata_from_payload(payload), indent=2, ensure_ascii=False))

        print("=== validation ===")
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

        print("=== webhook_payload_summary ===")
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
                    "metadata": {
                        "context_mode": payload.get("metadata", {}).get("context_mode"),
                        "context_used": payload.get("metadata", {}).get("context_used"),
                        "scoring_is_heuristic": payload.get("metadata", {}).get("scoring_is_heuristic"),
                        "alignment_method": payload.get("metadata", {}).get("alignment_method"),
                    },
                    "feedback_ai_result_present": isinstance(payload.get("feedback", {}).get("ai_result"), dict),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        if post_requested:
            webhook_url = args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")
            secret = args.secret or os.getenv("AI_WEBHOOK_SECRET")
            if not webhook_url or not secret:
                print("post_skipped_reason=--webhook-url/NODE_WEBHOOK_URL and --secret/AI_WEBHOOK_SECRET are required.")
                return 1
            posted, post_message = post_payload(webhook_url, secret, payload)
            print(f"post_success={posted}")
            print(f"post_result={post_message}")
            return 0 if posted else 1

        print("dry_run=True")
        return 0 if ai_result_valid and payload_valid else 1
    finally:
        if generated_audio:
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
