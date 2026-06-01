from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.alignment.alignment_service import align_audio  # noqa: E402
from app.contracts.ai_result_validator import validate_ai_result  # noqa: E402
from app.contracts.alignment_contract import get_alignment_segments  # noqa: E402
from app.contracts.webhook_payload import build_success_webhook_payload, validate_webhook_payload  # noqa: E402
from app.scorers import cnn_attention_scorer as context_scorer  # noqa: E402


SCORER_MODE = "cnn_attention_context"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "ai-training"
    / "models"
    / "l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt"
)
DEFAULT_OUTPUT_JSON = AI_WORKER_ROOT / "docs" / "context_runtime_benchmark_latest.json"
DEFAULT_PHONES = ["EH", "G", "Z", "AE", "M", "P", "AH", "L"]
TIMING_FIELDS = [
    "total_runtime_seconds",
    "setup_time_seconds",
    "model_load_time_seconds",
    "audio_prepare_time_seconds",
    "alignment_time_seconds",
    "inference_time_seconds",
    "ai_result_validation_time_seconds",
    "webhook_payload_build_time_seconds",
    "webhook_payload_validation_time_seconds",
    "post_time_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AI Worker context CNN Attention runtime.")
    parser.add_argument("--audio-path", default=None, help="Optional local audio file. Generates temp WAV if omitted.")
    parser.add_argument("--job-id", default="benchmark-job-id", help="Job id used in the webhook payload.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local context checkpoint path.")
    parser.add_argument("--runs", type=int, default=3, help="Measured run count. Default: 3.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup run count before measurement. Default: 1.")
    parser.add_argument("--post", action="store_true", help="Explicitly POST the payload. Default never posts.")
    parser.add_argument("--webhook-url", default=None, help="Webhook URL used only with --post.")
    parser.add_argument("--secret", default=None, help="Webhook secret used only with --post. Never printed.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Small JSON report path.")
    return parser.parse_args()


def monotonic() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(monotonic() - start, 6)


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


def resolve_checkpoint_path(configured: str | None) -> Path:
    if configured:
        return Path(configured).expanduser()
    env_path = os.getenv("CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH")
    return Path(env_path).expanduser() if env_path else DEFAULT_CHECKPOINT


def configure_environment(checkpoint_path: Path) -> None:
    os.environ["SCORER_MODE"] = SCORER_MODE
    os.environ["CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"] = str(checkpoint_path)
    os.environ["CNN_ATTENTION_CONTEXT_MODE"] = "context_0_10"
    os.environ["CNN_ATTENTION_CONTEXT_LEFT_SECONDS"] = "0.10"
    os.environ["CNN_ATTENTION_CONTEXT_RIGHT_SECONDS"] = "0.10"
    os.environ.setdefault("ALIGNMENT_MODE", "fallback")
    os.environ.setdefault("ALLOW_ALIGNMENT_FALLBACK", "true")
    os.environ.setdefault("SCORING_MODE", "heuristic_gop")


def dependency_info() -> dict[str, Any]:
    torch = context_scorer.torch
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_available": cuda_available,
        "torch_device_name": device_name,
        "mode": "gpu" if cuda_available else "cpu",
    }


def build_job(job_id: str, audio_path: Path) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "student_id": "benchmark-student",
        "target_word": "example",
        "target_text": "example",
        "prompt_text": "example",
        "canonical_phones": DEFAULT_PHONES,
        "audio_path": str(audio_path),
    }


def timed_context_segment_prediction(
    *,
    model: Any,
    index_to_label: dict[int, str],
    device: Any,
    checkpoint_file: Path,
    audio_path: Path,
    segment: dict[str, Any],
    fallback_index: int,
    context_config: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    start_time = float(segment.get("start") or 0.0)
    end_time = float(segment.get("end") if segment.get("end") is not None else start_time)
    if end_time <= start_time:
        end_time = start_time + context_scorer.MAX_SECONDS

    audio_prepare_start = monotonic()
    context_metadata = context_scorer._context_crop_metadata(audio_path, start_time, end_time, context_config)  # noqa: SLF001
    feature, audio_metadata = context_scorer._feature_from_audio_path(  # noqa: SLF001
        audio_path,
        context_metadata["crop_start_time"],
        context_metadata["crop_end_time"],
    )
    audio_prepare_time = monotonic() - audio_prepare_start

    inference_start = monotonic()
    with context_scorer.torch.no_grad():
        logits = model(feature.to(device))
        probabilities = context_scorer.torch.softmax(logits, dim=1).squeeze(0).cpu()
    inference_time = monotonic() - inference_start

    predicted_index = int(context_scorer.torch.argmax(probabilities).item())
    predicted_error_type = index_to_label[predicted_index]
    class_probabilities = {
        index_to_label[index]: float(probabilities[index].item())
        for index in sorted(index_to_label)
    }
    audio_metadata["context"] = context_metadata
    prediction = {
        "predicted_error_type": predicted_error_type,
        "class_probabilities": class_probabilities,
        "diagnosis_confidence": class_probabilities[predicted_error_type],
        "device": str(device),
        "checkpoint_path": str(checkpoint_file),
        "audio": audio_metadata,
    }
    return (
        context_scorer._format_segment_prediction(  # noqa: SLF001
            prediction,
            start_time,
            end_time,
            phone=segment.get("phone"),
            word=segment.get("word"),
            segment_type=segment.get("type"),
            index=segment.get("index", fallback_index),
        ),
        audio_prepare_time,
        inference_time,
    )


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


def benchmark_once(
    *,
    run_index: int,
    run_kind: str,
    job_id: str,
    audio_path: Path,
    checkpoint_path: Path,
    post: bool,
    webhook_url: str | None,
    secret: str | None,
) -> dict[str, Any]:
    run_start = monotonic()
    setup_start = monotonic()
    configure_environment(checkpoint_path)
    job = build_job(job_id, audio_path)
    context_config = context_scorer._context_config()  # noqa: SLF001
    setup_time = elapsed(setup_start)

    model_load_start = monotonic()
    model, index_to_label, checkpoint_file, device = context_scorer._load_context_model(checkpoint_path)  # noqa: SLF001
    model_load_time = elapsed(model_load_start)

    alignment_start = monotonic()
    alignment_result = align_audio(
        audio_path,
        prompt_text=job["prompt_text"],
        canonical_phones=job["canonical_phones"],
    )
    alignment_time = elapsed(alignment_start)

    audio_prepare_time = 0.0
    inference_time = 0.0
    segment_predictions = []
    for fallback_index, segment in enumerate(get_alignment_segments(alignment_result)):
        segment_prediction, segment_audio_prepare, segment_inference = timed_context_segment_prediction(
            model=model,
            index_to_label=index_to_label,
            device=device,
            checkpoint_file=checkpoint_file,
            audio_path=audio_path,
            segment=segment,
            fallback_index=fallback_index,
            context_config=context_config,
        )
        segment_predictions.append(segment_prediction)
        audio_prepare_time += segment_audio_prepare
        inference_time += segment_inference

    ai_result = context_scorer._aggregate_segment_predictions(  # noqa: SLF001
        segment_predictions,
        alignment_result=alignment_result,
        confidence_threshold=0.65,
        scorer_metadata=context_scorer.CONTEXT_SCORER_METADATA,
        extra_metadata={
            **context_config,
            "context_inference_note": (
                "CNN Attention ran on context-expanded crops while user-facing locations retain original "
                "alignment segment boundaries."
            ),
        },
    )

    ai_validation_start = monotonic()
    ai_result_valid, ai_result_issues = validate_ai_result(ai_result)
    ai_result_validation_time = elapsed(ai_validation_start)

    payload_build_start = monotonic()
    payload = build_success_webhook_payload(job_id, ai_result)
    payload_build_time = elapsed(payload_build_start)

    payload_validation_start = monotonic()
    payload_valid, payload_issues = validate_webhook_payload(payload)
    payload_validation_time = elapsed(payload_validation_start)

    post_time = None
    post_success = None
    post_message = None
    if post:
        post_start = monotonic()
        if not webhook_url or not secret:
            post_success = False
            post_message = "--webhook-url/NODE_WEBHOOK_URL and --secret/AI_WEBHOOK_SECRET are required for --post."
        elif not ai_result_valid or not payload_valid:
            post_success = False
            post_message = "AI result or webhook payload validation failed."
        else:
            post_success, post_message = post_payload(webhook_url, secret, payload)
        post_time = elapsed(post_start)

    timings = {
        "total_runtime_seconds": elapsed(run_start),
        "setup_time_seconds": setup_time,
        "model_load_time_seconds": model_load_time,
        "audio_prepare_time_seconds": round(audio_prepare_time, 6),
        "alignment_time_seconds": alignment_time,
        "inference_time_seconds": round(inference_time, 6),
        "ai_result_validation_time_seconds": ai_result_validation_time,
        "webhook_payload_build_time_seconds": payload_build_time,
        "webhook_payload_validation_time_seconds": payload_validation_time,
    }
    if post:
        timings["post_time_seconds"] = post_time
    return {
        "run_index": run_index,
        "run_kind": run_kind,
        "timings": timings,
        "segments_count": len(segment_predictions),
        "ai_result_valid": ai_result_valid,
        "ai_result_issues": ai_result_issues,
        "payload_valid": payload_valid,
        "payload_issues": payload_issues,
        "post_attempted": post,
        "post_success": post_success,
        "post_result": post_message,
        "predicted_error_type": ai_result.get("predicted_error_type"),
        "diagnosis_confidence": (ai_result.get("diagnosis") or {}).get("diagnosis_confidence"),
        "score": ai_result.get("score"),
        "score_note": ai_result.get("score_note"),
        "metadata": {
            "context_mode": (ai_result.get("metadata") or {}).get("context_mode"),
            "alignment_method": (ai_result.get("metadata") or {}).get("alignment_method"),
            "scoring_method": (ai_result.get("metadata") or {}).get("scoring_method"),
            "scoring_is_heuristic": (ai_result.get("metadata") or {}).get("scoring_is_heuristic"),
            "model_output_is_scoring": (ai_result.get("metadata") or {}).get("model_output_is_scoring"),
        },
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for field_name in TIMING_FIELDS:
        values = [
            float(run["timings"][field_name])
            for run in runs
            if run["timings"].get(field_name) is not None
        ]
        if not values:
            continue
        aggregate[field_name] = {
            "mean": round(statistics.mean(values), 6),
            "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return aggregate


def bottleneck_recommendation(aggregate: dict[str, Any]) -> str:
    candidates = {
        "model_load_time_seconds": "model_load_time dominates: prioritize model caching in the worker.",
        "inference_time_seconds": "inference_time dominates: evaluate GPU torch and batching/crop efficiency.",
        "audio_prepare_time_seconds": "audio_prepare_time dominates: optimize audio decoding and log-mel preprocessing.",
        "post_time_seconds": "post_time dominates: inspect backend/network latency.",
    }
    means = {
        field_name: aggregate.get(field_name, {}).get("mean", 0.0)
        for field_name in candidates
    }
    dominant = max(means, key=means.get)
    if means[dominant] <= 0:
        return "No timing bottleneck identified."
    return candidates[dominant]


def print_run(run: dict[str, Any]) -> None:
    print(f"=== {run['run_kind']}_run_{run['run_index']} ===")
    print(json.dumps(run["timings"], indent=2))
    print(f"ai_result_valid={run['ai_result_valid']} payload_valid={run['payload_valid']}")
    print("confidence_note=Classifier confidence is not pronunciation correctness.")
    print("score_note=Heuristic score is not real GOP.")


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("--runs must be at least 1")
        return 2
    if args.warmup_runs < 0:
        print("--warmup-runs must be greater than or equal to 0")
        return 2

    generated_audio = False
    audio_path = Path(args.audio_path) if args.audio_path else write_temp_wav()
    generated_audio = args.audio_path is None
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    webhook_url = args.webhook_url or os.getenv("NODE_WEBHOOK_URL") or os.getenv("AI_WEBHOOK_URL")
    secret = args.secret or os.getenv("AI_WEBHOOK_SECRET")
    output_json = Path(args.output_json)

    try:
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return 2
        if not checkpoint_path.exists():
            print("Context checkpoint not found.")
            print(f"Expected local checkpoint: {checkpoint_path}")
            print("Set --checkpoint-path or CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH to an existing local .pt file.")
            return 1

        configure_environment(checkpoint_path)
        env_info = dependency_info()
        print("=== dependency_info ===")
        print(json.dumps(env_info, indent=2))
        print(f"scorer_mode={SCORER_MODE}")
        print("context_mode=context_0_10")
        print(f"runs={args.runs} warmup_runs={args.warmup_runs}")
        print(f"post_attempted={bool(args.post)}")

        warmup_runs = [
            benchmark_once(
                run_index=index + 1,
                run_kind="warmup",
                job_id=args.job_id,
                audio_path=audio_path,
                checkpoint_path=checkpoint_path,
                post=False,
                webhook_url=None,
                secret=None,
            )
            for index in range(args.warmup_runs)
        ]
        for run in warmup_runs:
            print_run(run)

        measured_runs = []
        for index in range(args.runs):
            run = benchmark_once(
                run_index=index + 1,
                run_kind="measured",
                job_id=args.job_id,
                audio_path=audio_path,
                checkpoint_path=checkpoint_path,
                post=bool(args.post),
                webhook_url=webhook_url,
                secret=secret,
            )
            measured_runs.append(run)
            print_run(run)

        aggregate = aggregate_runs(measured_runs)
        recommendation = bottleneck_recommendation(aggregate)
        print("=== aggregate_timings ===")
        print(json.dumps(aggregate, indent=2))
        print(f"bottleneck_recommendation={recommendation}")

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": env_info,
            "config": {
                "scorer_mode": SCORER_MODE,
                "context_mode": "context_0_10",
                "context_left_seconds": 0.10,
                "context_right_seconds": 0.10,
                "runs": args.runs,
                "warmup_runs": args.warmup_runs,
                "post_attempted": bool(args.post),
                "generated_audio": generated_audio,
                "audio_path_source": "generated_temp_wav" if generated_audio else "provided_audio_path",
                "checkpoint_path": str(checkpoint_path),
                "output_json": str(output_json),
            },
            "warmup_runs": warmup_runs,
            "per_run_timings": measured_runs,
            "aggregate_timings": aggregate,
            "bottleneck_recommendation": recommendation,
            "notes": [
                f"CPU/GPU mode: {env_info['mode']}.",
                "Classifier confidence is not pronunciation correctness.",
                "Heuristic score is not real GOP.",
                "Fallback alignment is approximate.",
                "Default benchmark does not POST; post_time_seconds is present only when --post is passed.",
            ],
        }
        write_json_report(output_json, report)
        print(f"wrote_json={output_json}")
        return 0 if all(run["ai_result_valid"] and run["payload_valid"] for run in measured_runs) else 1
    finally:
        if generated_audio:
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
