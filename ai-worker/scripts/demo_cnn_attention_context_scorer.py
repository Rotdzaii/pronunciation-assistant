from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import wave
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))


def _default_context_checkpoint_path() -> Path:
    return REPO_ROOT / "ai-training" / "models" / "l2_arctic_cnn_attention_context_0_10.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN Attention context_0_10 scorer demo.")
    parser.add_argument("audio_path", nargs="?", default=None, help="Optional local audio file.")
    parser.add_argument("--prompt-text", default="example", help="Prompt text for fallback alignment.")
    parser.add_argument(
        "--phones",
        nargs="+",
        default=["EH", "G", "Z", "AE", "M", "P", "AH", "L"],
        help="Canonical phones for fallback alignment.",
    )
    return parser.parse_args()


def _write_temp_wav() -> Path:
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


def main() -> int:
    args = _parse_args()
    generated_audio = False
    audio_path = Path(args.audio_path) if args.audio_path else _write_temp_wav()
    generated_audio = args.audio_path is None

    try:
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return 2

        try:
            from app.scorers.cnn_attention_scorer import default_context_checkpoint_path, score_pronunciation_context

            checkpoint_path = default_context_checkpoint_path()
            result = score_pronunciation_context(
                {
                    "job_id": "demo-cnn-attention-context",
                    "target_word": args.prompt_text,
                    "prompt_text": args.prompt_text,
                    "canonical_phones": args.phones,
                    "audio_path": str(audio_path),
                }
            )
        except Exception as exc:
            print("CNN Attention context demo could not run inference.")
            print(str(exc))
            try:
                checkpoint_path = default_context_checkpoint_path()
            except UnboundLocalError:
                checkpoint_path = _default_context_checkpoint_path()
            print(f"Expected local context checkpoint: {checkpoint_path}")
            print("Set CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH to override.")
            print("Checkpoint files are local artifacts and are not committed.")
            print("Classifier confidence is not pronunciation correctness.")
            print("Heuristic score is not real GOP.")
            return 1

        print("=== normalized_result_metadata ===")
        print(json.dumps(result.get("metadata", {}), indent=2, ensure_ascii=False))
        print("=== confidence_note ===")
        print(result.get("diagnosis", {}).get("confidence_note", "Classifier confidence, not pronunciation correctness."))
        print("=== score_note ===")
        print(result.get("score_note", "Heuristic score is not real GOP."))
        print("=== segments ===")
        print(json.dumps(result.get("segments", []), indent=2, ensure_ascii=False))
        print(f"Context checkpoint: {checkpoint_path}")
        return 0
    finally:
        if generated_audio:
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
