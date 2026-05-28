from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))


def _default_checkpoint_path() -> Path:
    return REPO_ROOT / "ai-training" / "models" / "l2_arctic_error_type_cnn_attention.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN Attention aligned inference with approximate fallback alignment."
    )
    parser.add_argument("audio_path", help="Path to a local audio file.")
    parser.add_argument("prompt_text", help="Prompt text spoken in the audio.")
    parser.add_argument(
        "--phones",
        nargs="+",
        default=None,
        help="Optional canonical phones, for example: --phones EH G Z AE M P AH L",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audio_path = Path(args.audio_path)
    checkpoint_path = _default_checkpoint_path()

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 2

    if not checkpoint_path.exists():
        print("CNN Attention checkpoint not found.")
        print(f"Expected local checkpoint: {checkpoint_path}")
        print("Checkpoint files are local artifacts and are not committed.")
        return 1

    print("Warning: fallback alignment is approximate and not real forced alignment.")

    try:
        from app.alignment.fallback_aligner import align_prompt_fallback
        from app.scorers.cnn_attention_scorer import score_aligned_audio

        alignment_result = align_prompt_fallback(
            audio_path,
            prompt_text=args.prompt_text,
            canonical_phones=args.phones,
        )
        result = score_aligned_audio(audio_path, alignment_result)
    except Exception as exc:
        print(str(exc))
        print(f"Expected local checkpoint: {checkpoint_path}")
        return 1

    print("=== normalized_ai_result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=== segment_predictions ===")
    print(json.dumps(result.get("segments", []), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
