from __future__ import annotations

import json
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))


def _default_checkpoint_path() -> Path:
    return REPO_ROOT / "ai-training" / "models" / "l2_arctic_error_type_cnn_attention.pt"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python ai-worker/scripts/demo_cnn_attention_scorer.py path/to/audio.wav")
        print()
        print(f"Expected local checkpoint: {_default_checkpoint_path()}")
        print("Checkpoint files are local artifacts and are not committed.")
        return 2

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        return 2

    try:
        from app.scorers.cnn_attention_scorer import default_checkpoint_path, score_pronunciation

        result = score_pronunciation(
            {
                "job_id": "demo-cnn-attention",
                "target_word": audio_path.stem,
                "audio_path": str(audio_path),
            }
        )
    except Exception as exc:
        print(str(exc))
        try:
            checkpoint_path = default_checkpoint_path()
        except UnboundLocalError:
            checkpoint_path = _default_checkpoint_path()
        print(f"Expected local checkpoint: {checkpoint_path}")
        print("Checkpoint files are local artifacts and are not committed.")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
