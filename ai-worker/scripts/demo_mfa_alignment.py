from __future__ import annotations

import json
import os
import sys
from pathlib import Path


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.alignment.alignment_service import align_audio  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage:")
        print('  python ai-worker/scripts/demo_mfa_alignment.py path/to/audio.wav "example prompt text"')
        return 2

    audio_path = Path(sys.argv[1])
    prompt_text = sys.argv[2]
    mode = os.getenv("ALIGNMENT_MODE", "fallback")
    print(f"ALIGNMENT_MODE={mode}")

    if mode.strip().lower() == "fallback":
        print("Using fallback alignment. This is approximate and not real forced alignment.")
    elif mode.strip().lower() == "mfa":
        print("Attempting local MFA alignment. MFA must already be installed and configured.")

    result = align_audio(audio_path, prompt_text)
    if result.get("status") == "failed":
        print("Alignment did not run successfully. See note/metadata.error for details.")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
