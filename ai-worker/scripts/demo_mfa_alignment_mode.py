from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.alignment.alignment_service import align_audio  # noqa: E402


DEFAULT_TRANSCRIPT = "Architecture"


def _env_bool(value: bool) -> str:
    return "true" if value else "false"


def _command_preview(mfa_command: str, conda_env: str | None) -> str:
    if conda_env:
        return f"conda run -n {conda_env} {mfa_command}"
    return mfa_command


def _print_json(title: str, payload: dict[str, Any]) -> None:
    print()
    print(title)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _segment_preview(segments: list[dict[str, Any]], label_key: str, limit: int) -> list[dict[str, Any]]:
    preview = []
    for segment in segments[:limit]:
        preview.append(
            {
                label_key: segment.get(label_key),
                "start": segment.get("start"),
                "end": segment.get("end"),
            }
        )
    return preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo AI Worker MFA alignment mode without training models.")
    parser.add_argument("--audio-path", help="Local audio path. Required for real alignment.")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT, help=f'Transcript text. Default: "{DEFAULT_TRANSCRIPT}"')
    parser.add_argument("--alignment-mode", default="mfa", choices=["fallback", "mfa", "none"])
    parser.add_argument("--allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mfa-command", default="mfa")
    parser.add_argument("--conda-env", default="mfa")
    parser.add_argument("--dictionary-path", default="english_us_mfa")
    parser.add_argument("--acoustic-model-path", default="english_mfa")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    conda_env = args.conda_env.strip() or None

    os.environ["ALIGNMENT_MODE"] = args.alignment_mode
    os.environ["ALLOW_ALIGNMENT_FALLBACK"] = _env_bool(args.allow_fallback)
    os.environ["MFA_COMMAND"] = args.mfa_command
    os.environ["MFA_DICTIONARY_PATH"] = args.dictionary_path
    os.environ["MFA_ACOUSTIC_MODEL_PATH"] = args.acoustic_model_path
    if conda_env:
        os.environ["MFA_CONDA_ENV"] = conda_env
    else:
        os.environ.pop("MFA_CONDA_ENV", None)

    _print_json(
        "CONFIG",
        {
            "alignment_mode": args.alignment_mode,
            "allow_alignment_fallback": args.allow_fallback,
            "effective_mfa_command": _command_preview(args.mfa_command, conda_env),
            "dictionary_model_or_path": args.dictionary_path,
            "acoustic_model_or_path": args.acoustic_model_path,
            "transcript": args.transcript,
            "audio_path_configured": bool(args.audio_path),
            "dry_run": args.dry_run,
        },
    )

    if args.dry_run or not args.audio_path:
        print()
        print("Dry setup check only: real MFA alignment was not executed.")
        print("Run with --audio-path pointing to a local WAV file to execute alignment service MFA mode.")
        return 0

    result = align_audio(args.audio_path, prompt_text=args.transcript)
    words = result.get("words") or []
    phones = result.get("phones") or []

    _print_json(
        "ALIGNMENT RESULT",
        {
            "alignment_status": result.get("alignment_status"),
            "alignment_method": result.get("alignment_method"),
            "word_segment_count": len(words),
            "phone_segment_count": len(phones),
            "first_word_segments": _segment_preview(words, "word", 5),
            "first_phone_segments": _segment_preview(phones, "phone", 10),
            "metadata": result.get("metadata") or {},
        },
    )
    return 0 if result.get("alignment_status") not in {"failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
