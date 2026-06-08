from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))


DEFAULT_TRANSCRIPT = "Architecture"
SENSITIVE_KEYS = {
    "audio_path",
    "checkpoint_path",
    "local_path",
    "mfa_output_dir",
    "mfa_temp_dir",
    "temp_dir",
    "textgrid_path",
}


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CNN Attention context inference with MFA alignment metadata.",
    )
    parser.add_argument("--audio-path", default=None, help="Optional local audio file. Real MFA is skipped when omitted.")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT, help=f'Transcript text. Default: "{DEFAULT_TRANSCRIPT}"')
    parser.add_argument("--checkpoint-path", default=None, help="Optional override for the context checkpoint path.")
    parser.add_argument("--alignment-mode", default="mfa", help='Alignment mode. Default: "mfa".')
    parser.add_argument(
        "--allow-fallback",
        type=_parse_bool,
        default=True,
        help="Allow fallback alignment when MFA fails. Default: true.",
    )
    parser.add_argument("--conda-env", default="mfa", help='MFA conda environment name. Default: "mfa".')
    parser.add_argument(
        "--dictionary-path",
        default="english_us_mfa",
        help='MFA dictionary model name/path. Default: "english_us_mfa".',
    )
    parser.add_argument(
        "--acoustic-model-path",
        default="english_mfa",
        help='MFA acoustic model name/path. Default: "english_mfa".',
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config only. Do not run scorer or MFA.")
    return parser.parse_args()


def _set_env(args: argparse.Namespace) -> dict[str, str]:
    config = {
        "SCORER_MODE": "cnn_attention_context",
        "ALIGNMENT_MODE": str(args.alignment_mode or "mfa"),
        "ALLOW_ALIGNMENT_FALLBACK": "true" if args.allow_fallback else "false",
        "MFA_CONDA_ENV": str(args.conda_env or "mfa"),
        "MFA_DICTIONARY_PATH": str(args.dictionary_path or "english_us_mfa"),
        "MFA_ACOUSTIC_MODEL_PATH": str(args.acoustic_model_path or "english_mfa"),
        "CNN_ATTENTION_CONTEXT_MODE": "context_0_10",
        "CNN_ATTENTION_CONTEXT_LEFT_SECONDS": "0.10",
        "CNN_ATTENTION_CONTEXT_RIGHT_SECONDS": "0.10",
    }
    if args.checkpoint_path:
        config["CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH"] = str(Path(args.checkpoint_path).expanduser())
    for key, value in config.items():
        os.environ[key] = value
    return config


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_value(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


def _find_sensitive_keys(value: Any, path: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in SENSITIVE_KEYS:
                issues.append(next_path)
            issues.extend(_find_sensitive_keys(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            next_path = f"{path}[{index}]"
            issues.extend(_find_sensitive_keys(item, next_path))
    return issues


def _print_section(title: str, payload: Any) -> None:
    print(title)
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def _alignment_summary(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return {
        "alignment_status": metadata.get("alignment_status"),
        "alignment_method": metadata.get("alignment_method"),
        "requested_alignment_mode": metadata.get("requested_alignment_mode"),
        "is_forced_alignment": metadata.get("is_forced_alignment"),
        "mfa_used": metadata.get("mfa_used"),
        "mfa_attempted": metadata.get("mfa_attempted"),
        "mfa_exit_code": metadata.get("mfa_exit_code"),
        "textgrid_parse_success": metadata.get("textgrid_parse_success"),
        "fallback_alignment": metadata.get("fallback_alignment"),
        "fallback_reason": metadata.get("fallback_reason"),
        "word_segments_count": metadata.get("word_segments_count"),
        "phone_segments_count": metadata.get("phone_segments_count"),
        "alignment_note": metadata.get("alignment_note"),
    }


def _scorer_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    diagnosis = result.get("diagnosis") if isinstance(result.get("diagnosis"), dict) else {}
    return {
        "status": result.get("status"),
        "score": result.get("score"),
        "score_note": result.get("score_note"),
        "pronunciation_score_source": result.get("pronunciation_score_source"),
        "predicted_error_type": result.get("predicted_error_type"),
        "diagnosis_confidence": diagnosis.get("diagnosis_confidence"),
        "segments_count": metadata.get("segments_count"),
        "context_mode": metadata.get("context_mode"),
        "context_used": metadata.get("context_used"),
        "location_reliability": metadata.get("location_reliability"),
    }


def _validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    try:
        from app.contracts.ai_result_validator import validate_ai_result
    except Exception as exc:
        return {
            "validator_available": False,
            "validator_error": str(exc),
        }

    is_valid, issues = validate_ai_result(result)
    return {
        "validator_available": True,
        "ai_result_valid": is_valid,
        "issues": issues,
    }


def main() -> int:
    args = _parse_args()
    config = _set_env(args)
    audio_path = Path(args.audio_path).expanduser() if args.audio_path else None

    config_summary = {
        **config,
        "audio_path_provided": bool(audio_path),
        "audio_path_exists": bool(audio_path and audio_path.exists()),
        "transcript": args.transcript,
        "dry_run": args.dry_run,
    }
    _print_section("CONFIG", config_summary)

    if args.dry_run or audio_path is None:
        reason = "Dry run requested." if args.dry_run else "No audio path provided."
        _print_section(
            "ALIGNMENT SUMMARY",
            {
                "executed": False,
                "reason": reason,
                "alignment_mode": config["ALIGNMENT_MODE"],
                "fallback_enabled": args.allow_fallback,
            },
        )
        _print_section(
            "SCORER RESULT SUMMARY",
            {
                "executed": False,
                "note": "Setup/config only. Real MFA and scorer inference were skipped.",
            },
        )
        _print_section(
            "VALIDATION",
            {
                "executed": False,
                "note": "No AI result was produced during dry-run/config-only mode.",
            },
        )
        _print_section(
            "METADATA SAFETY CHECK",
            {
                "executed": False,
                "sensitive_keys_blocked": sorted(SENSITIVE_KEYS),
            },
        )
        return 0

    if not audio_path.exists():
        _print_section(
            "ALIGNMENT SUMMARY",
            {
                "executed": False,
                "error": f"Audio file not found: {audio_path}",
            },
        )
        return 2

    try:
        from app.scorers.cnn_attention_scorer import score_pronunciation_context
    except Exception as exc:
        _print_section(
            "SCORER RESULT SUMMARY",
            {
                "executed": False,
                "error": str(exc),
            },
        )
        return 1

    result = score_pronunciation_context(
        {
            "job_id": "demo-context-mfa-aligned-inference",
            "audio_path": str(audio_path),
            "prompt_text": args.transcript,
            "target_text": args.transcript,
            "target_sentence": args.transcript,
            "target_word": args.transcript,
        }
    )
    safe_result = _safe_value(result)

    _print_section("ALIGNMENT SUMMARY", _alignment_summary(safe_result))
    _print_section("SCORER RESULT SUMMARY", _scorer_result_summary(safe_result))
    _print_section("VALIDATION", _validation_summary(safe_result))

    sensitive_issues = _find_sensitive_keys(result)
    _print_section(
        "METADATA SAFETY CHECK",
        {
            "passed": not sensitive_issues,
            "sensitive_paths_found": sensitive_issues,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
