from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.alignment.mfa_aligner import run_mfa_alignment
from app.contracts.alignment_contract import AlignmentError


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and validate one MFA alignment locally.")
    parser.add_argument("--audio", required=True, help="Local input audio. It is never overwritten.")
    parser.add_argument("--text", required=True, help="Expected transcript for MFA.")
    parser.add_argument("--dictionary", default=os.getenv("MFA_DICTIONARY_PATH"), help="MFA dictionary path or model name.")
    parser.add_argument("--acoustic-model", default=os.getenv("MFA_ACOUSTIC_MODEL_PATH"), help="MFA acoustic model path or name.")
    parser.add_argument("--output-dir", help="Directory to retain generated TextGrid artifacts.")
    parser.add_argument("--json-output", help="Write the complete report as JSON.")
    parser.add_argument("--segments-json", help="Write normalized word and phone segments as JSON.")
    parser.add_argument("--keep-debug", action="store_true", help="Keep temporary artifacts when MFA fails.")
    return parser


def _print_report(result: dict[str, Any]) -> None:
    quality = result.get("quality") or {}
    metrics = quality.get("metrics") or {}
    print(f"Alignment status: {result.get('alignment_status')}")
    print(f"Source: {result.get('alignment_source')}")
    print(f"Words: {metrics.get('number_of_words', len(result.get('words') or []))}")
    print(f"Phones: {metrics.get('number_of_phones', len(result.get('phones') or []))}")
    print(f"Coverage: {metrics.get('speech_coverage_ratio', 0.0):.2f}")
    print(f"OOV: {metrics.get('oov_count', 0)}")
    issues = quality.get("issues") or []
    print("Issues: " + (", ".join(issues) if issues else "none"))


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_mfa_alignment(
            args.audio,
            args.text,
            output_dir=args.output_dir,
            dictionary_path=args.dictionary,
            acoustic_model_path=args.acoustic_model,
            keep_debug_artifacts=args.keep_debug,
        )
    except AlignmentError as exc:
        report = {"alignment_status": "failed", "alignment_source": "none", "segments": [], "quality": {}, "error": exc.as_dict()}
        _print_report(report)
        print(f"Error: {exc.code}: {exc}")
        if args.json_output:
            _write_json(args.json_output, report)
        return 2

    _print_report(result)
    if args.json_output:
        _write_json(args.json_output, result)
    if args.segments_json:
        _write_json(args.segments_json, {"words": result.get("words", []), "phones": result.get("phones", [])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
