from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.alignment.textgrid_parser import parse_textgrid as parse_textgrid_file  # noqa: E402
from app.contracts.alignment_contract import AlignmentError  # noqa: E402


DEFAULT_TRANSCRIPT = "Architecture"
SUPPORTED_AUDIO_SUFFIXES = {".wav"}


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _resolve_command(command: str) -> str | None:
    resolved = shutil.which(command)
    if resolved:
        return resolved

    command_path = Path(command).expanduser()
    if command_path.exists():
        return str(command_path)
    return None


def _build_mfa_command_prefix(mfa_command: str, conda_env: str | None) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, mfa_command]
    return [mfa_command]


def _format_command(command_args: list[str]) -> str:
    return " ".join(f'"{arg}"' if " " in arg else arg for arg in command_args)


def _probe_mfa(command_prefix: list[str], conda_env: str | None) -> tuple[list[str] | None, str]:
    if conda_env:
        resolved_prefix = command_prefix
    else:
        resolved = _resolve_command(command_prefix[0])
        if not resolved:
            return None, f"MFA command not found: {command_prefix[0]}"
        resolved_prefix = [resolved]

    try:
        completed = subprocess.run(
            [*resolved_prefix, "version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return resolved_prefix, f"MFA command found but version probe failed: {exc}"

    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode == 0:
        return resolved_prefix, output or "MFA command is available."

    return resolved_prefix, output or "MFA command was found, but version output was not available."


def _path_status(path_value: str | None, label: str) -> tuple[Path | None, str | None]:
    if not path_value:
        return None, f"{label} is not configured."

    path = Path(path_value).expanduser()
    if not path.exists():
        return path, f"{label} does not exist: {path}"
    return path, None


def _model_status(model_value: str | None, label: str) -> tuple[str | None, str | None]:
    if not model_value:
        return None, f"{label} is not configured."

    path = Path(model_value).expanduser()
    if path.exists():
        return str(path), None
    return model_value, None


def _print_setup_instructions() -> None:
    print()
    print("Setup required before real MFA validation:")
    print("  1. Install Montreal Forced Aligner locally.")
    print("  2. Provide a pronunciation dictionary via --dictionary-path or MFA_DICTIONARY_PATH.")
    print("  3. Provide an acoustic model via --acoustic-model-path or MFA_ACOUSTIC_MODEL_PATH.")
    print("  4. Provide a local WAV audio file with --audio-path.")
    print("  5. Provide the expected transcript with --transcript.")
    print()
    print("This script does not install MFA, download models, or train anything.")
    print("Alignment boundaries are timing evidence only, not pronunciation correctness.")


def _write_lab_file(path: Path, transcript: str) -> None:
    text = _normalized_text(transcript)
    if not text:
        raise ValueError("Transcript is empty after normalization.")
    path.write_text(text + "\n", encoding="utf-8")


def _create_work_dirs(output_dir: str | None, keep_temp: bool) -> tuple[Path, bool]:
    if output_dir:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="mfa-local-validation-", dir=str(root)))
        return work_dir, True

    if keep_temp:
        return Path(tempfile.mkdtemp(prefix="mfa-local-validation-")), True

    return Path(tempfile.mkdtemp(prefix="mfa-local-validation-")), False


def _summarize_segments(result: dict[str, Any]) -> None:
    words = result.get("words") or []
    phones = result.get("phones") or []
    metadata = result.get("metadata") or {}

    print()
    print("Parsed TextGrid summary:")
    print("  textgrid_parse_success=True")
    print(f"  alignment_status: {result.get('alignment_status')}")
    print(f"  alignment_method: {result.get('alignment_method')}")
    print(f"  is_forced_alignment: {str(bool(metadata.get('is_forced_alignment'))).lower()}")
    print(f"  word segments: {len(words)}")
    print(f"  phone segments: {len(phones)}")

    if words:
        print("  first words:")
        for segment in words[:5]:
            print(f"    {segment.get('word')} [{segment.get('start')}s, {segment.get('end')}s]")

    if phones:
        print("  first phones:")
        for segment in phones[:10]:
            print(f"    {segment.get('phone')} [{segment.get('start')}s, {segment.get('end')}s]")


def _run_mfa(
    *,
    command_prefix: list[str],
    audio_path: Path,
    transcript: str,
    dictionary_model: str,
    acoustic_model: str,
    output_dir: str | None,
    keep_temp: bool,
    should_parse_textgrid: bool,
) -> int:
    work_dir, explicit_work_dir = _create_work_dirs(output_dir, keep_temp)
    corpus_dir = work_dir / "corpus"
    aligned_dir = work_dir / "aligned"

    try:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        working_audio = corpus_dir / audio_path.name
        shutil.copy2(audio_path, working_audio)
        _write_lab_file(working_audio.with_suffix(".lab"), transcript)

        command_args = [
            *command_prefix,
            "align",
            str(corpus_dir),
            dictionary_model,
            acoustic_model,
            str(aligned_dir),
            "--clean",
            "--single_speaker",
        ]

        print()
        print("Running MFA align:")
        print("  " + _format_command(command_args))

        completed = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        print(f"MFA exit code: {completed.returncode}")
        if completed.stdout.strip():
            print("MFA stdout:")
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print("MFA stderr:")
            print(completed.stderr.strip())

        if completed.returncode != 0:
            print()
            print("MFA validation failed. Check MFA setup, dictionary coverage, model compatibility, and audio format.")
            return completed.returncode or 1

        textgrid_candidates = sorted(aligned_dir.rglob(f"{audio_path.stem}.TextGrid"))
        if not textgrid_candidates:
            textgrid_candidates = sorted(aligned_dir.rglob("*.TextGrid"))

        if not textgrid_candidates:
            print()
            print(f"MFA completed but no TextGrid was found under: {aligned_dir}")
            return 1

        textgrid_path = textgrid_candidates[0]
        print()
        print(f"TextGrid generated locally: {textgrid_path}")
        print("Do not commit this TextGrid output.")

        if should_parse_textgrid:
            try:
                result = parse_textgrid_file(textgrid_path)
            except AlignmentError as exc:
                print()
                print("TextGrid was generated but parsing failed.")
                print("MFA alignment completed successfully; this is a parser/script integration validation failure.")
                print(f"TextGrid parser error: {exc}")
                return 1
            _summarize_segments(result)

        return 0
    finally:
        if keep_temp:
            print()
            print(f"Keeping temporary validation directory: {work_dir}")
            print("Do not commit this directory or generated alignment files.")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            if explicit_work_dir:
                print()
                print(f"Cleaned temporary validation directory: {work_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local MFA readiness and parse a generated TextGrid when available.",
    )
    parser.add_argument("--audio-path", help="Local WAV audio file to align. Required unless --dry-run is used.")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT, help=f'Transcript text. Default: "{DEFAULT_TRANSCRIPT}"')
    parser.add_argument("--output-dir", help="Directory where a temporary MFA validation folder will be created.")
    parser.add_argument("--mfa-command", default="mfa", help='MFA command. Default: "mfa".')
    parser.add_argument("--conda-env", default=None, help="Optional conda environment name used to run MFA via conda run.")
    parser.add_argument(
        "--dictionary-path",
        default=os.getenv("MFA_DICTIONARY_PATH"),
        help="MFA dictionary model name/path or env MFA_DICTIONARY_PATH.",
    )
    parser.add_argument(
        "--acoustic-model-path",
        default=os.getenv("MFA_ACOUSTIC_MODEL_PATH"),
        help="MFA acoustic model name/path or env MFA_ACOUSTIC_MODEL_PATH.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary corpus/output files for local inspection.")
    parser.add_argument("--parse-textgrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Check configuration without running MFA alignment.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transcript = _normalized_text(args.transcript)

    print("MFA local alignment validation")
    print(f"Transcript: {transcript!r}")
    print(f"MFA command requested: {args.mfa_command}")
    command_prefix = _build_mfa_command_prefix(args.mfa_command, args.conda_env)
    print(f"Effective MFA command: {_format_command(command_prefix)}")

    command, availability = _probe_mfa(command_prefix, args.conda_env)
    print(f"MFA availability: {availability}")

    dictionary_model, dictionary_error = _model_status(args.dictionary_path, "Dictionary model/path")
    acoustic_model, acoustic_error = _model_status(args.acoustic_model_path, "Acoustic model/path")
    audio_path, audio_error = _path_status(args.audio_path, "Audio path")

    for error in (dictionary_error, acoustic_error):
        if error:
            print(f"Config issue: {error}")

    if args.audio_path:
        if audio_error:
            print(f"Config issue: {audio_error}")
        elif audio_path and audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            print(f"Config warning: audio suffix {audio_path.suffix!r} may be unsupported. Prefer a 16 kHz mono WAV file.")
    else:
        print("Config issue: Audio path is required for real alignment. Dry-run can run without audio.")

    if args.dry_run:
        print()
        print("Dry run only: MFA alignment was not executed.")
        if not command or dictionary_error or acoustic_error or not args.audio_path or audio_error:
            _print_setup_instructions()
        return 0

    if not command or dictionary_error or acoustic_error or not args.audio_path or audio_error:
        _print_setup_instructions()
        return 2

    if not transcript:
        print("Config issue: transcript is empty after normalization.")
        return 2

    assert audio_path is not None
    assert dictionary_model is not None
    assert acoustic_model is not None
    return _run_mfa(
        command_prefix=command,
        audio_path=audio_path,
        transcript=transcript,
        dictionary_model=dictionary_model,
        acoustic_model=acoustic_model,
        output_dir=args.output_dir,
        keep_temp=args.keep_temp,
        should_parse_textgrid=args.parse_textgrid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
