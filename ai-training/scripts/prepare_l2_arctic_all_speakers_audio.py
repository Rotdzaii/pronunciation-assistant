from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")
IGNORED_ARCHIVES = {"suitcase_corpus.zip"}


@dataclass
class SpeakerAudioState:
    speaker_id: str
    extracted: bool
    archive_path: Path | None
    expected_folder: Path
    expected_wav_folder: Path
    metadata_rows: int
    existing_audio_rows: int
    missing_audio_rows: int

    @property
    def archived(self) -> bool:
        return self.archive_path is not None

    @property
    def needs_extraction(self) -> bool:
        return self.archived and not self.extracted


def load_metadata() -> pd.DataFrame:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    return pd.read_csv(METADATA_CSV)


def discover_archives() -> dict[str, Path]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    archives = {}

    for archive_path in sorted(DATASET_ROOT.glob("*.zip")):
        if archive_path.name in IGNORED_ARCHIVES:
            continue

        archives[archive_path.stem] = archive_path

    return archives


def discover_extracted_speakers() -> set[str]:
    return {
        path.name
        for path in DATASET_ROOT.iterdir()
        if path.is_dir() and (path / "wav").exists()
    }


def build_speaker_states(df: pd.DataFrame) -> list[SpeakerAudioState]:
    archives = discover_archives()
    extracted_speakers = discover_extracted_speakers()
    speaker_ids = sorted(set(df["speaker_id"]) | set(archives) | extracted_speakers)
    states = []

    audio_exists = df["audio_path"].apply(lambda path: Path(path).exists())
    df = df.assign(audio_exists=audio_exists)

    for speaker_id in speaker_ids:
        subset = df[df["speaker_id"] == speaker_id]
        expected_folder = DATASET_ROOT / speaker_id
        expected_wav_folder = expected_folder / "wav"

        states.append(
            SpeakerAudioState(
                speaker_id=speaker_id,
                extracted=speaker_id in extracted_speakers,
                archive_path=archives.get(speaker_id),
                expected_folder=expected_folder,
                expected_wav_folder=expected_wav_folder,
                metadata_rows=int(len(subset)),
                existing_audio_rows=int(subset["audio_exists"].sum()) if len(subset) else 0,
                missing_audio_rows=int((~subset["audio_exists"]).sum()) if len(subset) else 0,
            )
        )

    return states


def assert_safe_zip_members(zip_file: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in zip_file.infolist():
        target_path = (destination / member.filename).resolve()

        if destination != target_path and destination not in target_path.parents:
            raise ValueError(f"Unsafe archive member path: {member.filename}")


def extract_speaker_archive(state: SpeakerAudioState, allow_existing: bool) -> bool:
    if state.archive_path is None:
        print(f"SKIP {state.speaker_id}: archive not found")
        return False

    if state.expected_folder.exists() and not allow_existing:
        print(
            f"SKIP {state.speaker_id}: extracted folder already exists. "
            "Use --allow-existing to extract archive members into an existing folder."
        )
        return False

    with zipfile.ZipFile(state.archive_path) as zip_file:
        assert_safe_zip_members(zip_file, DATASET_ROOT)
        zip_file.extractall(DATASET_ROOT)

    print(f"EXTRACTED {state.speaker_id}: {state.archive_path} -> {DATASET_ROOT}")
    return True


def print_summary(states: list[SpeakerAudioState]) -> None:
    extracted = [state.speaker_id for state in states if state.extracted]
    archived = [state.speaker_id for state in states if state.archived]
    missing_folders = [state.speaker_id for state in states if state.metadata_rows > 0 and not state.extracted]
    needs_extraction = [state.speaker_id for state in states if state.needs_extraction]

    print("Dataset root:", DATASET_ROOT)
    print("Metadata CSV:", METADATA_CSV)
    print()
    print("Extracted speakers:", ", ".join(extracted) if extracted else "(none)")
    print("Archived speakers:", ", ".join(archived) if archived else "(none)")
    print("Missing speaker folders:", ", ".join(missing_folders) if missing_folders else "(none)")
    print("Speakers needing extraction:", ", ".join(needs_extraction) if needs_extraction else "(none)")
    print()
    print("Expected audio path pattern:")
    print("  ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/<SPEAKER>/wav/<UTTERANCE>.wav")
    print("Actual extracted Vietnamese pattern:")
    print("  ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/HQTV/wav/arctic_a0003.wav")
    print()
    print("Per-speaker audio state:")
    print("speaker_id,extracted,archive,metadata_rows,existing_audio_rows,missing_audio_rows,expected_wav_folder")

    for state in states:
        archive_name = state.archive_path.name if state.archive_path else ""
        print(
            f"{state.speaker_id},{state.extracted},{archive_name},"
            f"{state.metadata_rows},{state.existing_audio_rows},{state.missing_audio_rows},"
            f"{state.expected_wav_folder}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare L2-ARCTIC all-speaker audio folders from speaker ZIP archives."
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract missing speaker archives into the expected L2-ARCTIC raw folder.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow extraction into an existing speaker folder. Files may be overwritten by zipfile extraction.",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        help="Optional speaker IDs to inspect or extract. Defaults to all metadata/archive speakers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_metadata()
    states = build_speaker_states(df)

    if args.speakers:
        selected = {speaker.upper() for speaker in args.speakers}
        states = [state for state in states if state.speaker_id in selected]

    print_summary(states)

    if not args.extract:
        print()
        print("Dry run only. No files were extracted.")
        print("To extract missing speakers, run:")
        print(
            r"  .\ai-training\.venv\Scripts\python.exe "
            r"ai-training\scripts\prepare_l2_arctic_all_speakers_audio.py --extract"
        )
        return

    print()
    print("Extraction requested.")
    extracted_count = 0

    for state in states:
        if state.needs_extraction or (state.archive_path and args.allow_existing):
            extracted_count += int(extract_speaker_archive(state, args.allow_existing))

    print(f"Archives extracted: {extracted_count}")
    print("Re-run validation after extraction:")
    print(
        r"  .\ai-training\.venv\Scripts\python.exe "
        r"ai-training\scripts\validate_l2_arctic_all_speakers_audio.py"
    )


if __name__ == "__main__":
    main()
