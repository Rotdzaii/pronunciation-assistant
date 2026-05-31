from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import pandas as pd


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")

CONTEXT_SECONDS = 0.25
MIN_CONTEXT_DURATION = 0.50
MAX_CONTEXT_DURATION = 1.00

SPEAKER_INFO = {
    "ABA": {"l1": "Arabic", "gender": "M"},
    "SKA": {"l1": "Arabic", "gender": "F"},
    "YBAA": {"l1": "Arabic", "gender": "M"},
    "ZHAA": {"l1": "Arabic", "gender": "F"},
    "BWC": {"l1": "Chinese", "gender": "M"},
    "LXC": {"l1": "Chinese", "gender": "F"},
    "NCC": {"l1": "Chinese", "gender": "F"},
    "TXHC": {"l1": "Chinese", "gender": "M"},
    "ASI": {"l1": "Hindi", "gender": "M"},
    "RRBI": {"l1": "Hindi", "gender": "M"},
    "SVBI": {"l1": "Hindi", "gender": "F"},
    "TNI": {"l1": "Hindi", "gender": "F"},
    "HJK": {"l1": "Korean", "gender": "F"},
    "HKK": {"l1": "Korean", "gender": "M"},
    "YDCK": {"l1": "Korean", "gender": "F"},
    "YKWK": {"l1": "Korean", "gender": "M"},
    "EBVS": {"l1": "Spanish", "gender": "M"},
    "ERMS": {"l1": "Spanish", "gender": "M"},
    "MBMPS": {"l1": "Spanish", "gender": "F"},
    "NJS": {"l1": "Spanish", "gender": "F"},
    "HQTV": {"l1": "Vietnamese", "gender": "M"},
    "PNV": {"l1": "Vietnamese", "gender": "F"},
    "THV": {"l1": "Vietnamese", "gender": "F"},
    "TLV": {"l1": "Vietnamese", "gender": "M"},
}


@dataclass
class TextGridInterval:
    speaker_id: str
    utterance_id: str
    tier_name: str
    start_time: float
    end_time: float
    label: str


def _quoted_value(line: str) -> str:
    match = re.search(r'"(.*)"', line)
    return match.group(1) if match else ""


def parse_textgrid_lines(lines: list[str], speaker_id: str, utterance_id: str) -> list[TextGridInterval]:
    intervals: list[TextGridInterval] = []
    current_tier = ""
    in_interval_tier = False
    pending_start: float | None = None
    pending_end: float | None = None

    for raw_line in lines:
        line = raw_line.strip()

        if line.startswith("class ="):
            in_interval_tier = _quoted_value(line) == "IntervalTier"
            current_tier = ""
            pending_start = None
            pending_end = None
            continue

        if in_interval_tier and line.startswith("name ="):
            current_tier = _quoted_value(line)
            continue

        if not in_interval_tier or not current_tier:
            continue

        if line.startswith("intervals ["):
            pending_start = None
            pending_end = None
            continue

        if line.startswith("xmin ="):
            pending_start = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("xmax ="):
            pending_end = float(line.split("=", 1)[1].strip())
            continue

        if line.startswith("text =") and pending_start is not None and pending_end is not None:
            intervals.append(
                TextGridInterval(
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    tier_name=current_tier,
                    start_time=pending_start,
                    end_time=pending_end,
                    label=_quoted_value(line),
                )
            )
            pending_start = None
            pending_end = None

    return intervals


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def normalize_error_type(label: str) -> str:
    normalized = normalize_label(label).lower()

    if not normalized:
        return "unknown"

    parts = [part.strip().lower() for part in normalized.split(",")]
    code = parts[-1] if len(parts) >= 2 else ""

    if code in {"s", "sub", "substitution"}:
        return "substitution"

    if code in {"d", "del", "delete", "deletion"}:
        return "deletion"

    if code in {"a", "add", "addition", "i", "ins", "insert", "insertion"}:
        return "addition"

    return "unknown"


def get_split(index: int) -> str:
    if index < 906:
        return "train"

    if index < 1019:
        return "val"

    return "test"


def add_context_window(start_time: float, end_time: float) -> tuple[float, float]:
    center = (start_time + end_time) / 2.0
    context_start = max(0.0, start_time - CONTEXT_SECONDS)
    context_end = end_time + CONTEXT_SECONDS
    duration = context_end - context_start

    if duration < MIN_CONTEXT_DURATION:
        half = MIN_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = center + half

    duration = context_end - context_start

    if duration > MAX_CONTEXT_DURATION:
        half = MAX_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = context_start + MAX_CONTEXT_DURATION

    return round(context_start, 6), round(context_end, 6)


def discover_speaker_ids() -> list[str]:
    speaker_ids = {
        path.name
        for path in DATASET_ROOT.iterdir()
        if path.is_dir() and (path / "annotation").exists()
    }
    speaker_ids.update(path.stem for path in DATASET_ROOT.glob("*.zip") if path.stem != "suitcase_corpus")
    return sorted(speaker_ids)


def read_text_file_from_zip(zip_file: zipfile.ZipFile, member: str) -> str:
    try:
        return zip_file.read(member).decode("utf-8", errors="replace")
    except KeyError:
        return ""


def collect_folder_metadata(speaker_id: str) -> dict[str, dict[str, str]]:
    speaker_dir = DATASET_ROOT / speaker_id
    transcript_dir = speaker_dir / "transcript"
    wav_files = sorted((speaker_dir / "wav").glob("*.wav"))
    metadata = {}

    for index, wav_path in enumerate(wav_files):
        utterance_id = wav_path.stem
        transcript_path = transcript_dir / f"{utterance_id}.txt"
        target_text = ""

        if transcript_path.exists():
            target_text = " ".join(transcript_path.read_text(encoding="utf-8", errors="replace").split())

        metadata[utterance_id] = {
            "split": get_split(index),
            "audio_path": str(wav_path).replace("\\", "/"),
            "target_text": target_text,
            "audio_exists": str(wav_path.exists()),
        }

    return metadata


def collect_zip_metadata(speaker_id: str, zip_path: Path) -> dict[str, dict[str, str]]:
    metadata = {}

    with zipfile.ZipFile(zip_path) as zip_file:
        wav_members = sorted(
            member for member in zip_file.namelist()
            if member.startswith(f"{speaker_id}/wav/") and member.endswith(".wav")
        )

        for index, wav_member in enumerate(wav_members):
            utterance_id = Path(wav_member).stem
            transcript_member = f"{speaker_id}/transcript/{utterance_id}.txt"
            target_text = " ".join(read_text_file_from_zip(zip_file, transcript_member).split())
            extracted_audio_path = DATASET_ROOT / speaker_id / "wav" / f"{utterance_id}.wav"

            metadata[utterance_id] = {
                "split": get_split(index),
                "audio_path": str(extracted_audio_path).replace("\\", "/"),
                "target_text": target_text,
                "audio_exists": str(extracted_audio_path.exists()),
            }

    return metadata


def collect_speaker_metadata(speaker_id: str) -> dict[str, dict[str, str]]:
    speaker_dir = DATASET_ROOT / speaker_id

    if speaker_dir.exists():
        return collect_folder_metadata(speaker_id)

    zip_path = DATASET_ROOT / f"{speaker_id}.zip"

    if zip_path.exists():
        return collect_zip_metadata(speaker_id, zip_path)

    return {}


def iter_annotation_intervals(speaker_id: str):
    speaker_dir = DATASET_ROOT / speaker_id

    if speaker_dir.exists():
        for textgrid_path in sorted((speaker_dir / "annotation").glob("*.TextGrid")):
            lines = textgrid_path.read_text(encoding="utf-8", errors="replace").splitlines()
            yield str(textgrid_path).replace("\\", "/"), parse_textgrid_lines(lines, speaker_id, textgrid_path.stem)
        return

    zip_path = DATASET_ROOT / f"{speaker_id}.zip"

    if not zip_path.exists():
        return

    with zipfile.ZipFile(zip_path) as zip_file:
        members = sorted(
            member for member in zip_file.namelist()
            if member.startswith(f"{speaker_id}/annotation/") and member.endswith(".TextGrid")
        )

        for member in members:
            lines = zip_file.read(member).decode("utf-8", errors="replace").splitlines()
            textgrid_path = str(DATASET_ROOT / member).replace("\\", "/")
            yield textgrid_path, parse_textgrid_lines(lines, speaker_id, Path(member).stem)


def build_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    stats = {
        "speakers": 0,
        "annotation_files": 0,
        "missing_audio_lookup": 0,
        "raw_error_rows": 0,
    }

    for speaker_id in discover_speaker_ids():
        speaker_info = SPEAKER_INFO.get(speaker_id, {"l1": "Unknown", "gender": ""})
        utterance_metadata = collect_speaker_metadata(speaker_id)
        stats["speakers"] += 1

        for textgrid_path, intervals in iter_annotation_intervals(speaker_id):
            stats["annotation_files"] += 1

            for interval in intervals:
                if interval.tier_name.lower() != "phones":
                    continue

                error_type = normalize_error_type(interval.label)

                if error_type not in {"addition", "deletion", "substitution"}:
                    continue

                stats["raw_error_rows"] += 1
                base_info = utterance_metadata.get(interval.utterance_id)

                if base_info is None:
                    stats["missing_audio_lookup"] += 1
                    continue

                original_duration = float(interval.end_time) - float(interval.start_time)
                context_start, context_end = add_context_window(interval.start_time, interval.end_time)

                rows.append(
                    {
                        "dataset": "l2_arctic",
                        "speaker_id": speaker_id,
                        "l1": speaker_info["l1"],
                        "gender": speaker_info["gender"],
                        "split": base_info["split"],
                        "audio_path": base_info["audio_path"],
                        "utterance_id": interval.utterance_id,
                        "target_text": base_info["target_text"],
                        "tier_name": interval.tier_name,
                        "label": interval.label,
                        "error_type": error_type,
                        "start_time": round(float(interval.start_time), 6),
                        "end_time": round(float(interval.end_time), 6),
                        "original_duration": round(original_duration, 6),
                        "context_start_time": context_start,
                        "context_end_time": context_end,
                        "context_duration": round(context_end - context_start, 6),
                    }
                )

    df = pd.DataFrame(rows)

    dedup_columns = ["speaker_id", "utterance_id", "start_time", "end_time", "error_type"]
    stats["duplicates_removed"] = len(df) - len(df.drop_duplicates(subset=dedup_columns))
    df = df.drop_duplicates(subset=dedup_columns).copy()

    return df, stats


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    df, stats = build_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    audio_exists = df["audio_path"].apply(lambda path: Path(path).exists())

    print("Discovered speakers:", stats["speakers"])
    print("Annotation TextGrid files processed:", stats["annotation_files"])
    print("Raw phones error rows:", stats["raw_error_rows"])
    print("Missing audio lookup rows:", stats["missing_audio_lookup"])
    print("Removed duplicates:", stats["duplicates_removed"])
    print("Output rows:", len(df))
    print()
    print("Output columns:")
    print(list(df.columns))
    print()
    print("Output error distribution:")
    print(df["error_type"].value_counts())
    print()
    print("Output L1 distribution:")
    print(df["l1"].value_counts())
    print()
    print("Output split distribution:")
    print(df["split"].value_counts())
    print()
    print("Audio exists:")
    print(audio_exists.value_counts())
    print()
    print("Original duration summary:")
    print(df.groupby("error_type")["original_duration"].describe().round(4))
    print()
    print("Context duration summary:")
    print(df.groupby("error_type")["context_duration"].describe().round(4))
    print()
    print(f"Saved all-speaker v2 dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
