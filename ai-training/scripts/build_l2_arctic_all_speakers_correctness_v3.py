from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


DEFAULT_DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
DEFAULT_OUTPUT_CSV = Path(
    "ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv"
)
DEFAULT_AUDIT_JSON = Path(
    "ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3_audit.json"
)
CANONICAL_DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")

EXPECTED_ANNOTATION_FILES = 3_599
EXPECTED_TOTAL_INTERVALS = 135_890
EXPECTED_DISTRIBUTION = {
    "correct": 101_626,
    "substitution": 14_098,
    "deletion": 3_420,
    "addition": 1_092,
    "non_speech": 15_644,
    "unknown": 10,
}
EXPECTED_TRAINING_ROWS = 120_236
EXPECTED_UNKNOWN_LABELS = {
    "spn": 3,
    "D_": 1,
    "ER)": 1,
    "V``": 1,
    "W`": 1,
    "Y_": 1,
    "Z_": 1,
    "s": 1,
}

TRAINING_CLASSES = ("correct", "substitution", "deletion", "addition")
ALL_CLASSES = (*TRAINING_CLASSES, "non_speech", "unknown")
ERROR_CODE_TO_CLASS = {
    "s": "substitution",
    "d": "deletion",
    "a": "addition",
}

ARPABET_PHONES = frozenset(
    "AA AE AH AO AW AX AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
    "OW OY P R S SH T TH UH UW V W Y Z ZH".split()
)
ARPABET_VOWELS = frozenset("AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW".split())

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

FIELDNAMES = [
    "dataset",
    "dataset_version",
    "speaker_id",
    "l1",
    "gender",
    "split",
    "audio_path",
    "utterance_id",
    "target_text",
    "textgrid_path",
    "tier_name",
    "raw_label",
    "label",
    "expected_phone",
    "error_type",
    "start_time",
    "end_time",
    "original_duration",
    "context_start_time",
    "context_end_time",
    "context_duration",
]


@dataclass(frozen=True)
class TextGridInterval:
    speaker_id: str
    utterance_id: str
    start_time: float
    end_time: float
    label: str


@dataclass(frozen=True)
class LabelClassification:
    error_type: str
    normalized_label: str
    expected_phone: str


def normalize_label(label: object) -> str:
    if label is None:
        return ""
    return " ".join(str(label).strip().split())


def is_plain_arpabet_phone(label: object) -> bool:
    normalized = normalize_label(label)
    if normalized in ARPABET_PHONES:
        return True
    return (
        len(normalized) > 1
        and normalized[-1] in {"0", "1", "2"}
        and normalized[:-1] in ARPABET_VOWELS
    )


def _error_parts(label: object) -> tuple[list[str], str | None]:
    normalized = normalize_label(label)
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return parts, None
    return parts, ERROR_CODE_TO_CLASS.get(parts[-1].lower())


def classify_label(label: object) -> LabelClassification:
    normalized = normalize_label(label)
    parts, error_type = _error_parts(normalized)

    if error_type is not None:
        return LabelClassification(error_type, normalized, parts[0])

    if normalized.lower() in {"", "sp", "sil"}:
        return LabelClassification("non_speech", normalized, "")

    if is_plain_arpabet_phone(normalized):
        return LabelClassification("correct", normalized, normalized)

    return LabelClassification("unknown", normalized, "")


def _quoted_value(line: str) -> str:
    match = re.search(r'"(.*)"', line)
    return match.group(1) if match else ""


def parse_phone_intervals(
    lines: Iterable[str], speaker_id: str, utterance_id: str
) -> list[TextGridInterval]:
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
            try:
                pending_start = float(line.split("=", 1)[1].strip())
            except ValueError:
                pending_start = None
            continue

        if line.startswith("xmax ="):
            try:
                pending_end = float(line.split("=", 1)[1].strip())
            except ValueError:
                pending_end = None
            continue

        if line.startswith("text =") and pending_start is not None and pending_end is not None:
            if current_tier.lower() == "phones":
                intervals.append(
                    TextGridInterval(
                        speaker_id=speaker_id,
                        utterance_id=utterance_id,
                        start_time=pending_start,
                        end_time=pending_end,
                        label=_quoted_value(line),
                    )
                )
            pending_start = None
            pending_end = None

    return intervals


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

    if context_end - context_start < MIN_CONTEXT_DURATION:
        half = MIN_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = center + half

    if context_end - context_start > MAX_CONTEXT_DURATION:
        half = MAX_CONTEXT_DURATION / 2.0
        context_start = max(0.0, center - half)
        context_end = context_start + MAX_CONTEXT_DURATION

    return round(context_start, 6), round(context_end, 6)


def discover_speaker_ids(dataset_root: Path) -> list[str]:
    speakers = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "annotation").is_dir()
    )
    expected = set(SPEAKER_INFO)
    discovered = set(speakers)
    if discovered != expected:
        raise RuntimeError(
            "Expected exactly the 24 scripted-speech speaker annotation directories; "
            f"missing={sorted(expected - discovered)}, extra={sorted(discovered - expected)}"
        )
    return speakers


def _canonical_reference(*parts: str) -> str:
    return str(CANONICAL_DATASET_ROOT.joinpath(*parts)).replace("\\", "/")


def collect_speaker_metadata(dataset_root: Path, speaker_id: str) -> dict[str, dict[str, object]]:
    speaker_dir = dataset_root / speaker_id
    transcript_dir = speaker_dir / "transcript"
    metadata: dict[str, dict[str, object]] = {}

    for index, wav_path in enumerate(sorted((speaker_dir / "wav").glob("*.wav"))):
        utterance_id = wav_path.stem
        transcript_path = transcript_dir / f"{utterance_id}.txt"
        target_text = ""
        if transcript_path.exists():
            target_text = " ".join(
                transcript_path.read_text(encoding="utf-8", errors="replace").split()
            )
        metadata[utterance_id] = {
            "split": get_split(index),
            "audio_path": _canonical_reference(speaker_id, "wav", wav_path.name),
            "target_text": target_text,
            "source_audio_exists": wav_path.exists(),
        }
    return metadata


def iter_annotation_intervals(dataset_root: Path, speaker_id: str):
    annotation_dir = dataset_root / speaker_id / "annotation"
    for textgrid_path in sorted(annotation_dir.glob("*.TextGrid")):
        lines = textgrid_path.read_text(encoding="utf-8", errors="replace").splitlines()
        yield (
            _canonical_reference(speaker_id, "annotation", textgrid_path.name),
            parse_phone_intervals(lines, speaker_id, textgrid_path.stem),
        )


def _predicate_count(label: str) -> int:
    _, error_type = _error_parts(label)
    return sum(
        (
            error_type is not None,
            normalize_label(label).lower() in {"", "sp", "sil"},
            is_plain_arpabet_phone(label),
        )
    )


def build_dataset(dataset_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    distribution: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()
    unknown_locations: list[dict[str, object]] = []
    non_speech_labels: Counter[str] = Counter()
    speaker_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    seen_intervals: set[tuple[object, ...]] = set()
    training_speakers: set[str] = set()
    annotation_files = 0
    duplicate_intervals = 0
    overlapping_rules = 0
    missing_audio_lookup = 0
    invalid_duration = 0
    valid_plain_phone_dropped = 0
    error_tagged_mapped_correct = 0
    non_speech_mapped_correct = 0

    speakers = discover_speaker_ids(dataset_root)
    for speaker_id in speakers:
        speaker_info = SPEAKER_INFO[speaker_id]
        utterance_metadata = collect_speaker_metadata(dataset_root, speaker_id)

        for textgrid_reference, intervals in iter_annotation_intervals(dataset_root, speaker_id):
            annotation_files += 1
            for interval in intervals:
                classification = classify_label(interval.label)
                normalized = classification.normalized_label
                _, tagged_error = _error_parts(normalized)
                is_non_speech = normalized.lower() in {"", "sp", "sil"}
                is_plain = is_plain_arpabet_phone(normalized)

                overlapping_rules += int(_predicate_count(normalized) > 1)
                valid_plain_phone_dropped += int(
                    is_plain and classification.error_type != "correct"
                )
                error_tagged_mapped_correct += int(
                    tagged_error is not None and classification.error_type == "correct"
                )
                non_speech_mapped_correct += int(
                    is_non_speech and classification.error_type == "correct"
                )

                key = (
                    textgrid_reference,
                    interval.start_time,
                    interval.end_time,
                    interval.label,
                )
                if key in seen_intervals:
                    duplicate_intervals += 1
                seen_intervals.add(key)

                distribution[classification.error_type] += 1
                speaker_distribution[speaker_id][classification.error_type] += 1

                if classification.error_type == "non_speech":
                    non_speech_labels[normalized] += 1
                    continue

                if classification.error_type == "unknown":
                    unknown_labels[normalized] += 1
                    unknown_locations.append(
                        {
                            "speaker_id": speaker_id,
                            "utterance_id": interval.utterance_id,
                            "textgrid_path": textgrid_reference,
                            "start_time": interval.start_time,
                            "end_time": interval.end_time,
                            "raw_label": interval.label,
                        }
                    )
                    continue

                base_info = utterance_metadata.get(interval.utterance_id)
                if base_info is None or not base_info["source_audio_exists"]:
                    missing_audio_lookup += 1
                    continue

                original_duration = interval.end_time - interval.start_time
                if original_duration <= 0:
                    invalid_duration += 1
                    continue

                context_start, context_end = add_context_window(
                    interval.start_time, interval.end_time
                )
                rows.append(
                    {
                        "dataset": "l2_arctic",
                        "dataset_version": "correctness_v3",
                        "speaker_id": speaker_id,
                        "l1": speaker_info["l1"],
                        "gender": speaker_info["gender"],
                        "split": base_info["split"],
                        "audio_path": base_info["audio_path"],
                        "utterance_id": interval.utterance_id,
                        "target_text": base_info["target_text"],
                        "textgrid_path": textgrid_reference,
                        "tier_name": "phones",
                        "raw_label": interval.label,
                        "label": interval.label,
                        "expected_phone": classification.expected_phone,
                        "error_type": classification.error_type,
                        "start_time": round(interval.start_time, 6),
                        "end_time": round(interval.end_time, 6),
                        "original_duration": round(original_duration, 6),
                        "context_start_time": context_start,
                        "context_end_time": context_end,
                        "context_duration": round(context_end - context_start, 6),
                    }
                )
                training_speakers.add(speaker_id)

    distribution_dict = {name: distribution[name] for name in ALL_CLASSES}
    training_distribution = Counter(str(row["error_type"]) for row in rows)
    invalid_training_rows = sum(
        1
        for row in rows
        if not str(row["raw_label"]).strip()
        or not str(row["expected_phone"]).strip()
        or not str(row["error_type"]).strip()
        or row["error_type"] not in TRAINING_CLASSES
    )

    audit: dict[str, object] = {
        "dataset_version": "correctness_v3",
        "usage": [
            "RESEARCH_ONLY",
            "NOT_USED_BY_RUNTIME",
            "NOT_USED_BY_CURRENT_CHECKPOINT",
        ],
        "source": "l2arctic_release_v5.0/<speaker>/annotation/*.TextGrid; tier=phones",
        "speakers": len(speakers),
        "annotation_files": annotation_files,
        "input_total_intervals": sum(distribution.values()),
        "distribution": distribution_dict,
        "training_rows": len(rows),
        "training_distribution": {
            name: training_distribution[name] for name in TRAINING_CLASSES
        },
        "non_speech_labels": {
            "empty": non_speech_labels[""],
            "sp": sum(count for label, count in non_speech_labels.items() if label.lower() == "sp"),
            "sil": sum(count for label, count in non_speech_labels.items() if label.lower() == "sil"),
        },
        "unknown_labels": dict(
            sorted(unknown_labels.items(), key=lambda item: (-item[1], item[0]))
        ),
        "unknown_locations": unknown_locations,
        "speaker_distribution": {
            speaker: {name: speaker_distribution[speaker][name] for name in ALL_CLASSES}
            for speaker in speakers
        },
        "sanity_checks": {
            "duplicate_intervals": duplicate_intervals,
            "overlapping_label_rules": overlapping_rules,
            "missing_audio_lookup": missing_audio_lookup,
            "invalid_duration": invalid_duration,
            "valid_plain_phone_dropped": valid_plain_phone_dropped,
            "error_tagged_mapped_correct": error_tagged_mapped_correct,
            "non_speech_mapped_correct": non_speech_mapped_correct,
            "invalid_training_rows": invalid_training_rows,
            "all_24_speakers_in_training": len(training_speakers) == len(SPEAKER_INFO),
            "category_sum_matches_total": sum(distribution_dict.values())
            == EXPECTED_TOTAL_INTERVALS,
            "training_sum_matches_rows": sum(training_distribution.values()) == len(rows),
        },
    }
    return rows, audit


def validate_audit(rows: list[dict[str, object]], audit: dict[str, object]) -> None:
    failures: list[str] = []
    if audit["annotation_files"] != EXPECTED_ANNOTATION_FILES:
        failures.append(
            f"annotation_files={audit['annotation_files']} expected={EXPECTED_ANNOTATION_FILES}"
        )
    if audit["input_total_intervals"] != EXPECTED_TOTAL_INTERVALS:
        failures.append(
            f"total={audit['input_total_intervals']} expected={EXPECTED_TOTAL_INTERVALS}"
        )
    if audit["distribution"] != EXPECTED_DISTRIBUTION:
        failures.append(
            f"distribution={audit['distribution']} expected={EXPECTED_DISTRIBUTION}"
        )
    if audit["unknown_labels"] != EXPECTED_UNKNOWN_LABELS:
        failures.append(
            f"unknown_labels={audit['unknown_labels']} expected={EXPECTED_UNKNOWN_LABELS}"
        )
    if len(rows) != EXPECTED_TRAINING_ROWS:
        failures.append(f"training_rows={len(rows)} expected={EXPECTED_TRAINING_ROWS}")

    sanity = audit["sanity_checks"]
    zero_checks = (
        "duplicate_intervals",
        "overlapping_label_rules",
        "missing_audio_lookup",
        "invalid_duration",
        "valid_plain_phone_dropped",
        "error_tagged_mapped_correct",
        "non_speech_mapped_correct",
        "invalid_training_rows",
    )
    for name in zero_checks:
        if sanity[name] != 0:
            failures.append(f"{name}={sanity[name]} expected=0")
    for name in (
        "all_24_speakers_in_training",
        "category_sum_matches_total",
        "training_sum_matches_rows",
    ):
        if sanity[name] is not True:
            failures.append(f"{name}={sanity[name]} expected=True")

    if failures:
        raise RuntimeError("V3 validation failed before write:\n- " + "\n- ".join(failures))


def write_outputs(
    rows: list[dict[str, object]],
    audit: dict[str, object],
    output_csv: Path,
    audit_json: Path,
) -> None:
    for path in (output_csv, audit_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V3 output: {path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        written_rows = list(csv.DictReader(file))
    if len(written_rows) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            f"Written CSV row count changed: {len(written_rows)} != {EXPECTED_TRAINING_ROWS}"
        )
    if any(
        not row.get("raw_label")
        or not row.get("expected_phone")
        or row.get("error_type") not in TRAINING_CLASSES
        for row in written_rows
    ):
        raise RuntimeError("Written CSV contains blank phone/label fields or excluded classes")

    audit["written_csv_validation"] = {
        "rows": len(written_rows),
        "blank_required_label_fields": 0,
        "excluded_class_rows": 0,
        "status": "PASS",
    }
    audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the research-only L2-ARCTIC four-class correctness V3 metadata."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    rows, audit = build_dataset(args.dataset_root)
    validate_audit(rows, audit)
    write_outputs(rows, audit, args.output_csv, args.audit_json)

    print("V3_DATASET_BUILD_PASS")
    print("source=manual /annotation/*.TextGrid tier=phones")
    print(f"speakers={audit['speakers']}")
    print(f"annotation_files={audit['annotation_files']}")
    print(f"total_intervals={audit['input_total_intervals']}")
    for name in ALL_CLASSES:
        print(f"{name}={audit['distribution'][name]}")
    print(f"training_rows={audit['training_rows']}")
    print(f"output_csv={args.output_csv}")
    print(f"audit_json={args.audit_json}")


if __name__ == "__main__":
    main()
