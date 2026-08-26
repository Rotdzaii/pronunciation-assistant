from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import build_l2_arctic_all_speakers_correctness_v3 as v3


DEFAULT_DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
DEFAULT_OUTPUT_CSV = Path(
    "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
)
DEFAULT_AUDIT_JSON = Path(
    "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4_audit.json"
)

DATASET_VERSION = "expected_observed_v4"
EXPECTED_ANNOTATION_FILES = 3_599
EXPECTED_TOTAL_INTERVALS = 135_890
EXPECTED_RELATION_COUNTS = {
    "correct": 101_626,
    "substitution": 12_361,
    "deletion": 3_418,
    "addition": 1_044,
    "non_speech": 15_644,
    "unresolved": 1_797,
}
EXPECTED_QUALITY_COUNTS = {
    "clean": 117_405,
    "unresolved": 1_797,
    "excluded_non_speech": 15_644,
    "excluded_addition": 1_044,
}
EXPECTED_SUBSET_COUNTS = {
    "PHONE_IDENTIFICATION_ELIGIBLE": 113_987,
    "DELETION_ELIGIBLE": 3_418,
    "ADDITION_AUDIT": 1_044,
    "UNRESOLVED": 1_797,
    "EXCLUDED_NON_SPEECH": 15_644,
}
EXPECTED_UNRESOLVED_REASONS = {
    "substitution_invalid_observed_phone": 1_737,
    "deletion_invalid_expected_phone": 2,
    "addition_invalid_observed_phone": 48,
    "unrecognized_label": 10,
}
EXPECTED_UNKNOWN_NON_ERROR_LABELS = {
    "spn": 3,
    "D_": 1,
    "ER)": 1,
    "V``": 1,
    "W`": 1,
    "Y_": 1,
    "Z_": 1,
    "s": 1,
}
EXPECTED_HIGH_EXCLUSION_SPEAKERS = {"ABA", "ASI", "ERMS", "SKA", "SVBI", "YBAA"}
HIGH_SUBSTITUTION_EXCLUSION_RATE = 0.20

REAL_CANONICAL_PHONES = frozenset(v3.ARPABET_PHONES)
ERROR_CODE_TO_RELATION = {"s": "substitution", "d": "deletion", "a": "addition"}

FIELDNAMES = [
    "dataset",
    "dataset_version",
    "usage",
    "speaker_id",
    "l1",
    "gender",
    "audio_path",
    "utterance_id",
    "textgrid_path",
    "tier_name",
    "interval_index",
    "start_time",
    "end_time",
    "duration",
    "raw_label",
    "tagged_relation",
    "relation",
    "expected_phone_raw",
    "observed_phone_raw",
    "expected_phone_canonical",
    "observed_phone_canonical",
    "label_quality",
    "exclusion_reason",
    "research_subset",
    "is_main_clean",
    "is_phone_identification_eligible",
    "is_deletion",
    "is_deletion_eligible",
    "is_addition_audit",
    "is_unresolved",
]


@dataclass(frozen=True)
class LabelContract:
    raw_label: str
    tagged_relation: str
    relation: str
    expected_phone_raw: str
    observed_phone_raw: str
    expected_phone_canonical: str
    observed_phone_canonical: str
    label_quality: str
    exclusion_reason: str
    research_subset: str

    @property
    def is_main_clean(self) -> bool:
        return self.label_quality == "clean" and self.relation in {"correct", "substitution", "deletion"}

    @property
    def is_phone_identification_eligible(self) -> bool:
        return self.research_subset == "PHONE_IDENTIFICATION_ELIGIBLE"

    @property
    def is_deletion(self) -> bool:
        return self.relation == "deletion"

    @property
    def is_deletion_eligible(self) -> bool:
        return self.research_subset == "DELETION_ELIGIBLE"

    @property
    def is_addition_audit(self) -> bool:
        return self.research_subset == "ADDITION_AUDIT"

    @property
    def is_unresolved(self) -> bool:
        return self.label_quality == "unresolved"


def canonicalize_phone(phone: str) -> str:
    if phone == "<SIL>":
        return phone
    if len(phone) > 1 and phone[-1] in {"0", "1", "2"} and phone[:-1] in v3.ARPABET_VOWELS:
        return phone[:-1]
    return phone


def _split_triplet(normalized: str) -> tuple[list[str], str]:
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return parts, ""
    return parts, ERROR_CODE_TO_RELATION.get(parts[2].lower(), "")


def _unresolved(
    raw_label: str,
    tagged_relation: str,
    expected_raw: str,
    observed_raw: str,
    reason: str,
) -> LabelContract:
    return LabelContract(
        raw_label=raw_label,
        tagged_relation=tagged_relation or "unknown",
        relation="unresolved",
        expected_phone_raw=expected_raw,
        observed_phone_raw=observed_raw,
        expected_phone_canonical=canonicalize_phone(expected_raw),
        observed_phone_canonical=canonicalize_phone(observed_raw),
        label_quality="unresolved",
        exclusion_reason=reason,
        research_subset="UNRESOLVED",
    )


def derive_label(raw_label: object) -> LabelContract:
    preserved = "" if raw_label is None else str(raw_label)
    normalized = v3.normalize_label(raw_label)
    parts, tagged_relation = _split_triplet(normalized)

    if tagged_relation == "substitution":
        expected_raw, observed_raw = parts[0], parts[1]
        reasons = []
        if not v3.is_plain_arpabet_phone(expected_raw):
            reasons.append("substitution_invalid_expected_phone")
        if not v3.is_plain_arpabet_phone(observed_raw):
            reasons.append("substitution_invalid_observed_phone")
        if reasons:
            return _unresolved(preserved, tagged_relation, expected_raw, observed_raw, "+".join(reasons))
        return LabelContract(
            preserved,
            tagged_relation,
            "substitution",
            expected_raw,
            observed_raw,
            canonicalize_phone(expected_raw),
            canonicalize_phone(observed_raw),
            "clean",
            "",
            "PHONE_IDENTIFICATION_ELIGIBLE",
        )

    if tagged_relation == "deletion":
        expected_raw = parts[0]
        annotation_observed = parts[1]
        reasons = []
        if not v3.is_plain_arpabet_phone(expected_raw):
            reasons.append("deletion_invalid_expected_phone")
        if annotation_observed.lower() != "sil":
            reasons.append("deletion_observed_not_sil")
        observed_raw = "<SIL>" if annotation_observed.lower() == "sil" else annotation_observed
        if reasons:
            return _unresolved(preserved, tagged_relation, expected_raw, observed_raw, "+".join(reasons))
        return LabelContract(
            preserved,
            tagged_relation,
            "deletion",
            expected_raw,
            "<SIL>",
            canonicalize_phone(expected_raw),
            "<SIL>",
            "clean",
            "",
            "DELETION_ELIGIBLE",
        )

    if tagged_relation == "addition":
        annotation_expected, observed_raw = parts[0], parts[1]
        reasons = []
        if annotation_expected.lower() != "sil":
            reasons.append("addition_expected_not_sil")
        if not v3.is_plain_arpabet_phone(observed_raw):
            reasons.append("addition_invalid_observed_phone")
        expected_raw = annotation_expected
        if reasons:
            return _unresolved(preserved, tagged_relation, expected_raw, observed_raw, "+".join(reasons))
        return LabelContract(
            preserved,
            tagged_relation,
            "addition",
            annotation_expected,
            observed_raw,
            "<SIL>",
            canonicalize_phone(observed_raw),
            "excluded_addition",
            "addition_not_main_target",
            "ADDITION_AUDIT",
        )

    if normalized.lower() in {"", "sp", "sil"}:
        token = "empty" if not normalized else normalized.lower()
        return LabelContract(
            preserved,
            "non_speech",
            "non_speech",
            "",
            "",
            "",
            "",
            "excluded_non_speech",
            f"non_speech_{token}",
            "EXCLUDED_NON_SPEECH",
        )

    if v3.is_plain_arpabet_phone(normalized):
        canonical = canonicalize_phone(normalized)
        return LabelContract(
            preserved,
            "correct",
            "correct",
            normalized,
            normalized,
            canonical,
            canonical,
            "clean",
            "",
            "PHONE_IDENTIFICATION_ELIGIBLE",
        )

    reason = "unrecognized_label"
    if "," in normalized:
        if len(parts) != 3 or not parts[0] or not parts[1]:
            reason = "malformed_triplet"
        elif not tagged_relation:
            reason = "invalid_error_code"
    return _unresolved(preserved, "unknown", "", "", reason)


def _canonical_reference(*parts: str) -> str:
    return str(v3.CANONICAL_DATASET_ROOT.joinpath(*parts)).replace("\\", "/")


def _bool(value: bool) -> int:
    return int(value)


def build_dataset(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()
    unresolved_by_tagged_relation: Counter[str] = Counter()
    unknown_non_error_labels: Counter[str] = Counter()
    non_speech_labels: Counter[str] = Counter()
    clean_observed_inventory: Counter[str] = Counter()
    malformed_tokens_by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    speaker_stats: dict[str, Counter[str]] = defaultdict(Counter)
    suffix_case_variants: Counter[str] = Counter()
    duplicate_identities = 0
    conflicting_identities = 0
    seen_identity_labels: dict[tuple[Any, ...], str] = {}
    missing_audio_lookup = 0
    invalid_duration = 0
    annotation_files = 0

    speakers = v3.discover_speaker_ids(dataset_root)
    for speaker_id in speakers:
        speaker_info = v3.SPEAKER_INFO[speaker_id]
        utterance_metadata = v3.collect_speaker_metadata(dataset_root, speaker_id)
        for textgrid_reference, intervals in v3.iter_annotation_intervals(dataset_root, speaker_id):
            annotation_files += 1
            for interval_index, interval in enumerate(intervals, start=1):
                contract = derive_label(interval.label)
                base_info = utterance_metadata.get(interval.utterance_id)
                if base_info is None or not base_info["source_audio_exists"]:
                    missing_audio_lookup += 1
                duration = interval.end_time - interval.start_time
                if duration <= 0:
                    invalid_duration += 1

                identity = (speaker_id, interval.utterance_id, interval.start_time, interval.end_time)
                previous_label = seen_identity_labels.get(identity)
                if previous_label is not None:
                    duplicate_identities += 1
                    conflicting_identities += int(previous_label != contract.raw_label)
                else:
                    seen_identity_labels[identity] = contract.raw_label

                relation_counts[contract.relation] += 1
                quality_counts[contract.label_quality] += 1
                subset_counts[contract.research_subset] += 1
                speaker_stats[speaker_id]["total_intervals"] += 1
                speaker_stats[speaker_id][f"relation_{contract.relation}"] += 1
                if contract.tagged_relation == "substitution":
                    speaker_stats[speaker_id]["substitution_annotations"] += 1
                    if contract.relation == "substitution":
                        speaker_stats[speaker_id]["clean_substitutions"] += 1
                    else:
                        speaker_stats[speaker_id]["excluded_substitutions"] += 1
                if contract.is_unresolved:
                    unresolved_by_tagged_relation[contract.tagged_relation] += 1
                    speaker_stats[speaker_id]["unresolved_intervals"] += 1
                    for reason in contract.exclusion_reason.split("+"):
                        unresolved_reasons[reason] += 1
                    if contract.tagged_relation in {"substitution", "addition"}:
                        malformed_tokens_by_speaker[speaker_id][f"observed:{contract.observed_phone_raw}"] += 1
                    elif contract.tagged_relation == "deletion":
                        malformed_tokens_by_speaker[speaker_id][f"expected:{contract.expected_phone_raw}"] += 1
                    else:
                        malformed_tokens_by_speaker[speaker_id][f"label:{contract.raw_label}"] += 1
                        unknown_non_error_labels[v3.normalize_label(contract.raw_label)] += 1
                if contract.relation == "non_speech":
                    normalized = v3.normalize_label(contract.raw_label)
                    non_speech_labels["empty" if not normalized else normalized.lower()] += 1
                if contract.is_phone_identification_eligible:
                    clean_observed_inventory[contract.observed_phone_canonical] += 1

                normalized = v3.normalize_label(contract.raw_label)
                parts = [part.strip() for part in normalized.split(",")]
                if len(parts) == 3 and parts[2] in {"S", "D", "A"}:
                    suffix_case_variants[parts[2]] += 1

                rows.append(
                    {
                        "dataset": "l2_arctic",
                        "dataset_version": DATASET_VERSION,
                        "usage": "RESEARCH_ONLY",
                        "speaker_id": speaker_id,
                        "l1": speaker_info["l1"],
                        "gender": speaker_info["gender"],
                        "audio_path": base_info["audio_path"] if base_info else _canonical_reference(speaker_id, "wav", f"{interval.utterance_id}.wav"),
                        "utterance_id": interval.utterance_id,
                        "textgrid_path": textgrid_reference,
                        "tier_name": "phones",
                        "interval_index": interval_index,
                        "start_time": round(interval.start_time, 6),
                        "end_time": round(interval.end_time, 6),
                        "duration": round(duration, 6),
                        "raw_label": contract.raw_label,
                        "tagged_relation": contract.tagged_relation,
                        "relation": contract.relation,
                        "expected_phone_raw": contract.expected_phone_raw,
                        "observed_phone_raw": contract.observed_phone_raw,
                        "expected_phone_canonical": contract.expected_phone_canonical,
                        "observed_phone_canonical": contract.observed_phone_canonical,
                        "label_quality": contract.label_quality,
                        "exclusion_reason": contract.exclusion_reason,
                        "research_subset": contract.research_subset,
                        "is_main_clean": _bool(contract.is_main_clean),
                        "is_phone_identification_eligible": _bool(contract.is_phone_identification_eligible),
                        "is_deletion": _bool(contract.is_deletion),
                        "is_deletion_eligible": _bool(contract.is_deletion_eligible),
                        "is_addition_audit": _bool(contract.is_addition_audit),
                        "is_unresolved": _bool(contract.is_unresolved),
                    }
                )

    speaker_quality: dict[str, Any] = {}
    high_exclusion_speakers = []
    for speaker_id in speakers:
        stats = speaker_stats[speaker_id]
        substitution_total = stats["substitution_annotations"]
        excluded = stats["excluded_substitutions"]
        exclusion_rate = excluded / substitution_total if substitution_total else 0.0
        high = exclusion_rate >= HIGH_SUBSTITUTION_EXCLUSION_RATE
        if high:
            high_exclusion_speakers.append(speaker_id)
        speaker_quality[speaker_id] = {
            "total_intervals": stats["total_intervals"],
            "total_substitution_annotations": substitution_total,
            "clean_substitutions": stats["clean_substitutions"],
            "excluded_substitutions": excluded,
            "substitution_retention_percentage": 100.0 * stats["clean_substitutions"] / substitution_total,
            "substitution_exclusion_percentage": 100.0 * exclusion_rate,
            "unresolved_intervals": stats["unresolved_intervals"],
            "unresolved_interval_percentage": 100.0 * stats["unresolved_intervals"] / stats["total_intervals"],
            "high_substitution_exclusion_rate": high,
        }

    audit = {
        "dataset_version": DATASET_VERSION,
        "usage": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED", "NOT_USED_BY_CURRENT_CHECKPOINTS"],
        "source": "l2arctic_release_v5.0/<speaker>/annotation/*.TextGrid; IntervalTier phones only",
        "r3_0_gate": "R3_0_PASS_WITH_WARNINGS",
        "speakers": len(speakers),
        "annotation_files": annotation_files,
        "raw_interval_count": len(rows),
        "relation_counts": {name: relation_counts[name] for name in EXPECTED_RELATION_COUNTS},
        "quality_counts": {name: quality_counts[name] for name in EXPECTED_QUALITY_COUNTS},
        "research_subset_counts": {name: subset_counts[name] for name in EXPECTED_SUBSET_COUNTS},
        "main_clean_rows": quality_counts["clean"],
        "phone_identification_eligible_rows": subset_counts["PHONE_IDENTIFICATION_ELIGIBLE"],
        "deletion_eligible_rows": subset_counts["DELETION_ELIGIBLE"],
        "addition_audit_rows": subset_counts["ADDITION_AUDIT"],
        "unresolved_rows": subset_counts["UNRESOLVED"],
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        "unresolved_by_tagged_relation": dict(sorted(unresolved_by_tagged_relation.items())),
        "unknown_non_error_labels": dict(sorted(unknown_non_error_labels.items(), key=lambda item: (-item[1], item[0]))),
        "non_speech_labels": dict(sorted(non_speech_labels.items())),
        "clean_observed_canonical_inventory": sorted(clean_observed_inventory),
        "clean_observed_canonical_inventory_size": len(clean_observed_inventory),
        "clean_observed_canonical_distribution": dict(sorted(clean_observed_inventory.items())),
        "malformed_tokens_by_speaker": {
            speaker: dict(sorted(tokens.items(), key=lambda item: (-item[1], item[0])))
            for speaker, tokens in sorted(malformed_tokens_by_speaker.items())
        },
        "speaker_quality": speaker_quality,
        "high_exclusion_threshold_percentage": HIGH_SUBSTITUTION_EXCLUSION_RATE * 100.0,
        "high_exclusion_speakers": high_exclusion_speakers,
        "suffix_case_variants": dict(sorted(suffix_case_variants.items())),
        "sanity_checks": {
            "duplicate_interval_identities": duplicate_identities,
            "conflicting_interval_classifications": conflicting_identities,
            "missing_audio_lookup": missing_audio_lookup,
            "invalid_duration": invalid_duration,
            "category_sum_matches_raw_total": sum(relation_counts.values()) == len(rows),
            "quality_sum_matches_raw_total": sum(quality_counts.values()) == len(rows),
            "subset_sum_matches_raw_total": sum(subset_counts.values()) == len(rows),
            "main_clean_sum_matches": quality_counts["clean"] == (
                relation_counts["correct"] + relation_counts["substitution"] + relation_counts["deletion"]
            ),
            "phone_identification_sum_matches": subset_counts["PHONE_IDENTIFICATION_ELIGIBLE"] == (
                relation_counts["correct"] + relation_counts["substitution"]
            ),
            "deletion_isolated": all(
                row["observed_phone_canonical"] == "<SIL>" and row["is_deletion"] == 1
                for row in rows if row["research_subset"] == "DELETION_ELIGIBLE"
            ),
            "addition_isolated": all(
                row["relation"] == "addition" and row["is_main_clean"] == 0
                for row in rows if row["research_subset"] == "ADDITION_AUDIT"
            ),
            "no_unresolved_in_clean_phone_target": all(
                row["label_quality"] == "clean"
                and row["observed_phone_canonical"] in REAL_CANONICAL_PHONES
                for row in rows if row["research_subset"] == "PHONE_IDENTIFICATION_ELIGIBLE"
            ),
            "all_24_speakers_represented": len({row["speaker_id"] for row in rows}) == 24,
        },
    }
    return rows, audit


def validate_audit(rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    failures = []
    expected_equals = {
        "annotation_files": EXPECTED_ANNOTATION_FILES,
        "raw_interval_count": EXPECTED_TOTAL_INTERVALS,
        "relation_counts": EXPECTED_RELATION_COUNTS,
        "quality_counts": EXPECTED_QUALITY_COUNTS,
        "research_subset_counts": EXPECTED_SUBSET_COUNTS,
        "unresolved_reasons": EXPECTED_UNRESOLVED_REASONS,
        "unknown_non_error_labels": EXPECTED_UNKNOWN_NON_ERROR_LABELS,
        "clean_observed_canonical_inventory_size": 40,
    }
    for name, expected in expected_equals.items():
        if audit[name] != expected:
            failures.append(f"{name}={audit[name]!r} expected={expected!r}")
    if set(audit["clean_observed_canonical_inventory"]) != REAL_CANONICAL_PHONES:
        failures.append("clean observed canonical inventory differs from the 40-phone inventory")
    if set(audit["high_exclusion_speakers"]) != EXPECTED_HIGH_EXCLUSION_SPEAKERS:
        failures.append(
            f"high_exclusion_speakers={audit['high_exclusion_speakers']} expected={sorted(EXPECTED_HIGH_EXCLUSION_SPEAKERS)}"
        )
    sanity = audit["sanity_checks"]
    for name in (
        "duplicate_interval_identities",
        "conflicting_interval_classifications",
        "missing_audio_lookup",
        "invalid_duration",
    ):
        if sanity[name] != 0:
            failures.append(f"{name}={sanity[name]} expected=0")
    for name, value in sanity.items():
        if isinstance(value, bool) and value is not True:
            failures.append(f"{name}={value} expected=True")
    if len(rows) != EXPECTED_TOTAL_INTERVALS:
        failures.append(f"rows={len(rows)} expected={EXPECTED_TOTAL_INTERVALS}")
    if failures:
        raise RuntimeError("V4 contract validation failed before write:\n- " + "\n- ".join(failures))


def write_outputs(rows: list[dict[str, Any]], audit: dict[str, Any], output_csv: Path, audit_json: Path) -> None:
    for path in (output_csv, audit_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V4 output: {path}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    written_count = 0
    written_subsets: Counter[str] = Counter()
    written_speakers = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            written_count += 1
            written_subsets[row["research_subset"]] += 1
            written_speakers.add(row["speaker_id"])
    if written_count != EXPECTED_TOTAL_INTERVALS or dict(written_subsets) != EXPECTED_SUBSET_COUNTS or len(written_speakers) != 24:
        raise RuntimeError(
            f"Written V4 CSV validation failed: rows={written_count}, subsets={dict(written_subsets)}, speakers={len(written_speakers)}"
        )
    audit["written_csv_validation"] = {
        "rows": written_count,
        "research_subset_counts": dict(written_subsets),
        "speakers": len(written_speakers),
        "status": "PASS",
    }
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the research-only L2-ARCTIC expected/observed V4 full ledger.")
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
    print("R3_V4_CONTRACT_BUILD_PASS")
    print("source=manual /annotation/*.TextGrid tier=phones")
    print(f"raw_intervals={audit['raw_interval_count']}")
    print(f"relation_counts={audit['relation_counts']}")
    print(f"research_subset_counts={audit['research_subset_counts']}")
    print(f"unresolved_reasons={audit['unresolved_reasons']}")


if __name__ == "__main__":
    main()
