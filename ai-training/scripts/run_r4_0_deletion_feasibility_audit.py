from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
EXPECTED_V4_SHA256 = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_0_deletion_feasibility"

TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
SPLITS = {"train": TRAIN_SPEAKERS, "validation": VALIDATION_SPEAKERS, "test": TEST_SPEAKERS}
EXPECTED_DELETION_COUNTS = {"train": 1612, "validation": 979, "test": 827}

SAMPLE_RATE = 16_000
WINDOWS = (0.30, 0.50, 1.00)
SILENCE_DBFS = -40.0
FRAME_SAMPLES = 320
HOP_SAMPLES = 160
MATCH_BIN_SECONDS = 0.010
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_audio_root() -> Path:
    value = os.environ.get("L2_ARCTIC_ROOT", "").strip()
    if not value:
        raise RuntimeError("L2_ARCTIC_ROOT is required")
    root = Path(value).resolve()
    if not root.is_dir() or root.name.lower() != "l2arctic_release_v5.0":
        raise RuntimeError(f"Invalid L2_ARCTIC_ROOT: {root}")
    return root


def resolve_audio_path(reference: str, root: Path) -> Path:
    parts = Path(reference.replace("\\", "/")).parts
    indexes = [i for i, part in enumerate(parts) if part.lower() == "l2arctic_release_v5.0"]
    if not indexes:
        raise RuntimeError(f"Audio reference lacks corpus marker: {reference}")
    return root.joinpath(*parts[indexes[-1] + 1 :])


def split_for(speaker: str) -> str:
    for split, speakers in SPLITS.items():
        if speaker in speakers:
            return split
    raise RuntimeError(f"Speaker outside S1 split: {speaker}")


def load_rows(audio_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    actual_sha = sha256_file(V4_PATH)
    if actual_sha != EXPECTED_V4_SHA256:
        raise RuntimeError(f"V4 SHA mismatch: {actual_sha}")
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    if len(ledger) != 135_890:
        raise RuntimeError(f"Raw ledger row mismatch: {len(ledger)}")

    clean: list[dict[str, Any]] = []
    counts = {split: Counter() for split in SPLITS}
    for source_index, source in enumerate(ledger):
        split = split_for(source["speaker_id"])
        if source["label_quality"] != "clean" or source["relation"] not in {"correct", "substitution", "deletion"}:
            continue
        row = dict(source)
        row.update({
            "_source_index": source_index,
            "_split": split,
            "_start": float(source["start_time"]),
            "_end": float(source["end_time"]),
            "_duration": float(source["duration"]),
            "_truth": 1 if source["relation"] == "deletion" else 0,
        })
        if row["_end"] <= row["_start"] or abs(row["_duration"] - (row["_end"] - row["_start"])) > 1e-5:
            raise RuntimeError(f"Invalid interval at V4 row {source_index + 2}")
        counts[split][source["relation"]] += 1
        # R4 TEST stays metadata-only. Its paths are never resolved.
        if split in {"train", "validation"}:
            row["_audio_path"] = resolve_audio_path(source["audio_path"], audio_root)
        clean.append(row)

    deletion_counts = {split: counts[split]["deletion"] for split in SPLITS}
    if deletion_counts != EXPECTED_DELETION_COUNTS or sum(deletion_counts.values()) != 3418:
        raise RuntimeError(f"Deletion split counts changed: {deletion_counts}")
    sets = [set(speakers) for speakers in SPLITS.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("S1 split overlap")

    deletion_phones = {
        row["expected_phone_canonical"]
        for row in clean
        if row["_split"] in {"train", "validation"} and row["relation"] == "deletion"
    }
    analysis = [
        row for row in clean
        if row["_split"] in {"train", "validation"}
        and row["expected_phone_canonical"] in deletion_phones
    ]
    return ledger, analysis, {
        "v4_sha256": actual_sha,
        "raw_rows": len(ledger),
        "deletion_counts": deletion_counts,
        "relation_counts": {split: dict(counts[split]) for split in SPLITS},
        "deletion_phone_count_train_validation": len(deletion_phones),
        "deletion_phones_train_validation": sorted(deletion_phones),
        "analysis_scope": "TRAIN+VALIDATION clean correct/substitution/deletion restricted to expected phones with deletion support",
        "analysis_rows": len(analysis),
        "test_audio_paths_resolved": False,
        "test_audio_accessed": False,
    }


def fixed_window(audio: np.ndarray, center_seconds: float, seconds: float) -> tuple[np.ndarray, bool]:
    samples = int(round(seconds * SAMPLE_RATE))
    center = int(round(center_seconds * SAMPLE_RATE))
    requested_start = center - samples // 2
    requested_end = requested_start + samples
    source_start = max(0, requested_start)
    source_end = min(len(audio), requested_end)
    if source_end <= source_start:
        raise RuntimeError("Empty fixed context window")
    output = np.zeros(samples, dtype=np.float32)
    destination_start = source_start - requested_start
    output[destination_start : destination_start + source_end - source_start] = audio[source_start:source_end]
    return output, requested_start < 0 or requested_end > len(audio)


def interval_window(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(len(audio), int(round(end * SAMPLE_RATE)))
    if last <= first:
        raise RuntimeError(f"Empty annotated interval: {start}, {end}")
    return np.asarray(audio[first:last], dtype=np.float32)


def rms_dbfs(values: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(values, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-10))


def acoustic_stats(values: np.ndarray) -> dict[str, float]:
    dbfs = rms_dbfs(values)
    if len(values) <= 1:
        zcr = 0.0
    else:
        zcr = float(np.mean(np.signbit(values[:-1]) != np.signbit(values[1:])))
    if len(values) <= FRAME_SAMPLES:
        frame_db = [dbfs]
    else:
        starts = list(range(0, len(values) - FRAME_SAMPLES + 1, HOP_SAMPLES))
        if starts[-1] != len(values) - FRAME_SAMPLES:
            starts.append(len(values) - FRAME_SAMPLES)
        frame_db = [rms_dbfs(values[start : start + FRAME_SAMPLES]) for start in starts]
    return {
        "rms_dbfs": dbfs,
        "silence_ratio": float(np.mean(np.asarray(frame_db) <= SILENCE_DBFS)),
        "zero_crossing_rate": zcr,
    }


def extract_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["_audio_path"]].append(row)
    errors: list[str] = []
    rates: Counter[int] = Counter()
    total_hours = 0.0
    started = time.perf_counter()
    for path, path_rows in tqdm(sorted(by_path.items(), key=lambda item: str(item[0])), desc="R4 TRAIN+VALIDATION audio audit"):
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or info.duration <= 0:
                errors.append(f"invalid: {path}")
                continue
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if not len(audio) or not np.isfinite(audio).all():
                errors.append(f"decoded empty/non-finite: {path}")
                continue
            rates[info.samplerate] += 1
            total_hours += info.duration
            for row in path_rows:
                central = interval_window(audio, row["_start"], row["_end"])
                row["_central"] = acoustic_stats(central)
                center = (row["_start"] + row["_end"]) / 2.0
                for seconds in WINDOWS:
                    context, padded = fixed_window(audio, center, seconds)
                    row[f"_context_{seconds:.2f}"] = acoustic_stats(context)
                    row[f"_padded_{seconds:.2f}"] = padded
                flank_start = max(0, int(round((center - 0.25) * SAMPLE_RATE)))
                left_end = max(flank_start, int(round(row["_start"] * SAMPLE_RATE)))
                right_start = min(len(audio), int(round(row["_end"] * SAMPLE_RATE)))
                flank_end = min(len(audio), int(round((center + 0.25) * SAMPLE_RATE)))
                flanks = np.concatenate((audio[flank_start:left_end], audio[right_start:flank_end]))
                row["_local_snr_db"] = row["_central"]["rms_dbfs"] - rms_dbfs(flanks) if len(flanks) else None
                row["_audio_duration"] = len(audio) / SAMPLE_RATE
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError(f"Audio audit failed ({len(errors)}):\n" + "\n".join(errors[:20]))
    return {
        "scope": "TRAIN+VALIDATION only",
        "rows": len(rows),
        "unique_wav_files": len(by_path),
        "original_sample_rates": dict(sorted(rates.items())),
        "decoded_sample_rate": SAMPLE_RATE,
        "total_unique_audio_hours": total_hours / 3600.0,
        "elapsed_seconds": time.perf_counter() - started,
        "missing": 0,
        "unreadable": 0,
        "test_audio_accessed": False,
    }


def summarize(values: list[float], percentiles: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {"count": len(values), "mean": float(np.mean(array)), "median": float(np.median(array))}
    result.update({f"p{p}": float(np.percentile(array, p)) for p in percentiles})
    return result


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(item) for item in matrix.ravel())
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(truth), "deletion_support": int(np.sum(truth)), "non_deletion_support": int(np.sum(1 - truth)),
        "accuracy": float((tn + tp) / len(truth)), "balanced_accuracy": float((non_recall + recall) / 2.0),
        "macro_f1": float((non_f1 + f1) / 2.0), "deletion_precision": float(precision),
        "deletion_recall": float(recall), "deletion_f1": float(f1), "confusion_matrix": matrix.tolist(),
    }


def binary_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    rows = tn + fp + fn + tp
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": rows, "deletion_support": tp + fn, "non_deletion_support": tn + fp,
        "accuracy": (tn + tp) / rows, "balanced_accuracy": (non_recall + recall) / 2.0,
        "macro_f1": (non_f1 + f1) / 2.0, "deletion_precision": precision,
        "deletion_recall": recall, "deletion_f1": f1, "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def fit_threshold(values: np.ndarray, truth: np.ndarray, direction: str) -> dict[str, Any]:
    order = np.argsort(values, kind="stable") if direction == "low" else np.argsort(-values, kind="stable")
    sorted_values = values[order]
    sorted_truth = truth[order]
    cumulative_positive = np.cumsum(sorted_truth)
    cumulative_negative = np.cumsum(1 - sorted_truth)
    group_ends = np.flatnonzero(np.r_[sorted_values[1:] != sorted_values[:-1], True])
    total_positive = int(np.sum(truth))
    total_negative = len(truth) - total_positive
    if direction == "low":
        sentinel = float(np.nextafter(sorted_values[0], -np.inf))
        conservative: Callable[[float], float] = lambda threshold: -threshold
    elif direction == "high":
        sentinel = float(np.nextafter(sorted_values[0], np.inf))
        conservative = lambda threshold: threshold
    else:
        raise ValueError(direction)
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    candidate_counts = [(sentinel, 0, 0)] + [
        (float(sorted_values[end]), int(cumulative_positive[end]), int(cumulative_negative[end]))
        for end in group_ends
    ]
    for threshold, tp, fp in candidate_counts:
        metrics = binary_metrics_from_counts(total_negative - fp, fp, total_positive - tp, tp)
        key = (
            metrics["macro_f1"], metrics["deletion_f1"], metrics["deletion_precision"],
            -float(tp + fp), conservative(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": float(threshold), "train_metrics": metrics}
    assert best is not None
    return best


def apply_threshold(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    return (values <= threshold).astype(np.int64) if direction == "low" else (values >= threshold).astype(np.int64)


def baseline(
    name: str, train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float], direction: str,
) -> dict[str, Any]:
    train_values = np.asarray([getter(row) for row in train_rows], dtype=np.float64)
    train_truth = np.asarray([row["_truth"] for row in train_rows], dtype=np.int64)
    validation_values = np.asarray([getter(row) for row in validation_rows], dtype=np.float64)
    validation_truth = np.asarray([row["_truth"] for row in validation_rows], dtype=np.int64)
    fitted = fit_threshold(train_values, train_truth, direction)
    validation_prediction = apply_threshold(validation_values, fitted["threshold"], direction)
    auc_score = -validation_values if direction == "low" else validation_values
    return {
        "name": name,
        "feature_direction_for_deletion": direction,
        "threshold_selected_on": "TRAIN only; highest Macro-F1, then deletion F1, precision, fewer deletion calls",
        "threshold": fitted["threshold"],
        "train": fitted["train_metrics"],
        "validation": binary_metrics(validation_truth, validation_prediction),
        "validation_roc_auc": float(roc_auc_score(validation_truth, auc_score)),
        "validation_pr_auc": float(average_precision_score(validation_truth, auc_score)),
    }


def duration_bin(row: dict[str, Any]) -> int:
    return int(math.floor(row["_duration"] / MATCH_BIN_SECONDS + 1e-9))


def matched_subset(rows: list[dict[str, Any]], include_speaker: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        key_parts: list[Any] = [row["expected_phone_canonical"], duration_bin(row)]
        if include_speaker:
            key_parts.insert(0, row["speaker_id"])
        groups[tuple(key_parts)][row["_truth"]].append(row)
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        buckets = groups[key]
        count = min(len(buckets[0]), len(buckets[1]))
        if not count:
            continue
        for label in (0, 1):
            indexes = list(range(len(buckets[label])))
            rng.shuffle(indexes)
            selected.extend(buckets[label][index] for index in indexes[:count])
    return sorted(selected, key=lambda row: row["_source_index"])


def evaluate_on_subset(rows: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    truth = np.asarray([row["_truth"] for row in rows], dtype=np.int64)
    getters = {
        "duration_only": (lambda row: row["_duration"], "low"),
        "central_rms_only": (lambda row: row["_central"]["rms_dbfs"], "low"),
        "central_silence_ratio_only": (lambda row: row["_central"]["silence_ratio"], "high"),
        "context_0.50_rms_only": (lambda row: row["_context_0.50"]["rms_dbfs"], "low"),
    }
    metrics: dict[str, Any] = {
        "always_non_deletion": binary_metrics(truth, np.zeros(len(rows), dtype=np.int64)),
    }
    for name, (getter, direction) in getters.items():
        values = np.asarray([getter(row) for row in rows], dtype=np.float64)
        metrics[name] = binary_metrics(truth, apply_threshold(values, baselines[name]["threshold"], direction))
    return metrics


def timing_audit(ledger: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    allowed = set(TRAIN_SPEAKERS + VALIDATION_SPEAKERS)
    for row in ledger:
        if row["speaker_id"] in allowed:
            groups[(row["speaker_id"], row["textgrid_path"])].append(row)
    deletion_rows = 0
    positive_duration = 0
    contiguous_previous = 0
    contiguous_next = 0
    previous_types = Counter()
    next_types = Counter()
    for items in groups.values():
        items.sort(key=lambda row: int(row["interval_index"]))
        for index, row in enumerate(items):
            if row["research_subset"] != "DELETION_ELIGIBLE":
                continue
            deletion_rows += 1
            positive_duration += float(row["duration"]) > 0
            if index:
                previous = items[index - 1]
                contiguous_previous += abs(float(previous["end_time"]) - float(row["start_time"])) <= 1e-6
                previous_types[previous["relation"]] += 1
            if index + 1 < len(items):
                following = items[index + 1]
                contiguous_next += abs(float(row["end_time"]) - float(following["start_time"])) <= 1e-6
                next_types[following["relation"]] += 1
    return {
        "scope": "TRAIN+VALIDATION manual annotation TextGrid phones IntervalTier",
        "deletion_rows": deletion_rows,
        "positive_duration_intervals": positive_duration,
        "contiguous_previous_boundary": contiguous_previous,
        "contiguous_next_boundary": contiguous_next,
        "previous_neighbor_relations": dict(previous_types),
        "next_neighbor_relations": dict(next_types),
        "source_evidence": {
            "builder_input": "<speaker>/annotation/*.TextGrid only",
            "tier": "phones IntervalTier only",
            "deletion_label_contract": "valid expected,sil,d -> expected canonical phone and observed <SIL>",
            "MFA_or_textgrid_directory_used": False,
            "interpretation": "A deletion is represented by a non-zero manual phones-tier interval occupying a contiguous temporal slot; it is not a PointTier marker. The code does not prove how annotators chose boundaries relative to an MFA alignment.",
        },
        "runtime_warning": "Manual deletion boundaries and labels are supervision artifacts and are not inherently available from runtime audio; compatibility requires a separate alignment/sequence mechanism.",
    }


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite R4-0 audit: {EXPERIMENT_DIR}")
    random.seed(SEED)
    np.random.seed(SEED)
    audio_root = require_audio_root()
    ledger, rows, source = load_rows(audio_root)
    train_rows = [row for row in rows if row["_split"] == "train"]
    validation_rows = [row for row in rows if row["_split"] == "validation"]

    per_phone: dict[str, Any] = {}
    metadata_clean = [
        row for row in ledger
        if row["speaker_id"] in set(TRAIN_SPEAKERS + VALIDATION_SPEAKERS)
        and row["label_quality"] == "clean"
        and row["relation"] in {"correct", "substitution", "deletion"}
    ]
    for phone in sorted({row["expected_phone_canonical"] for row in metadata_clean}):
        block: dict[str, int] = {}
        for split, speakers in (("train", set(TRAIN_SPEAKERS)), ("validation", set(VALIDATION_SPEAKERS))):
            for relation in ("correct", "substitution", "deletion"):
                block[f"{split}_{relation}"] = sum(
                    row["speaker_id"] in speakers
                    and row["expected_phone_canonical"] == phone
                    and row["relation"] == relation
                    for row in metadata_clean
                )
        block["flags"] = [
            label for condition, label in (
                (block["train_deletion"] == 0, "NO_TRAIN_DELETION"),
                (block["validation_deletion"] == 0, "NO_VALIDATION_DELETION"),
                (block["train_deletion"] < 10, "TRAIN_DELETION_LT_10"),
                (block["validation_deletion"] < 5, "VALIDATION_DELETION_LT_5"),
            ) if condition
        ]
        per_phone[phone] = block

    timing = timing_audit(ledger)
    audio_preflight = extract_audio(rows)

    relation_rows = {
        relation: [row for row in rows if row["relation"] == relation]
        for relation in ("correct", "substitution", "deletion")
    }
    duration = {relation: summarize([row["_duration"] for row in items]) for relation, items in relation_rows.items()}
    acoustic: dict[str, Any] = {}
    for relation, items in relation_rows.items():
        acoustic[relation] = {
            "central_interval": {
                feature: summarize([row["_central"][feature] for row in items])
                for feature in ("rms_dbfs", "silence_ratio", "zero_crossing_rate")
            },
            "local_snr_db": summarize([row["_local_snr_db"] for row in items if row["_local_snr_db"] is not None]),
            "fixed_context": {
                f"{seconds:.2f}": {
                    feature: summarize([row[f"_context_{seconds:.2f}"][feature] for row in items])
                    for feature in ("rms_dbfs", "silence_ratio", "zero_crossing_rate")
                }
                for seconds in WINDOWS
            },
        }

    window_report: dict[str, Any] = {}
    for seconds in WINDOWS:
        window_report[f"{seconds:.2f}"] = {}
        for relation, items in relation_rows.items():
            durations = np.asarray([row["_duration"] for row in items])
            window_report[f"{seconds:.2f}"][relation] = {
                "rows": len(items),
                "edge_padded": int(sum(row[f"_padded_{seconds:.2f}"] for row in items)),
                "edge_padding_rate": float(np.mean([row[f"_padded_{seconds:.2f}"] for row in items])),
                "full_interval_coverage_rate": float(np.mean(durations <= seconds)),
                "mean_non_target_context_seconds": float(np.mean(np.maximum(0.0, seconds - durations))),
                "median_non_target_context_seconds": float(np.median(np.maximum(0.0, seconds - durations))),
            }

    baseline_specs = {
        "duration_only": (lambda row: row["_duration"], "low"),
        "central_rms_only": (lambda row: row["_central"]["rms_dbfs"], "low"),
        "central_silence_ratio_only": (lambda row: row["_central"]["silence_ratio"], "high"),
        "context_0.50_rms_only": (lambda row: row["_context_0.50"]["rms_dbfs"], "low"),
    }
    baselines = {
        name: baseline(name, train_rows, validation_rows, getter, direction)
        for name, (getter, direction) in baseline_specs.items()
    }
    validation_truth = np.asarray([row["_truth"] for row in validation_rows], dtype=np.int64)
    baselines["always_non_deletion"] = {
        "validation": binary_metrics(validation_truth, np.zeros(len(validation_rows), dtype=np.int64))
    }
    train_phone_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in train_rows:
        train_phone_counts[row["expected_phone_canonical"]][row["_truth"]] += 1
    phone_probabilities = {
        phone: counts[1] / (counts[0] + counts[1]) for phone, counts in train_phone_counts.items()
    }
    phone_prediction = np.asarray([
        phone_probabilities.get(row["expected_phone_canonical"], 0.0) >= 0.5 for row in validation_rows
    ], dtype=np.int64)
    baselines["expected_phone_prior"] = {
        "rule": "predict deletion when TRAIN P(deletion|expected_phone) >= 0.5",
        "train_probabilities": dict(sorted(phone_probabilities.items())),
        "validation": binary_metrics(validation_truth, phone_prediction),
    }

    strict = matched_subset(validation_rows, include_speaker=True)
    relaxed = matched_subset(validation_rows, include_speaker=False)
    matched_report = {
        "bin_seconds": MATCH_BIN_SECONDS,
        "seed": SEED,
        "without_replacement": True,
        "strict_same_speaker_phone_duration_bin": {
            "rows": len(strict), "pairs": len(strict) // 2,
            "deletion": sum(row["_truth"] for row in strict),
            "non_deletion": sum(1 - row["_truth"] for row in strict),
            "phones": len({row["expected_phone_canonical"] for row in strict}),
            "speakers": len({row["speaker_id"] for row in strict}),
            "baseline_metrics_with_train_frozen_thresholds": evaluate_on_subset(strict, baselines),
        },
        "relaxed_same_phone_duration_bin": {
            "rows": len(relaxed), "pairs": len(relaxed) // 2,
            "deletion": sum(row["_truth"] for row in relaxed),
            "non_deletion": sum(1 - row["_truth"] for row in relaxed),
            "phones": len({row["expected_phone_canonical"] for row in relaxed}),
            "speakers": len({row["speaker_id"] for row in relaxed}),
            "baseline_metrics_with_train_frozen_thresholds": evaluate_on_subset(relaxed, baselines),
        },
    }

    speaker_report: dict[str, Any] = {}
    for speaker in TRAIN_SPEAKERS + VALIDATION_SPEAKERS:
        items = [row for row in rows if row["speaker_id"] == speaker]
        deletions = [row for row in items if row["relation"] == "deletion"]
        all_clean_count = sum(row["speaker_id"] == speaker for row in metadata_clean)
        speaker_report[speaker] = {
            "split": split_for(speaker), "clean_rows_same_deletion_phone_inventory": len(items),
            "clean_rows_all_expected_phones": all_clean_count,
            "deletion_count": len(deletions), "deletion_rate": len(deletions) / all_clean_count,
            "median_deletion_duration": float(np.median([row["_duration"] for row in deletions])) if deletions else None,
            "median_deletion_rms_dbfs": float(np.median([row["_central"]["rms_dbfs"] for row in deletions])) if deletions else None,
            "median_deletion_silence_ratio": float(np.median([row["_central"]["silence_ratio"] for row in deletions])) if deletions else None,
        }

    EXPERIMENT_DIR.mkdir(parents=True)
    write_json(EXPERIMENT_DIR / "metadata_audit.json", {
        "source": source, "per_phone": per_phone, "speaker_variability": speaker_report,
        "test_policy": "TEST metadata used only for aggregate deletion count; TEST audio paths not resolved/accessed",
    })
    write_json(EXPERIMENT_DIR / "audio_audit.json", {
        "preflight": audio_preflight, "duration_seconds": duration, "energy_silence_zcr": acoustic,
        "candidate_windows": window_report, "silence_definition": f"20 ms frames with RMS <= {SILENCE_DBFS} dBFS",
    })
    write_json(EXPERIMENT_DIR / "trivial_baselines.json", baselines)
    write_json(EXPERIMENT_DIR / "matched_control_report.json", matched_report)
    write_json(EXPERIMENT_DIR / "textgrid_timing_audit.json", timing)
    with (EXPERIMENT_DIR / "matched_control_validation_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time", "duration", "duration_bin_10ms", "expected_phone_canonical", "relation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in strict:
            writer.writerow({
                "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"], "utterance_id": row["utterance_id"],
                "start_time": row["start_time"], "end_time": row["end_time"], "duration": row["duration"],
                "duration_bin_10ms": duration_bin(row), "expected_phone_canonical": row["expected_phone_canonical"],
                "relation": row["relation"],
            })
    write_json(EXPERIMENT_DIR / "run_identity.json", {
        "experiment": "R4-0 deletion feasibility audit", "markers": ["RESEARCH_ONLY", "NO_TRAINING", "R4_TEST_CLOSED"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "v4_sha256": source["v4_sha256"],
        "audio_root_identity": audio_root.name, "sample_rate": SAMPLE_RATE, "seed": SEED,
        "train_speakers": list(TRAIN_SPEAKERS), "validation_speakers": list(VALIDATION_SPEAKERS),
        "test_speakers_metadata_only": list(TEST_SPEAKERS), "test_audio_accessed": False,
    })

    print(json.dumps({
        "status": "R4_0_AUDIT_COMPLETE", "deletion_counts": source["deletion_counts"],
        "duration": duration, "baselines": {name: item["validation"] for name, item in baselines.items()},
        "matched": matched_report, "audio_preflight": audio_preflight,
        "test_audio_accessed": False, "training": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
