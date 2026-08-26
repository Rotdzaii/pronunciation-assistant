from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3a  # noqa: E402


REPO_ROOT = r3a.REPO_ROOT
R3D_DIR = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs"
CHECKPOINT_PATH = R3D_DIR / "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
MANIFEST_PATH = REPO_ROOT / "ai-training/experiments/r3_locked_test_protocol/r3_binary_margin_frozen_manifest.json"
PROTOCOL_PATH = REPO_ROOT / "ai-training/experiments/r3_locked_test_protocol/r3_binary_margin_frozen_protocol.md"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_locked_test_evaluation"

EXPECTED_CHECKPOINT_SHA256 = "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E"
EXPECTED_MANIFEST_SHA256 = "50AC3DC012191E2D401A5E7437B46F11F06356A5CC5EA931B916186B1C2B4576"
EXPECTED_PROTOCOL_SHA256 = "B2750BD8438A4759DE0F40C1A9B244DE78AF8494F5CDEB139F023C0C53F470DF"
EXPECTED_V4_SHA256 = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"
EXPECTED_TEST_ROWS = 28_216
FROZEN_THRESHOLD = -1.293920
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
IMPORTANT_PHONES = ("TH", "DH", "R", "V", "D", "T", "S", "Z")
IMPORTANT_PAIRS = (("TH", "T"), ("DH", "D"), ("R", "L"), ("V", "W"), ("Z", "S"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    correct_precision = tn / (tn + fn) if tn + fn else 0.0
    correct_recall = tn / (tn + fp) if tn + fp else 0.0
    correct_f1 = (
        2 * correct_precision * correct_recall / (correct_precision + correct_recall)
        if correct_precision + correct_recall
        else 0.0
    )
    substitution_precision = tp / (tp + fp) if tp + fp else 0.0
    substitution_recall = tp / (tp + fn) if tp + fn else 0.0
    substitution_f1 = (
        2 * substitution_precision * substitution_recall / (substitution_precision + substitution_recall)
        if substitution_precision + substitution_recall
        else 0.0
    )
    return {
        "rows": int(len(truth)),
        "correct_support": int(np.sum(truth == 0)),
        "substitution_support": int(np.sum(truth == 1)),
        "accuracy": float((tn + tp) / len(truth)),
        "balanced_accuracy": float((correct_recall + substitution_recall) / 2.0),
        "macro_f1": float((correct_f1 + substitution_f1) / 2.0),
        "correct_precision": float(correct_precision),
        "correct_recall": float(correct_recall),
        "correct_f1": float(correct_f1),
        "substitution_precision": float(substitution_precision),
        "substitution_recall": float(substitution_recall),
        "substitution_f1": float(substitution_f1),
        "confusion_matrix_labels": ["correct", "substitution"],
        "confusion_matrix": matrix.tolist(),
    }


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = {
        "checkpoint": sha256_file(CHECKPOINT_PATH),
        "manifest": sha256_file(MANIFEST_PATH),
        "protocol": sha256_file(PROTOCOL_PATH),
        "v4_dataset": sha256_file(r3a.METADATA_CSV),
    }
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "v4_dataset": EXPECTED_V4_SHA256,
    }
    failures = [name for name in expected if actual[name] != expected[name]]
    if failures:
        raise RuntimeError(f"R3_3_FREEZE_VERIFICATION_FAIL: SHA mismatch for {failures}; actual={actual}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    invariant_failures: list[str] = []
    checks = {
        "manifest status": manifest.get("status") == "FROZEN_BEFORE_TEST",
        "checkpoint SHA in manifest": manifest["source_identity"]["acoustic_checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "dataset SHA in manifest": manifest["source_identity"]["dataset_sha256"] == EXPECTED_V4_SHA256,
        "vocabulary": tuple(manifest["target_vocabulary"]["index_to_phone"]) == r3a.PHONE_VOCAB,
        "threshold": float(manifest["binary_decision"]["frozen_threshold"]) == FROZEN_THRESHOLD,
        "operator": manifest["binary_decision"]["comparison_operator"] == "<=",
        "test speakers": tuple(manifest["locked_test_evaluation"]["speakers"]) == TEST_SPEAKERS,
        "expected phone not neural input": manifest["acoustic_model"]["expected_phone_is_neural_input"] is False,
        "no 0-100 mapping": manifest["score_policy"]["mapping_to_0_100_exists"] is False,
    }
    invariant_failures.extend(name for name, passed in checks.items() if not passed)
    if invariant_failures:
        raise RuntimeError(f"R3_3_FREEZE_VERIFICATION_FAIL: manifest invariant mismatch: {invariant_failures}")
    return {
        "status": "PASS",
        "verified_before_test_audio_access": True,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "manifest_invariants": checks,
        "frozen_threshold": FROZEN_THRESHOLD,
        "comparison_operator": "<=",
    }, manifest


def test_audio_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["_audio_path"]].append(row)
    errors: list[str] = []
    sample_rates: Counter[int] = Counter()
    padding = Counter()
    origin_totals = Counter(row["relation"] for row in rows)
    total_seconds = 0.0
    half_window = r3a.WINDOW_SECONDS / 2.0
    for path, path_rows in tqdm(sorted(by_path.items(), key=lambda item: str(item[0])), desc="Locked TEST audio preflight"):
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or not math.isfinite(info.duration) or info.duration <= 0:
                errors.append(f"invalid duration: {path}")
                continue
            with sf.SoundFile(path, mode="r") as handle:
                probe = handle.read(frames=1, dtype="float32", always_2d=True)
            if probe.size == 0 or not np.isfinite(probe).all():
                errors.append(f"unreadable first frame: {path}")
                continue
            if max(row["_end"] for row in path_rows) > info.duration + 0.050:
                errors.append(f"annotation exceeds audio: {path}")
            for row in path_rows:
                center = (row["_start"] + row["_end"]) / 2.0
                overlap = min(info.duration, center + half_window) - max(0.0, center - half_window)
                if overlap <= 0:
                    errors.append(f"empty crop: {path} [{row['_start']}, {row['_end']}]")
                    continue
                row["_edge_padded"] = center - half_window < 0 or center + half_window > info.duration
                padding[(row["relation"], row["_edge_padded"])] += 1
            sample_rates[info.samplerate] += 1
            total_seconds += info.duration
        except Exception as exc:
            errors.append(f"unreadable: {path}: {exc}")
    if errors:
        raise RuntimeError(f"Locked TEST audio preflight failed with {len(errors)} error(s):\n" + "\n".join(errors[:20]))
    return {
        "scope": "locked TEST speakers only",
        "rows": len(rows),
        "unique_wav_files": len(by_path),
        "sample_rates": {str(key): value for key, value in sorted(sample_rates.items())},
        "total_unique_audio_hours": total_seconds / 3600.0,
        "missing": 0,
        "unreadable": 0,
        "invalid_duration": 0,
        "empty_crop": 0,
        "edge_padding_by_origin": {
            origin: {
                "rows": origin_totals[origin],
                "padded": padding[(origin, True)],
                "padded_rate": padding[(origin, True)] / origin_totals[origin],
            }
            for origin in ("correct", "substitution")
        },
    }


def acoustic_metrics(labels: np.ndarray, predictions: np.ndarray, top3: np.ndarray) -> dict[str, Any]:
    block = r3a.metric_block(labels.tolist(), predictions.tolist(), r3a.ALL_LABELS)
    block["top1_accuracy"] = block["accuracy"]
    block["top3_accuracy"] = float(np.mean([truth in guesses for truth, guesses in zip(labels, top3)]))
    block["confusion_matrix_labels"] = list(r3a.PHONE_VOCAB)
    block["confusion_matrix"] = confusion_matrix(labels, predictions, labels=r3a.ALL_LABELS).astype(int).tolist()
    return block


def phone_diagnostics(
    rows: list[dict[str, Any]], truth: np.ndarray, binary_predictions: np.ndarray, margins: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phone in IMPORTANT_PHONES:
        positions = np.asarray([i for i, row in enumerate(rows) if row["expected_phone_canonical"] == phone], dtype=int)
        correct = positions[truth[positions] == 0]
        substitution = positions[truth[positions] == 1]
        predicted_substitution = positions[binary_predictions[positions] == 1]
        true_positive = int(np.sum(truth[predicted_substitution] == 1))
        result[phone] = {
            "correct_support": int(len(correct)),
            "substitution_support": int(len(substitution)),
            "correct_recall": float(np.mean(binary_predictions[correct] == 0)) if len(correct) else None,
            "substitution_recall": float(np.mean(binary_predictions[substitution] == 1)) if len(substitution) else None,
            "substitution_precision": float(true_positive / len(predicted_substitution)) if len(predicted_substitution) else None,
            "median_margin_correct": float(np.median(margins[correct])) if len(correct) else None,
            "median_margin_substitution": float(np.median(margins[substitution])) if len(substitution) else None,
        }
    return result


def pair_diagnostics(
    rows: list[dict[str, Any]], binary_predictions: np.ndarray, margins: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for expected, observed in IMPORTANT_PAIRS:
        positions = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if row["relation"] == "substitution"
                and row["expected_phone_canonical"] == expected
                and row["observed_phone_canonical"] == observed
            ],
            dtype=int,
        )
        detected = int(np.sum(binary_predictions[positions] == 1))
        result[f"{expected}->{observed}"] = {
            "support": int(len(positions)),
            "detected_substitutions": detected,
            "detection_recall": float(detected / len(positions)) if len(positions) else None,
            "false_negatives": int(len(positions) - detected),
            "median_margin": float(np.median(margins[positions])) if len(positions) else None,
        }
    return result


def interpretation(binary: dict[str, Any], catastrophic: bool) -> str:
    if (
        binary["macro_f1"] >= 0.55
        and binary["substitution_recall"] >= 0.45
        and binary["substitution_f1"] >= 0.28
        and not catastrophic
    ):
        return "TEST_TRANSFER_CONFIRMED"
    if (
        binary["macro_f1"] >= 0.50
        and binary["substitution_recall"] >= 0.35
        and binary["substitution_f1"] >= 0.22
        and not catastrophic
    ):
        return "TEST_TRANSFER_PARTIAL"
    return "TEST_TRANSFER_NOT_CONFIRMED"


def self_check() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predicted = np.asarray([0, 1, 0, 1], dtype=np.int64)
    metrics = binary_metrics(truth, predicted)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == metrics["balanced_accuracy"] == metrics["macro_f1"] == 0.5
    margins = np.asarray([-1.2939201, -1.293920, -1.2939199])
    assert (margins <= FROZEN_THRESHOLD).tolist() == [True, True, False]


def write_report(
    acoustic: dict[str, Any], binary: dict[str, Any], speakers: dict[str, Any], phones: dict[str, Any],
    pairs: dict[str, Any], margins: dict[str, Any], final_interpretation: str, catastrophic_status: str,
    inference_seconds: float,
) -> None:
    lines = [
        "# R3-3 One-Time Locked TEST Evaluation",
        "",
        "RESEARCH_ONLY — NOT_PRODUCTION — NO_0_TO_100_SCORE",
        "",
        "The frozen artifacts were verified before TEST audio access. No training, threshold fitting,",
        "checkpoint selection, preprocessing change, or TEST-specific adjustment occurred.",
        "",
        "## Locked identity",
        "",
        f"- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`",
        f"- V4 SHA-256: `{EXPECTED_V4_SHA256}`",
        f"- Threshold: `{FROZEN_THRESHOLD:.6f}` with `margin <= threshold` mapping to substitution",
        "- Margin: `expected_logit - max(other 39 logits)`",
        f"- TEST rows: `{binary['rows']}`",
        f"- Inference seconds: `{inference_seconds:.3f}`",
        "",
        "## Acoustic observed-phone TEST metrics",
        "",
        f"- Top-1: `{acoustic['top1_accuracy']:.6f}`",
        f"- Top-3: `{acoustic['top3_accuracy']:.6f}`",
        f"- Macro-F1: `{acoustic['macro_f1']:.6f}`",
        f"- Balanced accuracy: `{acoustic['balanced_accuracy']:.6f}`",
        f"- Macro precision: `{acoustic['macro_precision']:.6f}`",
        "",
        "## Primary frozen binary TEST metrics",
        "",
        f"- Accuracy: `{binary['accuracy']:.6f}`",
        f"- Balanced accuracy: `{binary['balanced_accuracy']:.6f}`",
        f"- Binary Macro-F1: `{binary['macro_f1']:.6f}`",
        f"- Substitution precision: `{binary['substitution_precision']:.6f}`",
        f"- Substitution recall: `{binary['substitution_recall']:.6f}`",
        f"- Substitution F1: `{binary['substitution_f1']:.6f}`",
        f"- Confusion matrix: `{binary['confusion_matrix']}`",
        "",
        f"Speaker assessment: **{catastrophic_status}**",
        "",
        f"TEST interpretation: **{final_interpretation}**",
        "",
        "## Per-speaker binary metrics",
        "",
        "| Speaker | Rows | Correct | Substitution | Macro-F1 | Sub P | Sub R | Sub F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for speaker in TEST_SPEAKERS:
        item = speakers[speaker]["binary"]
        lines.append(
            f"| {speaker} | {item['rows']} | {item['correct_support']} | {item['substitution_support']} | "
            f"{item['macro_f1']:.6f} | {item['substitution_precision']:.6f} | "
            f"{item['substitution_recall']:.6f} | {item['substitution_f1']:.6f} |"
        )
    lines.extend(["", "## Frozen diagnostics", "", "### Expected phones", ""])
    for phone in IMPORTANT_PHONES:
        lines.append(f"- `{phone}`: `{json.dumps(phones[phone], ensure_ascii=False)}`")
    lines.extend(["", "### Predeclared substitution pairs", ""])
    for pair in pairs:
        lines.append(f"- `{pair}`: `{json.dumps(pairs[pair], ensure_ascii=False)}`")
    lines.extend([
        "",
        "### Margin diagnostics",
        "",
        f"- ROC-AUC (substitution positive): `{margins['roc_auc_substitution']:.6f}`",
        f"- PR-AUC (substitution positive): `{margins['pr_auc_substitution']:.6f}`",
        f"- Correct-origin: `{json.dumps(margins['correct_origin'])}`",
        f"- Substitution-origin: `{json.dumps(margins['substitution_origin'])}`",
        "",
        "No mapping from margin to a 0–100 score was created.",
    ])
    (EXPERIMENT_DIR / "r3_locked_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    self_check()
    # Freeze verification is deliberately complete before resolving or reading any TEST WAV.
    freeze, manifest = verify_freeze()
    if EXPERIMENT_DIR.exists():
        # A terminal-orchestration timeout may interrupt the run immediately after
        # freeze verification. Resumption is allowed only from that exact state:
        # no preflight, logits, predictions, or metrics may already exist.
        existing = {path.name for path in EXPERIMENT_DIR.iterdir()}
        if existing != {"freeze_verification.json"}:
            raise RuntimeError(f"Refusing to overwrite or repeat locked TEST artifacts: {sorted(existing)}")
        previous_freeze = json.loads((EXPERIMENT_DIR / "freeze_verification.json").read_text(encoding="utf-8"))
        if previous_freeze != freeze:
            raise RuntimeError("Partial-run freeze verification differs; refusing locked TEST resumption")
    else:
        EXPERIMENT_DIR.mkdir(parents=True)
        r3a.write_json(EXPERIMENT_DIR / "freeze_verification.json", freeze)

    r3a.set_seed(r3a.SEED)
    audio_root = r3a.require_audio_root()
    split_rows, dataset_summary = r3a.load_and_validate_rows(audio_root)
    test_rows = split_rows["test"]
    if len(test_rows) != EXPECTED_TEST_ROWS or tuple(dataset_summary["split_counts"]["test"]["speakers"]) != TEST_SPEAKERS:
        raise RuntimeError("Locked TEST split count or speakers changed")
    if set(TEST_SPEAKERS) & set(r3a.TRAIN_SPEAKERS + r3a.VALIDATION_SPEAKERS):
        raise RuntimeError("TEST speaker overlap detected")
    for row in test_rows:
        row["_audio_path"] = r3a.resolve_audio_path(row["audio_path"], audio_root)

    audio_preflight = test_audio_preflight(test_rows)
    preflight = {
        "status": "PASS",
        "freeze_verified_before_test_access": True,
        "dataset_sha256": dataset_summary["dataset_sha256"],
        "eligible_subset": "PHONE_IDENTIFICATION_ELIGIBLE",
        "test_speakers": list(TEST_SPEAKERS),
        "test_rows": len(test_rows),
        "correct_rows": int(sum(row["relation"] == "correct" for row in test_rows)),
        "substitution_rows": int(sum(row["relation"] == "substitution" for row in test_rows)),
        "speaker_disjoint": True,
        "audio": audio_preflight,
        "training_performed": False,
        "threshold_search_performed": False,
    }
    r3a.write_json(EXPERIMENT_DIR / "preflight.json", preflight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_started = time.perf_counter()
    test_dataset = r3a.materialize_audio_features(test_rows, device, "one-time locked TEST")
    feature_seconds = time.perf_counter() - feature_started

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 47:
        raise RuntimeError(f"Frozen checkpoint epoch changed: expected 47, got {checkpoint['epoch']}")
    model = r3a.SmallPronunciationCNNAttention().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits_batches: list[torch.Tensor] = []
    labels: list[int] = []
    loader = DataLoader(
        test_dataset,
        batch_size=r3a.BATCH_SIZE,
        shuffle=False,
        num_workers=r3a.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for features, targets, _ in tqdm(loader, desc="ONE-TIME locked TEST inference"):
            logits_batches.append(model(features.to(device, non_blocking=True)).cpu())
            labels.extend(targets.tolist())
    inference_seconds = time.perf_counter() - inference_started

    logits_tensor = torch.cat(logits_batches, dim=0)
    probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    logits = logits_tensor.numpy().astype(np.float64)
    probabilities = probabilities_tensor.numpy().astype(np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    top3 = np.argsort(-probabilities, axis=1)[:, :3]
    if len(labels_array) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Incomplete TEST inference: {len(labels_array)} rows")

    acoustic = acoustic_metrics(labels_array, predictions, top3)
    truth = np.asarray([0 if row["relation"] == "correct" else 1 for row in test_rows], dtype=np.int64)
    expected_ids = np.asarray([r3a.PHONE_TO_ID[row["expected_phone_canonical"]] for row in test_rows], dtype=np.int64)
    row_indexes = np.arange(len(test_rows))
    expected_logits = logits[row_indexes, expected_ids]
    alternative_logits = logits.copy()
    alternative_logits[row_indexes, expected_ids] = -np.inf
    best_alternative_ids = np.argmax(alternative_logits, axis=1)
    best_alternative_logits = alternative_logits[row_indexes, best_alternative_ids]
    margins = expected_logits - best_alternative_logits
    binary_predictions = (margins <= FROZEN_THRESHOLD).astype(np.int64)
    binary = binary_metrics(truth, binary_predictions)

    per_speaker: dict[str, Any] = {}
    catastrophic_speakers: list[str] = []
    for speaker in TEST_SPEAKERS:
        positions = np.asarray([i for i, row in enumerate(test_rows) if row["speaker_id"] == speaker], dtype=int)
        speaker_binary = binary_metrics(truth[positions], binary_predictions[positions])
        speaker_acoustic = acoustic_metrics(labels_array[positions], predictions[positions], top3[positions])
        per_speaker[speaker] = {
            "binary": speaker_binary,
            "acoustic": {
                "rows": speaker_acoustic["rows"],
                "top1_accuracy": speaker_acoustic["top1_accuracy"],
                "macro_f1": speaker_acoustic["macro_f1"],
            },
        }
        if speaker_binary["macro_f1"] < 0.35 or speaker_binary["substitution_recall"] < 0.20:
            catastrophic_speakers.append(speaker)
    catastrophic = bool(catastrophic_speakers)
    catastrophic_status = "CATASTROPHIC_SPEAKER_COLLAPSE" if catastrophic else "NO_CATASTROPHIC_SPEAKER_COLLAPSE"

    phones = phone_diagnostics(test_rows, truth, binary_predictions, margins)
    pairs = pair_diagnostics(test_rows, binary_predictions, margins)
    dh = pairs["DH->D"]
    validation_dh_recall = 0.21666666666666667
    if dh["detection_recall"] is None:
        dh_behavior = "DH_D_TEST_BEHAVIOR_CONSISTENT_WITH_KNOWN_FAILURE"
    elif dh["detection_recall"] >= validation_dh_recall + 0.05:
        dh_behavior = "DH_D_TEST_BEHAVIOR_BETTER_THAN_VALIDATION"
    elif dh["detection_recall"] <= validation_dh_recall - 0.05:
        dh_behavior = "DH_D_TEST_BEHAVIOR_WORSE_THAN_VALIDATION"
    else:
        dh_behavior = "DH_D_TEST_BEHAVIOR_CONSISTENT_WITH_KNOWN_FAILURE"
    pairs["DH->D"]["known_validation_reference"] = {
        "support": 300,
        "detection_recall": validation_dh_recall,
        "false_negatives": 235,
        "median_margin": 0.051110975444316864,
    }
    pairs["DH->D"]["descriptive_classification"] = dh_behavior

    margin_report = {
        "score": "expected_logit - max(other 39 logits)",
        "lower_indicates": "substitution",
        "roc_auc_substitution": float(roc_auc_score(truth, -margins)),
        "pr_auc_substitution": float(average_precision_score(truth, -margins)),
        "correct_origin": distribution(margins[truth == 0]),
        "substitution_origin": distribution(margins[truth == 1]),
        "diagnostic_only": True,
        "threshold_search_performed": False,
    }
    final_interpretation = interpretation(binary, catastrophic)

    row_fields = [
        "row_index", "source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time",
        "expected_phone_canonical", "true_observed_phone_canonical", "relation_origin",
        "predicted_observed_phone", "top1_probability", "expected_logit", "best_alternative_phone",
        "best_alternative_logit", "expected_margin", "frozen_binary_prediction", "correct_acoustic_prediction",
    ]
    row_path = EXPERIMENT_DIR / "test_row_predictions.csv"
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for index, row in enumerate(test_rows):
            writer.writerow({
                "row_index": index,
                "source_csv_row": row["_source_index"] + 2,
                "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"],
                "utterance_id": row["utterance_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "expected_phone_canonical": row["expected_phone_canonical"],
                "true_observed_phone_canonical": row["observed_phone_canonical"],
                "relation_origin": row["relation"],
                "predicted_observed_phone": r3a.PHONE_VOCAB[predictions[index]],
                "top1_probability": probabilities[index, predictions[index]],
                "expected_logit": expected_logits[index],
                "best_alternative_phone": r3a.PHONE_VOCAB[best_alternative_ids[index]],
                "best_alternative_logit": best_alternative_logits[index],
                "expected_margin": margins[index],
                "frozen_binary_prediction": "substitution" if binary_predictions[index] else "correct",
                "correct_acoustic_prediction": bool(predictions[index] == labels_array[index]),
            })

    r3a.write_json(EXPERIMENT_DIR / "test_acoustic_metrics.json", acoustic)
    r3a.write_json(EXPERIMENT_DIR / "test_binary_metrics.json", {
        **binary,
        "frozen_threshold": FROZEN_THRESHOLD,
        "comparison_operator": "<=",
        "margin_equation": "expected_logit - max(other 39 logits)",
        "threshold_changed": False,
    })
    r3a.write_json(EXPERIMENT_DIR / "test_per_class_metrics.json", {
        "vocabulary": list(r3a.PHONE_VOCAB),
        "per_class": acoustic["per_class"],
        "confusion_matrix_labels": acoustic["confusion_matrix_labels"],
        "confusion_matrix": acoustic["confusion_matrix"],
    })
    r3a.write_json(EXPERIMENT_DIR / "test_per_speaker_metrics.json", {
        "speakers": per_speaker,
        "catastrophic_definition": "binary Macro-F1 < 0.35 OR substitution recall < 0.20",
        "catastrophic_speakers": catastrophic_speakers,
        "assessment": catastrophic_status,
    })
    r3a.write_json(EXPERIMENT_DIR / "test_phone_diagnostics.json", phones)
    r3a.write_json(EXPERIMENT_DIR / "test_pair_diagnostics.json", pairs)
    r3a.write_json(EXPERIMENT_DIR / "test_margin_diagnostics.json", margin_report)
    final_status = {
        "status": "R3_3_LOCKED_TEST_COMPLETE",
        "test_interpretation": final_interpretation,
        "catastrophic_speaker_assessment": catastrophic_status,
        "catastrophic_speakers": catastrophic_speakers,
        "freeze_verification": "PASS",
        "test_rows": len(test_rows),
        "test_accessed_exactly_for_locked_evaluation": True,
        "threshold_changed": False,
        "threshold_search_performed": False,
        "training_performed": False,
        "checkpoint_changed": False,
        "preprocessing_changed": False,
        "zero_to_100_mapping_created": False,
        "device": str(device),
        "feature_materialization_seconds": feature_seconds,
        "inference_seconds": inference_seconds,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "librosa": librosa.__version__,
        "source_manifest_status": manifest["status"],
    }
    r3a.write_json(EXPERIMENT_DIR / "final_test_status.json", final_status)
    write_report(acoustic, binary, per_speaker, phones, pairs, margin_report, final_interpretation, catastrophic_status, inference_seconds)

    hash_targets = [
        "freeze_verification.json", "preflight.json", "test_acoustic_metrics.json", "test_binary_metrics.json",
        "test_per_class_metrics.json", "test_per_speaker_metrics.json", "test_phone_diagnostics.json",
        "test_pair_diagnostics.json", "test_margin_diagnostics.json", "test_row_predictions.csv",
        "final_test_status.json", "r3_locked_test_report.md",
    ]
    artifact_hashes = {name: sha256_file(EXPERIMENT_DIR / name) for name in hash_targets}
    r3a.write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256",
        "generated_after_locked_test_evaluation": True,
        "artifacts": artifact_hashes,
    })

    print(json.dumps({
        "status": "R3_3_LOCKED_TEST_COMPLETE",
        "interpretation": final_interpretation,
        "freeze": "PASS",
        "rows": len(test_rows),
        "device": str(device),
        "acoustic": {key: acoustic[key] for key in ("top1_accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy")},
        "binary": binary,
        "catastrophic": catastrophic_status,
        "catastrophic_speakers": catastrophic_speakers,
        "dh_to_d": pairs["DH->D"],
        "margin": margin_report,
        "training": False,
        "threshold_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
