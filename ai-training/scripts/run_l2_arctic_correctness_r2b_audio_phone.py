from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import run_l2_arctic_correctness_r2a as r2a


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r2b_audio_phone_seed42"
EXPECTED_DATASET_SHA256 = "433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E"

TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
EXPECTED_SPLITS = {
    "train": {"rows": 59_572, "correct": 51_400, "incorrect": 8_172, "substitution": 6_558, "deletion": 1_614},
    "validation": {"rows": 29_767, "correct": 25_303, "incorrect": 4_464, "substitution": 3_485, "deletion": 979},
}

PHONE_VOCAB = (
    "<UNK>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R",
    "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_VOCAB)}
ARPABET_VOWELS = frozenset("AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW".split())
EXPECTED_UNK_TRAIN = {"AX": 1, "d": 1, "l": 1}
FOCUS_PHONES = ("DH", "Z", "M", "F", "OY")
SUPPORTED_PHONE_SET = tuple(phone for phone in PHONE_VOCAB[1:] if phone not in {"CH", "M", "OY"})

SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 12
NUM_WORKERS = 0
DROPOUT = 0.2
EMBEDDING_DIM = 16
CLASS_WEIGHT_REFERENCE = (0.579494163, 3.644884973)

EXTERNAL_BASELINES = {
    "audio_only_r2a": {"macro_f1": 0.542294},
    "phone_only_r2b0": {
        "macro_f1": 0.598279,
        "balanced_accuracy": 0.580861,
        "incorrect_f1": 0.282390,
        "incorrect_recall": 0.201165,
    },
}
PRIMARY_THRESHOLDS = {
    "macro_f1": 0.65,
    "balanced_accuracy": 0.65,
    "incorrect_f1": 0.40,
    "incorrect_recall": 0.40,
    "substitution_recall": 0.40,
    "deletion_recall": 0.30,
}
PHONE_ONLY_IMPROVEMENTS = {"macro_f1": 0.05, "balanced_accuracy": 0.05, "incorrect_f1": 0.10}
MODALITY_THRESHOLDS = {"macro_f1_delta": 0.03, "incorrect_f1_delta": 0.05}
DURATION_DROP_THRESHOLDS = {"macro_f1": 0.10, "incorrect_f1": 0.10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked research-only R2-B audio + canonical-phone baseline.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate TRAIN/VALIDATION dataset, audio, vocabulary, and forward shapes without training or artifacts.",
    )
    return parser.parse_args()


def canonicalize_phone(raw_phone: str) -> str:
    if len(raw_phone) >= 2 and raw_phone[-1] in "012" and raw_phone[:-1] in ARPABET_VOWELS:
        return raw_phone[:-1]
    return raw_phone


def phone_id(raw_phone: str) -> int:
    return PHONE_TO_ID.get(canonicalize_phone(raw_phone), 0)


def load_train_validation_rows(audio_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    actual_hash = r2a.sha256_file(METADATA_CSV)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, got {actual_hash}")

    train_speakers = set(TRAIN_SPEAKERS)
    validation_speakers = set(VALIDATION_SPEAKERS)
    test_speakers = set(TEST_SPEAKERS)
    if train_speakers & validation_speakers or train_speakers & test_speakers or validation_speakers & test_speakers:
        raise RuntimeError("Locked TRAIN/VALIDATION/TEST speaker sets overlap")
    allowed_speakers = train_speakers | validation_speakers
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"speaker_id", "audio_path", "start_time", "end_time", "error_type", "expected_phone"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Metadata is missing required fields: {sorted(missing)}")
        for source_index, source in enumerate(reader):
            speaker = source["speaker_id"]
            if speaker not in allowed_speakers:
                continue
            error_type = source["error_type"]
            if error_type == "addition":
                continue
            if error_type not in {"correct", "substitution", "deletion"}:
                raise RuntimeError(f"Unexpected eligible label at source row {source_index + 2}: {error_type!r}")
            try:
                start = float(source["start_time"])
                end = float(source["end_time"])
            except ValueError as exc:
                raise RuntimeError(f"Invalid interval at source row {source_index + 2}") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise RuntimeError(f"Invalid interval at source row {source_index + 2}: {start}, {end}")
            raw_phone = source["expected_phone"]
            canonical_phone = canonicalize_phone(raw_phone)
            row = dict(source)
            row["_source_index"] = source_index
            row["_start"] = start
            row["_end"] = end
            row["_duration"] = end - start
            row["_binary_label"] = 0 if error_type == "correct" else 1
            row["_canonical_phone"] = canonical_phone
            row["_phone_id"] = PHONE_TO_ID.get(canonical_phone, 0)
            row["_audio_path"] = r2a.resolve_audio_path(source["audio_path"], audio_root)
            split_name = "train" if speaker in train_speakers else "validation"
            split_rows[split_name].append(row)

    summary: dict[str, Any] = {"dataset_sha256": actual_hash, "splits": {}}
    for split_name, expected in EXPECTED_SPLITS.items():
        rows = split_rows[split_name]
        classes = Counter(row["error_type"] for row in rows)
        binary = Counter(row["_binary_label"] for row in rows)
        actual = {
            "rows": len(rows),
            "correct": binary[0],
            "incorrect": binary[1],
            "substitution": classes["substitution"],
            "deletion": classes["deletion"],
        }
        if actual != expected:
            raise RuntimeError(f"{split_name} counts differ: expected {expected}, got {actual}")
        expected_speakers = train_speakers if split_name == "train" else validation_speakers
        actual_speakers = {row["speaker_id"] for row in rows}
        if actual_speakers != expected_speakers:
            raise RuntimeError(f"{split_name} speakers differ: {sorted(actual_speakers)}")
        summary["splits"][split_name] = {**actual, "speakers": sorted(actual_speakers)}

    train_canonical_support = Counter(row["_canonical_phone"] for row in split_rows["train"])
    dedicated = {phone for phone, support in train_canonical_support.items() if support >= 50}
    if dedicated != set(PHONE_VOCAB[1:]):
        raise RuntimeError(
            f"Frozen dedicated vocabulary differs from TRAIN support>=50: expected {list(PHONE_VOCAB[1:])}, got {sorted(dedicated)}"
        )
    train_unk = Counter(row["_canonical_phone"] for row in split_rows["train"] if row["_phone_id"] == 0)
    validation_unk = Counter(row["_canonical_phone"] for row in split_rows["validation"] if row["_phone_id"] == 0)
    if dict(train_unk) != EXPECTED_UNK_TRAIN or validation_unk:
        raise RuntimeError(f"Unexpected <UNK> mapping: train={dict(train_unk)}, validation={dict(validation_unk)}")

    validation_support = defaultdict(Counter)
    for row in split_rows["validation"]:
        validation_support[row["_canonical_phone"]][row["_binary_label"]] += 1
    supported_from_data = {
        phone for phone, support in validation_support.items() if support[0] >= 10 and support[1] >= 10
    }
    if supported_from_data != set(SUPPORTED_PHONE_SET):
        raise RuntimeError(
            f"Predeclared supported-phone set differs: expected {list(SUPPORTED_PHONE_SET)}, got {sorted(supported_from_data)}"
        )

    summary["phone_vocab"] = {
        "size": len(PHONE_VOCAB),
        "tokens": {phone: index for index, phone in enumerate(PHONE_VOCAB)},
        "train_support": {phone: train_canonical_support.get(phone, 0) for phone in PHONE_VOCAB[1:]},
        "train_unk_tokens": dict(train_unk),
        "validation_unk_rows": sum(validation_unk.values()),
        "supported_phone_diagnostic_set": list(SUPPORTED_PHONE_SET),
    }
    return split_rows, summary


class ConditionedFeatureDataset(Dataset):
    def __init__(self, audio_features: r2a.FeatureDataset, rows: list[dict[str, Any]]) -> None:
        if len(audio_features) != len(rows):
            raise RuntimeError("Audio feature and metadata row counts differ")
        self.features = audio_features.features
        self.labels = audio_features.labels
        self.phone_ids = torch.tensor([row["_phone_id"] for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.features[index], self.labels[index], self.phone_ids[index], index


class AudioPhoneCorrectnessModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.attention = r2a.TemporalAttentionPooling(channels=96)
        self.phone_embedding = nn.Embedding(len(PHONE_VOCAB), EMBEDDING_DIM, padding_idx=0)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(96 + EMBEDDING_DIM, 2))

    def assert_unk_is_zero(self) -> None:
        if torch.count_nonzero(self.phone_embedding.weight[0]).item() != 0:
            raise RuntimeError("Embedding row 0 (<UNK>) must remain fixed zero")

    def forward(self, audio_features: torch.Tensor, phone_ids: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode not in {"full", "no_phone", "no_audio"}:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
        feature_map = self.features(audio_features)
        audio_vector, _ = self.attention(feature_map)
        if mode == "no_audio":
            audio_vector = torch.zeros_like(audio_vector)
        if mode == "no_phone":
            phone_ids = torch.zeros_like(phone_ids)
        phone_vector = self.phone_embedding(phone_ids)
        fused = torch.cat((audio_vector, phone_vector), dim=1)
        if fused.shape[1] != 112:
            raise RuntimeError(f"Unexpected fusion dimension: {tuple(fused.shape)}")
        return self.classifier(fused)


def materialize_conditioned(
    rows: list[dict[str, Any]], device: torch.device, split_name: str
) -> ConditionedFeatureDataset:
    audio_features = r2a.materialize_audio_features(rows, device, split_name)
    return ConditionedFeatureDataset(audio_features, rows)


def evaluate(
    model: AudioPhoneCorrectnessModel,
    dataset: ConditionedFeatureDataset,
    rows: list[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    mode: str,
) -> tuple[dict[str, Any], list[int]]:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda"
    )
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for features, targets, phone_ids, indexes in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phone_ids = phone_ids.to(device, non_blocking=True)
            logits = model(features, phone_ids, mode=mode)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
    expected_labels = [row["_binary_label"] for row in rows]
    if labels != expected_labels:
        raise RuntimeError(f"{mode} evaluation order/label invariant failed")
    metrics = r2a.binary_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    metrics["mode"] = mode
    return r2a.add_audit_metrics(metrics, rows, predictions), predictions


def phone_diagnostics(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    def one_phone(phone: str) -> dict[str, Any]:
        positions = [index for index, row in enumerate(rows) if row["_canonical_phone"] == phone]
        correct_positions = [index for index in positions if rows[index]["_binary_label"] == 0]
        incorrect_positions = [index for index in positions if rows[index]["_binary_label"] == 1]
        return {
            "correct_support": len(correct_positions),
            "incorrect_support": len(incorrect_positions),
            "correct_recall": (
                sum(predictions[index] == 0 for index in correct_positions) / len(correct_positions)
                if correct_positions else None
            ),
            "incorrect_recall": (
                sum(predictions[index] == 1 for index in incorrect_positions) / len(incorrect_positions)
                if incorrect_positions else None
            ),
            "predicted_incorrect_rate": sum(predictions[index] == 1 for index in positions) / len(positions) if positions else None,
        }

    per_phone = {}
    per_phone_macro_f1 = []
    for phone in SUPPORTED_PHONE_SET:
        positions = [index for index, row in enumerate(rows) if row["_canonical_phone"] == phone]
        labels = [rows[index]["_binary_label"] for index in positions]
        phone_predictions = [predictions[index] for index in positions]
        metrics = r2a.binary_metrics(labels, phone_predictions)
        per_phone[phone] = {
            "correct_support": metrics["correct"]["support"],
            "incorrect_support": metrics["incorrect"]["support"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
        per_phone_macro_f1.append(metrics["macro_f1"])
    return {
        "focus_phones": {phone: one_phone(phone) for phone in FOCUS_PHONES},
        "supported_phone_definition": "validation correct support >=10 and incorrect support >=10; frozen before training",
        "supported_phone_count": len(SUPPORTED_PHONE_SET),
        "supported_phones": list(SUPPORTED_PHONE_SET),
        "macro_f1_mean_across_supported_phones": float(np.mean(per_phone_macro_f1)),
        "per_phone": per_phone,
    }


def config_payload(summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    return {
        "experiment": "R2-B audio + canonical expected-phone binary correctness",
        "research_only": True,
        "used_by_runtime": False,
        "production_model": False,
        "test_policy": "TEST remains closed; no TEST audio decoding, feature materialization, or inference",
        "dataset": str(METADATA_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": summary["dataset_sha256"],
        "eligible_labels": {"correct": 0, "substitution": 1, "deletion": 1},
        "excluded_labels": ["addition"],
        "speaker_split": {
            "train": list(TRAIN_SPEAKERS),
            "validation": list(VALIDATION_SPEAKERS),
            "test_locked_closed": list(TEST_SPEAKERS),
        },
        "split_counts": summary["splits"],
        "phone_vocabulary": {phone: index for index, phone in enumerate(PHONE_VOCAB)},
        "phone_canonicalization": "remove ARPAbet vowel stress 0/1/2; preserve consonants/case; AX,d,l/unseen -> <UNK>",
        "phone_embedding": {"vocab_size": 40, "dimension": 16, "padding_idx": 0, "unk_fixed_zero": True},
        "model_inputs": ["audio-derived log-mel", "canonical_phone_id"],
        "forbidden_model_features": [
            "raw_label", "error_type", "binary_label", "raw_stressed_expected_phone", "observed_phone", "speaker_id",
            "l1", "gender", "duration", "filename", "path", "utterance_text", "utterance_id", "start_time", "end_time",
        ],
        "audio": {
            "sample_rate": r2a.SAMPLE_RATE,
            "mono": True,
            "crop": "one second centered at (start_time + end_time) / 2 for every class",
            "samples": r2a.WINDOW_SAMPLES,
            "boundary_policy": "clip available utterance audio and zero-pad only the missing side",
        },
        "log_mel": {
            "n_mels": r2a.N_MELS,
            "n_fft": r2a.N_FFT,
            "hop_length": r2a.HOP_LENGTH,
            "win_length": r2a.WIN_LENGTH,
            "window": "hann",
            "center": True,
            "pad_mode": "constant",
            "power": 2.0,
            "power_to_db_ref": "max",
            "top_db": 80.0,
            "feature_shape": list(r2a.EXPECTED_FEATURE_SHAPE),
        },
        "architecture": {
            "audio_backbone": "legacy SmallPronunciationCNNAttention CNN + temporal attention",
            "audio_vector": 96,
            "phone_vector": 16,
            "fusion_vector": 112,
            "head": "Dropout(0.2) + Linear(112,2)",
            "hidden_mlp": False,
            "random_initialization": True,
            "legacy_checkpoint_loaded": False,
        },
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "epochs": EPOCHS,
        "loss": "class-weighted CrossEntropy",
        "class_weights": {"correct": class_weights[0], "incorrect": class_weights[1]},
        "sampler": "none; standard shuffled DataLoader",
        "augmentation": "none",
        "threshold": "argmax",
        "checkpoint_selection": "highest FULL validation Macro-F1; tie higher incorrect F1; tie earlier epoch",
        "external_baselines": EXTERNAL_BASELINES,
        "primary_thresholds": PRIMARY_THRESHOLDS,
        "phone_only_improvement_thresholds": PHONE_ONLY_IMPROVEMENTS,
        "modality_thresholds": MODALITY_THRESHOLDS,
        "duration_drop_thresholds": DURATION_DROP_THRESHOLDS,
    }


def save_epoch_outputs(records: list[dict[str, Any]]) -> None:
    r2a.write_json(EXPERIMENT_DIR / "epoch_metrics.json", records)
    fields = [
        "epoch", "train_loss", "validation_loss", "accuracy", "macro_f1", "balanced_accuracy",
        "incorrect_precision", "incorrect_recall", "incorrect_f1", "substitution_recall", "deletion_recall",
        "tn", "fp", "fn", "tp", "epoch_seconds",
    ]
    with (EXPERIMENT_DIR / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["validation_full"]
            tn, fp = metrics["confusion_matrix"][0]
            fn, tp = metrics["confusion_matrix"][1]
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "train_loss": record["train_loss"],
                    "validation_loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "incorrect_precision": metrics["incorrect"]["precision"],
                    "incorrect_recall": metrics["incorrect"]["recall"],
                    "incorrect_f1": metrics["incorrect"]["f1"],
                    "substitution_recall": metrics["subtype_recall"]["substitution"]["recall"],
                    "deletion_recall": metrics["subtype_recall"]["deletion"]["recall"],
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "epoch_seconds": record["epoch_seconds"],
                }
            )


def primary_gate(metrics: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    values = {
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "incorrect_f1": metrics["incorrect"]["f1"],
        "incorrect_recall": metrics["incorrect"]["recall"],
        "substitution_recall": metrics["subtype_recall"]["substitution"]["recall"],
        "deletion_recall": metrics["subtype_recall"]["deletion"]["recall"],
    }
    failures = [
        f"{name}: {values[name]:.6f} < {threshold:.6f}"
        for name, threshold in PRIMARY_THRESHOLDS.items() if values[name] < threshold
    ]
    baseline = EXTERNAL_BASELINES["phone_only_r2b0"]
    improvements = {
        "macro_f1": values["macro_f1"] - baseline["macro_f1"],
        "balanced_accuracy": values["balanced_accuracy"] - baseline["balanced_accuracy"],
        "incorrect_f1": values["incorrect_f1"] - baseline["incorrect_f1"],
    }
    failures.extend(
        f"{name} improvement over phone-only: {improvements[name]:.6f} < {threshold:.6f}"
        for name, threshold in PHONE_ONLY_IMPROVEMENTS.items() if improvements[name] < threshold
    )
    return failures, improvements


def main() -> int:
    args = parse_args()
    r2a.set_seed(SEED)
    audio_root = r2a.require_audio_root()
    split_rows, summary = load_train_validation_rows(audio_root)
    print(json.dumps(summary, indent=2))
    preflight = r2a.preflight_audio(split_rows["train"] + split_rows["validation"])
    print("TRAIN/VALIDATION audio preflight:", json.dumps(preflight, indent=2))

    train_counts = Counter(row["_binary_label"] for row in split_rows["train"])
    class_weights = [len(split_rows["train"]) / (2.0 * train_counts[index]) for index in (0, 1)]
    for actual, expected in zip(class_weights, CLASS_WEIGHT_REFERENCE):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"Class weight differs: expected {expected}, got {actual}")
    print(f"Class weights: correct={class_weights[0]:.9f}, incorrect={class_weights[1]:.9f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_audio = r2a.materialize_audio_features(split_rows["train"][:16], device, "R2-B smoke")
    probe = ConditionedFeatureDataset(probe_audio, split_rows["train"][:16])
    probe_model = AudioPhoneCorrectnessModel().to(device).eval()
    probe_model.assert_unk_is_zero()
    with torch.no_grad():
        probe_logits = probe_model(probe.features.to(device), probe.phone_ids.to(device), mode="full")
    if tuple(probe_logits.shape) != (16, 2) or not torch.isfinite(probe_logits).all():
        raise RuntimeError(f"R2-B forward preflight failed: {tuple(probe_logits.shape)}")
    print(f"R2-B forward preflight: PASS logits={tuple(probe_logits.shape)}, device={device}")
    del probe_audio, probe, probe_model, probe_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.preflight_only:
        print("R2-B preflight PASS; no experiment artifacts written and no training performed. TEST untouched.")
        return 0
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")

    EXPERIMENT_DIR.mkdir(parents=True)
    config = config_payload(summary, class_weights)
    r2a.write_json(EXPERIMENT_DIR / "config.json", config)
    environment = r2a.environment_payload(audio_root, device, summary["dataset_sha256"])
    environment.update({"task": "R2-B audio + canonical expected phone", "test_opened": False})
    r2a.write_json(EXPERIMENT_DIR / "environment.json", environment)
    r2a.write_json(EXPERIMENT_DIR / "preflight.json", {"dataset": summary, "train_validation_audio": preflight, "test_opened": False})
    r2a.write_json(
        EXPERIMENT_DIR / "phone_vocab.json",
        {
            "tokens": {phone: index for index, phone in enumerate(PHONE_VOCAB)},
            "vocab_size": len(PHONE_VOCAB),
            "embedding_dimension": EMBEDDING_DIM,
            "padding_idx": 0,
            "unk_fixed_zero": True,
            "canonicalization": "remove vowel stress suffix 0/1/2",
            "rare_unresolved_policy": "TRAIN support <50, malformed, and unseen -> <UNK>",
            "train_support": summary["phone_vocab"]["train_support"],
            "train_unk_tokens": summary["phone_vocab"]["train_unk_tokens"],
        },
    )

    run_started = time.perf_counter()
    train_dataset = materialize_conditioned(split_rows["train"], device, "R2-B train")
    validation_dataset = materialize_conditioned(split_rows["validation"], device, "R2-B validation")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    r2a.set_seed(SEED)
    model = AudioPhoneCorrectnessModel().to(device)
    model.assert_unk_is_zero()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    checkpoint_path = EXPERIMENT_DIR / "R2B_audio_phone_binary_correctness_seed42_best_validation_macro_f1.pt"
    records: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_incorrect_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows_seen = 0
        for features, targets, phone_ids, indexes in tqdm(train_loader, desc=f"R2-B epoch {epoch}/{EPOCHS} train"):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phone_ids = phone_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, phone_ids, mode="full")
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            model.assert_unk_is_zero()
            train_loss_sum += loss.item() * len(targets)
            train_rows_seen += len(targets)

        validation_metrics, _ = evaluate(
            model, validation_dataset, split_rows["validation"], criterion, device, mode="full"
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_rows_seen,
            "validation_full": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        records.append(record)
        save_epoch_outputs(records)
        macro_f1 = validation_metrics["macro_f1"]
        incorrect_f1 = validation_metrics["incorrect"]["f1"]
        better = macro_f1 > best_macro_f1 + 1e-12 or (
            abs(macro_f1 - best_macro_f1) <= 1e-12 and incorrect_f1 > best_incorrect_f1 + 1e-12
        )
        if better:
            best_epoch = epoch
            best_macro_f1 = macro_f1
            best_incorrect_f1 = incorrect_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1": macro_f1,
                    "validation_incorrect_f1": incorrect_f1,
                    "class_to_index": {"correct": 0, "incorrect": 1},
                    "phone_to_id": PHONE_TO_ID,
                    "config": config,
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch} full_val_loss={validation_metrics['loss']:.6f} accuracy={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced_accuracy={validation_metrics['balanced_accuracy']:.6f} "
            f"incorrect_f1={incorrect_f1:.6f} incorrect_recall={validation_metrics['incorrect']['recall']:.6f} "
            f"sub_recall={validation_metrics['subtype_recall']['substitution']['recall']:.6f} "
            f"del_recall={validation_metrics['subtype_recall']['deletion']['recall']:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.assert_unk_is_zero()
    selected_results: dict[str, dict[str, Any]] = {}
    selected_predictions: dict[str, list[int]] = {}
    for mode, filename in (
        ("full", "selected_validation_full.json"),
        ("no_phone", "selected_validation_no_phone.json"),
        ("no_audio", "selected_validation_no_audio.json"),
    ):
        metrics, predictions = evaluate(
            model, validation_dataset, split_rows["validation"], criterion, device, mode=mode
        )
        selected_results[mode] = metrics
        selected_predictions[mode] = predictions
        r2a.write_json(
            EXPERIMENT_DIR / filename,
            {"selected_epoch": best_epoch, "checkpoint_selection_used_this_mode": mode == "full", "metrics": metrics},
        )

    diagnostic_payload = {
        mode: phone_diagnostics(split_rows["validation"], predictions)
        for mode, predictions in selected_predictions.items()
    }
    r2a.write_json(EXPERIMENT_DIR / "phone_diagnostics.json", diagnostic_payload)

    matched = r2a.duration_matched_report(split_rows["validation"], selected_predictions["full"], SEED)
    if matched["pairs"] != 4_434 or matched["rows"] != 8_868:
        raise RuntimeError(f"Duration-matched support differs: pairs={matched['pairs']}, rows={matched['rows']}")
    full = selected_results["full"]
    duration_drops = {
        "macro_f1": full["macro_f1"] - matched["macro_f1"],
        "incorrect_f1": full["incorrect"]["f1"] - matched["incorrect"]["f1"],
    }
    duration_failures = [
        f"{name} full-to-matched drop: {duration_drops[name]:.6f} > {threshold:.6f}"
        for name, threshold in DURATION_DROP_THRESHOLDS.items() if duration_drops[name] > threshold
    ]
    r2a.write_json(
        EXPERIMENT_DIR / "duration_matched_validation.json",
        {
            "selected_epoch": best_epoch,
            "full_metrics": {"macro_f1": full["macro_f1"], "incorrect_f1": full["incorrect"]["f1"]},
            "matched_metrics": matched,
            "full_to_matched_drops": duration_drops,
            "thresholds": DURATION_DROP_THRESHOLDS,
            "failures": duration_failures,
            "gate_pass": not duration_failures,
        },
    )

    primary_failures, improvements = primary_gate(full)
    no_audio = selected_results["no_audio"]
    modality_deltas = {
        "macro_f1_delta": full["macro_f1"] - no_audio["macro_f1"],
        "incorrect_f1_delta": full["incorrect"]["f1"] - no_audio["incorrect"]["f1"],
    }
    modality_failures = [
        f"{name}: {modality_deltas[name]:.6f} < {threshold:.6f}"
        for name, threshold in MODALITY_THRESHOLDS.items() if modality_deltas[name] < threshold
    ]
    if primary_failures:
        status = "R2B_VALIDATION_FAIL"
    elif modality_failures:
        status = "R2B_PHONE_SHORTCUT_FAIL"
    elif duration_failures:
        status = "R2B_DURATION_SHORTCUT_FAIL"
    else:
        status = "R2B_PASS_VALIDATION"
    test_eligible = status == "R2B_PASS_VALIDATION"
    elapsed = time.perf_counter() - run_started

    r2a.write_json(
        EXPERIMENT_DIR / "model_metadata.json",
        {
            "name": "R2-B AudioPhoneCorrectnessModel",
            "task": "binary correctness conditioned on audio + canonical expected phone",
            "research_only": True,
            "used_by_runtime": False,
            "production_model": False,
            "selected_epoch": best_epoch,
            "checkpoint": checkpoint_path.name,
            "validation_macro_f1": best_macro_f1,
            "class_to_index": {"correct": 0, "incorrect": 1},
            "phone_to_id": PHONE_TO_ID,
            "unk_embedding_fixed_zero": True,
            "test_opened": False,
        },
    )
    r2a.write_json(
        EXPERIMENT_DIR / "final_status.json",
        {
            "status": status,
            "selected_epoch": best_epoch,
            "primary_gate": {"pass": not primary_failures, "failures": primary_failures, "improvements": improvements},
            "modality_gate": {"pass": not modality_failures, "failures": modality_failures, "full_minus_no_audio": modality_deltas},
            "duration_gate": {"pass": not duration_failures, "failures": duration_failures, "full_to_matched_drops": duration_drops},
            "test_opened": False,
            "test_eligible": test_eligible,
            "training_and_validation_seconds": elapsed,
        },
    )
    print(json.dumps({"status": status, "test_opened": False, "test_eligible": test_eligible}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
