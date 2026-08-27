from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r2a_correctness_seed42"
EXPECTED_DATASET_SHA256 = "433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E"

TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
SPLIT_SPEAKERS = {
    "train": TRAIN_SPEAKERS,
    "validation": VALIDATION_SPEAKERS,
    "test": TEST_SPEAKERS,
}
EXPECTED_SPLITS = {
    "train": {"rows": 59_572, "incorrect": 8_172},
    "validation": {"rows": 29_767, "incorrect": 4_464},
    "test": {"rows": 29_805, "incorrect": 4_882},
}

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 16_000
N_MELS = 64
N_FFT = 2_048
HOP_LENGTH = 512
WIN_LENGTH = 2_048
EXPECTED_FEATURE_SHAPE = (1, 64, 32)

SEED = 42
BATCH_SIZE = 8
PREPROCESS_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 12
NUM_WORKERS = 0
DROPOUT = 0.2

VALIDATION_THRESHOLDS = {
    "macro_f1": 0.60,
    "balanced_accuracy": 0.60,
    "incorrect_f1": 0.40,
    "incorrect_recall": 0.40,
    "substitution_recall": 0.40,
    "deletion_recall": 0.30,
}
MATCHED_TEST_THRESHOLDS = {
    "macro_f1": 0.55,
    "incorrect_f1": 0.35,
    "max_macro_f1_drop": 0.10,
    "max_incorrect_f1_drop": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked research-only R2-A binary correctness baseline.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate dataset identity, split/count invariants, audio files, and one feature shape without training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_audio_root() -> Path:
    value = os.environ.get("L2_ARCTIC_ROOT", "").strip()
    if not value:
        raise RuntimeError("L2_ARCTIC_ROOT is required and must point to l2arctic_release_v5.0")
    root = Path(value).expanduser().resolve()
    if not root.is_absolute() or not root.is_dir():
        raise RuntimeError(f"L2_ARCTIC_ROOT is not a readable directory: {root}")
    if root.name.lower() != "l2arctic_release_v5.0":
        raise RuntimeError(f"L2_ARCTIC_ROOT must be the l2arctic_release_v5.0 directory, got: {root}")
    return root


def resolve_audio_path(metadata_path: str, audio_root: Path) -> Path:
    normalized = metadata_path.replace("\\", "/")
    parts = Path(normalized).parts
    marker = "l2arctic_release_v5.0"
    indexes = [index for index, part in enumerate(parts) if part.lower() == marker]
    if not indexes:
        raise ValueError(f"Audio reference does not contain the corpus root marker: {metadata_path}")
    relative_parts = parts[indexes[-1] + 1 :]
    if not relative_parts:
        raise ValueError(f"Audio reference has no path below corpus root: {metadata_path}")
    return audio_root.joinpath(*relative_parts)


def load_and_validate_rows(audio_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    actual_hash = sha256_file(METADATA_CSV)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, got {actual_hash}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    required = {"speaker_id", "audio_path", "start_time", "end_time", "error_type", "raw_label", "expected_phone"}
    missing = required - set(source_rows[0] if source_rows else {})
    if missing:
        raise RuntimeError(f"Metadata is missing required fields: {sorted(missing)}")

    all_speakers = set().union(*map(set, SPLIT_SPEAKERS.values()))
    if set(TRAIN_SPEAKERS) & set(VALIDATION_SPEAKERS) or set(TRAIN_SPEAKERS) & set(TEST_SPEAKERS) or set(VALIDATION_SPEAKERS) & set(TEST_SPEAKERS):
        raise RuntimeError("Locked speaker split overlaps")
    if len(all_speakers) != 24:
        raise RuntimeError(f"Locked split must contain 24 unique speakers, got {len(all_speakers)}")

    eligible: list[dict[str, Any]] = []
    category_counts = Counter(row.get("error_type", "") for row in source_rows)
    for source_index, source in enumerate(source_rows):
        error_type = source["error_type"]
        if error_type == "addition":
            continue
        if error_type not in {"correct", "substitution", "deletion"}:
            raise RuntimeError(f"Unexpected training label at source row {source_index + 2}: {error_type!r}")
        speaker = source["speaker_id"]
        if speaker not in all_speakers:
            raise RuntimeError(f"Speaker outside locked split: {speaker}")
        try:
            start = float(source["start_time"])
            end = float(source["end_time"])
        except ValueError as exc:
            raise RuntimeError(f"Invalid interval at source row {source_index + 2}") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise RuntimeError(f"Invalid interval at source row {source_index + 2}: {start}, {end}")
        row = dict(source)
        row["_source_index"] = source_index
        row["_start"] = start
        row["_end"] = end
        row["_duration"] = end - start
        row["_binary_label"] = 0 if error_type == "correct" else 1
        row["_audio_path"] = resolve_audio_path(source["audio_path"], audio_root)
        eligible.append(row)

    counts = Counter(row["error_type"] for row in eligible)
    binary_counts = Counter(row["_binary_label"] for row in eligible)
    expected_categories = {"correct": 101_626, "substitution": 14_098, "deletion": 3_420}
    if dict(counts) != expected_categories:
        raise RuntimeError(f"Eligible category counts differ: expected {expected_categories}, got {dict(counts)}")
    if len(eligible) != 119_144 or binary_counts != Counter({0: 101_626, 1: 17_518}):
        raise RuntimeError(f"Eligible binary counts differ: rows={len(eligible)}, counts={dict(binary_counts)}")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, Any] = {}
    for split_name, speakers in SPLIT_SPEAKERS.items():
        speaker_set = set(speakers)
        rows = [row for row in eligible if row["speaker_id"] in speaker_set]
        split_rows[split_name] = rows
        split_counts = Counter(row["_binary_label"] for row in rows)
        subtype_counts = Counter(row["error_type"] for row in rows)
        expected = EXPECTED_SPLITS[split_name]
        if len(rows) != expected["rows"] or split_counts[1] != expected["incorrect"]:
            raise RuntimeError(
                f"{split_name} count mismatch: expected {expected}, got rows={len(rows)}, incorrect={split_counts[1]}"
            )
        represented = {row["speaker_id"] for row in rows}
        if represented != speaker_set:
            raise RuntimeError(f"{split_name} speaker mismatch: expected {sorted(speaker_set)}, got {sorted(represented)}")
        split_summary[split_name] = {
            "speakers": list(speakers),
            "rows": len(rows),
            "correct": split_counts[0],
            "incorrect": split_counts[1],
            "substitution": subtype_counts["substitution"],
            "deletion": subtype_counts["deletion"],
        }

    summary = {
        "dataset_sha256": actual_hash,
        "source_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "eligible_counts": dict(counts),
        "excluded_addition": category_counts["addition"],
        "splits": split_summary,
    }
    return eligible, split_rows, summary


def preflight_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["_audio_path"]].append(row)

    errors: list[str] = []
    sample_rates = Counter()
    total_seconds = 0.0
    for path, path_rows in tqdm(sorted(by_path.items(), key=lambda item: str(item[0])), desc="Audio preflight"):
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or not math.isfinite(info.duration) or info.duration <= 0:
                errors.append(f"invalid audio duration: {path}")
                continue
            with sf.SoundFile(path, mode="r") as handle:
                probe = handle.read(frames=1, dtype="float32", always_2d=True)
            if probe.size == 0 or not np.isfinite(probe).all():
                errors.append(f"unreadable/invalid first frame: {path}")
                continue
            max_end = max(row["_end"] for row in path_rows)
            if max_end > info.duration + 0.050:
                errors.append(f"annotation exceeds audio duration: {path} ({max_end:.6f} > {info.duration:.6f})")
            for row in path_rows:
                center = (row["_start"] + row["_end"]) / 2.0
                overlap = min(info.duration, center + 0.5) - max(0.0, center - 0.5)
                if overlap <= 0:
                    errors.append(f"empty centered crop: {path} [{row['_start']}, {row['_end']}]")
                    break
            sample_rates[info.samplerate] += 1
            total_seconds += info.duration
        except Exception as exc:
            errors.append(f"unreadable: {path}: {exc}")

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"Audio preflight failed with {len(errors)} error(s):\n{preview}")
    return {
        "unique_wav_files": len(by_path),
        "sample_rates": {str(key): value for key, value in sorted(sample_rates.items())},
        "total_audio_hours_across_unique_wavs": total_seconds / 3600.0,
        "missing": 0,
        "unreadable": 0,
        "invalid_duration": 0,
        "empty_centered_crop": 0,
    }


class SequentialWaveStore:
    """One-file cache; preprocessing walks metadata in source order."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.audio: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        if self.path != path:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"Empty or non-finite decoded audio: {path}")
            self.path = path
            self.audio = audio
        assert self.audio is not None
        return self.audio


def centered_window(audio: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    center_sample = int(round(((start_time + end_time) / 2.0) * SAMPLE_RATE))
    requested_start = center_sample - WINDOW_SAMPLES // 2
    requested_end = requested_start + WINDOW_SAMPLES
    source_start = max(0, requested_start)
    source_end = min(len(audio), requested_end)
    if source_end <= source_start:
        raise RuntimeError(f"Centered crop has no audio samples: start={start_time}, end={end_time}")
    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    destination_start = source_start - requested_start
    destination_end = destination_start + (source_end - source_start)
    output[destination_start:destination_end] = audio[source_start:source_end]
    if output.shape != (WINDOW_SAMPLES,):
        raise RuntimeError(f"Unexpected waveform shape: {output.shape}")
    return output


class AudioWindowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.store = SequentialWaveStore()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        audio = self.store.load(row["_audio_path"])
        window = centered_window(audio, row["_start"], row["_end"])
        return torch.from_numpy(window), torch.tensor(row["_binary_label"], dtype=torch.long), index


class FixedLogMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            window_fn=torch.hann_window,
            power=2.0,
            center=True,
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self.mel(waveform)
        log_power = 10.0 * torch.log10(torch.clamp(power, min=1e-10))
        reference = log_power.amax(dim=(-2, -1), keepdim=True)
        return torch.clamp(log_power - reference, min=-80.0).unsqueeze(1)


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[index], self.labels[index], index


def materialize_audio_features(rows: list[dict[str, Any]], device: torch.device, split_name: str) -> FeatureDataset:
    source = AudioWindowDataset(rows)
    loader = DataLoader(source, batch_size=PREPROCESS_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    extractor = FixedLogMel().to(device).eval()
    features = torch.empty((len(rows), *EXPECTED_FEATURE_SHAPE), dtype=torch.float32)
    labels = torch.empty(len(rows), dtype=torch.long)
    cursor = 0
    with torch.no_grad():
        for waveforms, batch_labels, indexes in tqdm(loader, desc=f"Materialize {split_name} log-mel"):
            if not torch.equal(indexes, torch.arange(cursor, cursor + len(indexes))):
                raise RuntimeError(f"Non-sequential preprocessing order in {split_name}")
            batch_features = extractor(waveforms.to(device, non_blocking=True)).cpu()
            if tuple(batch_features.shape[1:]) != EXPECTED_FEATURE_SHAPE:
                raise RuntimeError(
                    f"Unexpected {split_name} feature shape: {tuple(batch_features.shape[1:])}, expected {EXPECTED_FEATURE_SHAPE}"
                )
            next_cursor = cursor + len(indexes)
            features[cursor:next_cursor].copy_(batch_features)
            labels[cursor:next_cursor].copy_(batch_labels)
            cursor = next_cursor
    if cursor != len(rows) or not torch.isfinite(features).all():
        raise RuntimeError(f"Incomplete or non-finite materialized features for {split_name}")
    return FeatureDataset(features, labels)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = feature_map.mean(dim=2).transpose(1, 2)
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SmallPronunciationCNNAttention(nn.Module):
    def __init__(self, num_classes: int = 2):
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
        self.attention = TemporalAttentionPooling(channels=96)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(96, num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(features)
        pooled, _ = self.attention(feature_map)
        return self.classifier(pooled)


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "rows": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "correct": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "incorrect": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def add_audit_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    metrics = dict(metrics)
    subtype_recall = {}
    for subtype in ("substitution", "deletion"):
        positions = [index for index, row in enumerate(rows) if row["error_type"] == subtype]
        subtype_recall[subtype] = {
            "support": len(positions),
            "recall": sum(predictions[index] == 1 for index in positions) / len(positions) if positions else 0.0,
        }
    per_speaker = {}
    for speaker in sorted({row["speaker_id"] for row in rows}):
        positions = [index for index, row in enumerate(rows) if row["speaker_id"] == speaker]
        speaker_labels = [rows[index]["_binary_label"] for index in positions]
        speaker_predictions = [predictions[index] for index in positions]
        per_speaker[speaker] = binary_metrics(speaker_labels, speaker_predictions)
    metrics["subtype_recall"] = subtype_recall
    metrics["per_speaker"] = per_speaker
    return metrics


def evaluate(
    model: nn.Module,
    dataset: FeatureDataset,
    rows: list[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], list[int]]:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda"
    )
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for features, targets, indexes in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
    expected_labels = [row["_binary_label"] for row in rows]
    if labels != expected_labels:
        raise RuntimeError("Evaluation order/label invariant failed")
    metrics = binary_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(labels)
    return add_audit_metrics(metrics, rows, predictions), predictions


def duration_matched_report(rows: list[dict[str, Any]], predictions: list[int], seed: int) -> dict[str, Any]:
    by_bin: dict[int, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(rows):
        duration_bin = int(math.floor((row["_duration"] + 1e-12) / 0.010))
        by_bin[duration_bin][row["_binary_label"]].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    bin_support: dict[str, int] = {}
    for duration_bin in sorted(by_bin):
        correct_positions = sorted(by_bin[duration_bin][0])
        incorrect_positions = sorted(by_bin[duration_bin][1])
        pairs = min(len(correct_positions), len(incorrect_positions))
        if not pairs:
            continue
        selected.extend(rng.sample(correct_positions, pairs))
        selected.extend(rng.sample(incorrect_positions, pairs))
        bin_support[str(duration_bin)] = pairs
    selected.sort()
    labels = [rows[index]["_binary_label"] for index in selected]
    selected_predictions = [predictions[index] for index in selected]
    report = binary_metrics(labels, selected_predictions)
    report.update(
        {
            "duration_bin_seconds": 0.010,
            "seed": seed,
            "sampling": "equal correct/incorrect per bin, without replacement",
            "pairs": len(selected) // 2,
            "bin_pair_support": bin_support,
        }
    )
    return report


def gate_failures(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    observed = {
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "incorrect_f1": metrics["incorrect"]["f1"],
        "incorrect_recall": metrics["incorrect"]["recall"],
        "substitution_recall": metrics["subtype_recall"]["substitution"]["recall"],
        "deletion_recall": metrics["subtype_recall"]["deletion"]["recall"],
    }
    return [f"{name}: {observed[name]:.6f} < {threshold:.6f}" for name, threshold in thresholds.items() if observed[name] < threshold]


def config_payload(dataset_summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    return {
        "experiment": "R2-A binary correctness baseline",
        "research_only": True,
        "used_by_runtime": False,
        "production_model": False,
        "dataset": str(METADATA_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": dataset_summary["dataset_sha256"],
        "eligible_labels": {"correct": 0, "substitution": 1, "deletion": 1},
        "excluded_labels": ["addition"],
        "speaker_split": {name: list(speakers) for name, speakers in SPLIT_SPEAKERS.items()},
        "split_counts": dataset_summary["splits"],
        "feature_allowlist": ["audio waveform"],
        "forbidden_model_features": [
            "error_type", "raw_label", "label", "expected_phone", "observed_phone", "speaker_id", "l1", "gender",
            "split", "utterance_id", "filename", "path", "duration", "start_time", "end_time",
        ],
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "mono": True,
            "crop": "one second centered at (start_time + end_time) / 2 for every class",
            "samples": WINDOW_SAMPLES,
            "boundary_policy": "clip available utterance audio and zero-pad only the missing side",
        },
        "log_mel": {
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "window": "hann",
            "center": True,
            "pad_mode": "constant",
            "power": 2.0,
            "power_to_db_ref": "max",
            "top_db": 80.0,
            "mel_norm": "slaney (legacy librosa default)",
            "feature_shape": list(EXPECTED_FEATURE_SHAPE),
        },
        "model": "SmallPronunciationCNNAttention",
        "backbone_channels": [16, 32, 64, 96],
        "head": {"dropout": DROPOUT, "linear": [96, 2]},
        "random_initialization": True,
        "loss": "class-weighted CrossEntropy",
        "class_weight_formula": "N / (2 * train_class_count[c])",
        "class_weights": {"correct": class_weights[0], "incorrect": class_weights[1]},
        "sampler": "standard shuffled DataLoader",
        "augmentation": "none",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "threshold": "argmax",
        "checkpoint_selection": "highest validation Macro-F1; tie higher incorrect F1; tie earlier epoch",
        "validation_thresholds": VALIDATION_THRESHOLDS,
        "duration_matched_test_thresholds": MATCHED_TEST_THRESHOLDS,
    }


def environment_payload(audio_root: Path, device: torch.device, dataset_hash: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_sha256": dataset_hash,
        "git_commit": git_commit(),
        "l2_arctic_root": str(audio_root),
        "l2_arctic_root_identity": audio_root.name,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "num_workers": NUM_WORKERS,
    }


def save_epoch_outputs(records: list[dict[str, Any]]) -> None:
    write_json(EXPERIMENT_DIR / "epoch_metrics.json", records)
    fieldnames = [
        "epoch", "train_loss", "validation_loss", "accuracy", "macro_f1", "balanced_accuracy",
        "incorrect_precision", "incorrect_recall", "incorrect_f1", "substitution_recall", "deletion_recall",
        "tn", "fp", "fn", "tp", "epoch_seconds",
    ]
    with (EXPERIMENT_DIR / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record["validation"]
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
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "epoch_seconds": record["epoch_seconds"],
                }
            )


def main() -> int:
    args = parse_args()
    set_seed(SEED)
    audio_root = require_audio_root()
    _, split_rows, dataset_summary = load_and_validate_rows(audio_root)
    print(json.dumps(dataset_summary, indent=2))
    audio_preflight = preflight_audio([row for name in ("train", "validation", "test") for row in split_rows[name]])
    print("Audio preflight:", json.dumps(audio_preflight, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_row = split_rows["train"][0]
    probe_audio, _ = librosa.load(probe_row["_audio_path"], sr=SAMPLE_RATE, mono=True)
    probe_window = centered_window(np.asarray(probe_audio, dtype=np.float32), probe_row["_start"], probe_row["_end"])
    with torch.no_grad():
        probe_feature = FixedLogMel().to(device)(torch.from_numpy(probe_window).unsqueeze(0).to(device))
    if tuple(probe_feature.shape[1:]) != EXPECTED_FEATURE_SHAPE:
        raise RuntimeError(f"Feature-shape preflight failed: {tuple(probe_feature.shape)}")
    print(f"Feature-shape preflight: PASS {tuple(probe_feature.shape)}")

    train_counts = Counter(row["_binary_label"] for row in split_rows["train"])
    class_weights = [len(split_rows["train"]) / (2.0 * train_counts[index]) for index in (0, 1)]
    print(f"Class weights: correct={class_weights[0]:.9f}, incorrect={class_weights[1]:.9f}")
    print(f"Device: {device}")
    if args.preflight_only:
        print("R2-A preflight PASS; no experiment artifacts written and no training performed.")
        return 0

    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")
    EXPERIMENT_DIR.mkdir(parents=True)
    write_json(EXPERIMENT_DIR / "config.json", config_payload(dataset_summary, class_weights))
    write_json(EXPERIMENT_DIR / "environment.json", environment_payload(audio_root, device, dataset_summary["dataset_sha256"]))
    write_json(EXPERIMENT_DIR / "preflight.json", {"dataset": dataset_summary, "audio": audio_preflight})

    run_started = time.perf_counter()
    train_dataset = materialize_audio_features(split_rows["train"], device, "train")
    validation_dataset = materialize_audio_features(split_rows["validation"], device, "validation")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = SmallPronunciationCNNAttention(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = EXPERIMENT_DIR / "R2A_binary_correctness_seed42_best_validation_macro_f1.pt"
    epoch_records: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_incorrect_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows_seen = 0
        for features, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train"):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_rows_seen += len(targets)

        validation_metrics, _ = evaluate(model, validation_dataset, split_rows["validation"], criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_rows_seen,
            "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_records.append(record)
        save_epoch_outputs(epoch_records)
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
                    "config": config_payload(dataset_summary, class_weights),
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} accuracy={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced_accuracy={validation_metrics['balanced_accuracy']:.6f} "
            f"incorrect_f1={incorrect_f1:.6f} incorrect_recall={validation_metrics['incorrect']['recall']:.6f} "
            f"sub_recall={validation_metrics['subtype_recall']['substitution']['recall']:.6f} "
            f"del_recall={validation_metrics['subtype_recall']['deletion']['recall']:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_validation, selected_validation_predictions = evaluate(
        model, validation_dataset, split_rows["validation"], criterion, device
    )
    validation_matched = duration_matched_report(split_rows["validation"], selected_validation_predictions, SEED)
    validation_failures = gate_failures(selected_validation, VALIDATION_THRESHOLDS)
    validation_report = {
        "selected_epoch": best_epoch,
        "selection_metric": "validation Macro-F1",
        "metrics": selected_validation,
        "gate_thresholds": VALIDATION_THRESHOLDS,
        "gate_failures": validation_failures,
        "gate_pass": not validation_failures,
        "test_opened": False,
    }
    write_json(EXPERIMENT_DIR / "validation_report.json", validation_report)
    duration_reports: dict[str, Any] = {"validation": validation_matched}
    write_json(EXPERIMENT_DIR / "duration_matched_report.json", duration_reports)
    write_json(
        EXPERIMENT_DIR / "model_metadata.json",
        {
            "name": "SmallPronunciationCNNAttention",
            "task": "R2-A binary correctness",
            "research_only": True,
            "used_by_runtime": False,
            "production_model": False,
            "random_initialization": True,
            "loaded_legacy_checkpoint": False,
            "selected_epoch": best_epoch,
            "checkpoint": checkpoint_path.name,
            "validation_macro_f1": best_macro_f1,
            "class_to_index": {"correct": 0, "incorrect": 1},
        },
    )

    if validation_failures:
        elapsed = time.perf_counter() - run_started
        validation_report["training_and_evaluation_seconds"] = elapsed
        write_json(EXPERIMENT_DIR / "validation_report.json", validation_report)
        write_json(
            EXPERIMENT_DIR / "final_status.json",
            {"status": "R2A_VALIDATION_FAIL", "test_opened": False, "failures": validation_failures, "elapsed_seconds": elapsed},
        )
        print("Validation gate FAIL; test set was NOT materialized or evaluated.")
        print("R2A_VALIDATION_FAIL")
        return 0

    validation_report["test_opened"] = True
    write_json(EXPERIMENT_DIR / "validation_report.json", validation_report)
    test_dataset = materialize_audio_features(split_rows["test"], device, "test")
    test_metrics, test_predictions = evaluate(model, test_dataset, split_rows["test"], criterion, device)
    test_failures = gate_failures(test_metrics, VALIDATION_THRESHOLDS)
    test_matched = duration_matched_report(split_rows["test"], test_predictions, SEED)
    duration_reports["test"] = test_matched
    write_json(EXPERIMENT_DIR / "duration_matched_report.json", duration_reports)

    macro_drop = test_metrics["macro_f1"] - test_matched["macro_f1"]
    incorrect_f1_drop = test_metrics["incorrect"]["f1"] - test_matched["incorrect"]["f1"]
    shortcut_failures = []
    if test_matched["macro_f1"] < MATCHED_TEST_THRESHOLDS["macro_f1"]:
        shortcut_failures.append(
            f"duration-matched macro_f1: {test_matched['macro_f1']:.6f} < {MATCHED_TEST_THRESHOLDS['macro_f1']:.6f}"
        )
    if test_matched["incorrect"]["f1"] < MATCHED_TEST_THRESHOLDS["incorrect_f1"]:
        shortcut_failures.append(
            f"duration-matched incorrect_f1: {test_matched['incorrect']['f1']:.6f} < {MATCHED_TEST_THRESHOLDS['incorrect_f1']:.6f}"
        )
    if macro_drop > MATCHED_TEST_THRESHOLDS["max_macro_f1_drop"]:
        shortcut_failures.append(
            f"full-to-matched macro_f1 drop: {macro_drop:.6f} > {MATCHED_TEST_THRESHOLDS['max_macro_f1_drop']:.6f}"
        )
    if incorrect_f1_drop > MATCHED_TEST_THRESHOLDS["max_incorrect_f1_drop"]:
        shortcut_failures.append(
            f"full-to-matched incorrect_f1 drop: {incorrect_f1_drop:.6f} > {MATCHED_TEST_THRESHOLDS['max_incorrect_f1_drop']:.6f}"
        )
    elapsed = time.perf_counter() - run_started
    test_report = {
        "selected_epoch": best_epoch,
        "metrics": test_metrics,
        "normal_gate_thresholds": VALIDATION_THRESHOLDS,
        "normal_gate_failures": test_failures,
        "normal_gate_pass": not test_failures,
        "duration_matched": {
            "metrics": test_matched,
            "macro_f1_drop": macro_drop,
            "incorrect_f1_drop": incorrect_f1_drop,
            "thresholds": MATCHED_TEST_THRESHOLDS,
            "failures": shortcut_failures,
            "gate_pass": not shortcut_failures,
        },
        "training_and_evaluation_seconds": elapsed,
    }
    write_json(EXPERIMENT_DIR / "test_report.json", test_report)
    if test_failures:
        status = "R2A_TEST_FAIL"
    elif shortcut_failures:
        status = "R2A_SHORTCUT_FAIL"
    else:
        status = "R2A_PASS"
    write_json(
        EXPERIMENT_DIR / "final_status.json",
        {
            "status": status,
            "test_opened": True,
            "test_failures": test_failures,
            "shortcut_failures": shortcut_failures,
            "elapsed_seconds": elapsed,
        },
    )
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
