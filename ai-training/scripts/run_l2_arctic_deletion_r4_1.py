from __future__ import annotations

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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
MATCHED_PATH = REPO_ROOT / "ai-training/experiments/r4_0_deletion_feasibility/matched_control_validation_rows.csv"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_1_controlled_deletion_seed42"
CHECKPOINT_NAME = "R4_1_controlled_deletion_audio_phone_seed42_best_validation_macro_f1.pt"

EXPECTED_V4_SHA256 = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"
EXPECTED_MATCHED_SHA256 = "864591A259F0C7E6F16828C5F755049CFCF0F79A8B586ADA354D190F9C5C7823"
TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
SPLITS = {"train": TRAIN_SPEAKERS, "validation": VALIDATION_SPEAKERS, "test": TEST_SPEAKERS}
EXPECTED_COUNTS = {
    "train": {"correct": 51_400, "substitution": 6_159, "deletion": 1_612},
    "validation": {"correct": 25_303, "substitution": 2_909, "deletion": 979},
    "test": {"correct": 24_923, "substitution": 3_293, "deletion": 827},
}

PHONE_VOCAB = (
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_VOCAB)}

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 0.30
WINDOW_SAMPLES = 4_800
N_MELS = 64
N_FFT = 2_048
HOP_LENGTH = 512
WIN_LENGTH = 2_048
EXPECTED_FEATURE_SHAPE = (1, 64, 10)
EMBEDDING_DIM = 16
AUDIO_VECTOR_DIM = 96

SEED = 42
BATCH_SIZE = 8
PREPROCESS_BATCH_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
EPOCHS = 24
DROPOUT = 0.2

DURATION_BASELINE = {"macro_f1": 0.6681455065707653, "balanced_accuracy": 0.7063009965386512, "deletion_f1": 0.3641642472003318}
MATCHED_DURATION_BASELINE = {"macro_f1": 0.4850189762297753, "balanced_accuracy": 0.4993606138107417}
FULL_GATES = {"macro_f1": 0.70, "balanced_accuracy": 0.72, "deletion_f1": 0.40, "deletion_recall": 0.45}
MATCHED_GATES = {"macro_f1": 0.60, "balanced_accuracy": 0.60, "deletion_f1": 0.55}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


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
    indexes = [index for index, part in enumerate(parts) if part.lower() == "l2arctic_release_v5.0"]
    if not indexes:
        raise RuntimeError(f"Audio reference lacks corpus marker: {reference}")
    return root.joinpath(*parts[indexes[-1] + 1 :])


def split_for(speaker: str) -> str:
    for name, speakers in SPLITS.items():
        if speaker in speakers:
            return name
    raise RuntimeError(f"Speaker outside S1: {speaker}")


def duration_bin(duration: float) -> int:
    return int(math.floor(duration / 0.010 + 1e-9))


def load_and_verify(audio_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], set[int]]:
    v4_sha = sha256_file(V4_PATH)
    matched_sha = sha256_file(MATCHED_PATH)
    if v4_sha != EXPECTED_V4_SHA256:
        raise RuntimeError(f"V4 SHA mismatch: {v4_sha}")
    if matched_sha != EXPECTED_MATCHED_SHA256:
        raise RuntimeError(f"R4_1_MATCHED_CONTROL_FAIL: matched CSV SHA mismatch: {matched_sha}")
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 135_890:
        raise RuntimeError(f"V4 raw row mismatch: {len(source_rows)}")

    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    counts = {name: Counter() for name in SPLITS}
    for source_index, source in enumerate(source_rows):
        if source["label_quality"] != "clean" or source["relation"] not in {"correct", "substitution", "deletion"}:
            continue
        split = split_for(source["speaker_id"])
        phone = source["expected_phone_canonical"]
        if phone not in PHONE_TO_ID:
            raise RuntimeError(f"Invalid canonical expected phone at row {source_index + 2}: {phone!r}")
        start, end = float(source["start_time"]), float(source["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise RuntimeError(f"Invalid interval at V4 row {source_index + 2}")
        row = dict(source)
        row.update({
            "_source_index": source_index, "_split": split, "_start": start, "_end": end,
            "_target": 1 if source["relation"] == "deletion" else 0,
            "_phone_id": PHONE_TO_ID[phone], "_duration_bin": duration_bin(end - start),
        })
        # R4 TEST is metadata-only: never resolve its audio path.
        if split != "test":
            row["_audio_path"] = resolve_audio_path(source["audio_path"], audio_root)
        split_rows[split].append(row)
        counts[split][source["relation"]] += 1
    for split, expected in EXPECTED_COUNTS.items():
        actual = {relation: counts[split][relation] for relation in expected}
        if actual != expected:
            raise RuntimeError(f"{split} count mismatch: {actual}")
    speaker_sets = [set(speakers) for speakers in SPLITS.values()]
    if any(speaker_sets[i] & speaker_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("S1 speaker overlap")

    validation_by_source = {row["_source_index"]: row for row in split_rows["validation"]}
    matched_sources: set[int] = set()
    matched_groups: dict[tuple[str, str, int], Counter[int]] = defaultdict(Counter)
    with MATCHED_PATH.open(encoding="utf-8", newline="") as handle:
        matched_rows = list(csv.DictReader(handle))
    if len(matched_rows) != 1_564:
        raise RuntimeError(f"R4_1_MATCHED_CONTROL_FAIL: rows={len(matched_rows)}")
    for record in matched_rows:
        source_index = int(record["source_csv_row"]) - 2
        if source_index in matched_sources or source_index not in validation_by_source:
            raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL: duplicate or non-validation source identity")
        row = validation_by_source[source_index]
        fields = ("speaker_id", "audio_path", "utterance_id", "start_time", "end_time", "duration", "expected_phone_canonical", "relation")
        if any(str(row[field]) != str(record[field]) for field in fields):
            raise RuntimeError(f"R4_1_MATCHED_CONTROL_FAIL: identity mismatch at source row {source_index + 2}")
        if int(record["duration_bin_10ms"]) != row["_duration_bin"]:
            raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL: duration bin mismatch")
        matched_sources.add(source_index)
        matched_groups[(row["speaker_id"], row["expected_phone_canonical"], row["_duration_bin"])][row["_target"]] += 1
    if sum(validation_by_source[index]["_target"] for index in matched_sources) != 782:
        raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL: deletion count is not 782")
    unequal = {key: dict(value) for key, value in matched_groups.items() if value[0] != value[1] or not value[0]}
    if unequal:
        raise RuntimeError(f"R4_1_MATCHED_CONTROL_FAIL: unpaired strata: {list(unequal.items())[:5]}")
    if {validation_by_source[index]["speaker_id"] for index in matched_sources} != set(VALIDATION_SPEAKERS):
        raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL: speaker coverage")
    return split_rows, {
        "dataset_sha256": v4_sha, "matched_control_sha256": matched_sha,
        "splits": {
            split: {
                "rows": len(rows), "correct": counts[split]["correct"],
                "substitution": counts[split]["substitution"], "deletion": counts[split]["deletion"],
                "speakers": list(SPLITS[split]),
            }
            for split, rows in split_rows.items()
        },
        "matched_control": {
            "status": "PASS", "rows": len(matched_sources), "deletion": 782, "non_deletion": 782,
            "pairs": 782, "strata": len(matched_groups), "speakers": list(VALIDATION_SPEAKERS),
            "matching": "same speaker + canonical expected phone + half-open 10 ms annotation-duration bin",
            "rebuilt_or_resampled": False,
        },
        "test_audio_paths_resolved": False, "test_audio_accessed": False,
    }, matched_sources


def preflight_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["_audio_path"]].append(row)
    errors: list[str] = []
    rates: Counter[int] = Counter()
    padding = Counter()
    for path, path_rows in tqdm(sorted(by_path.items(), key=lambda item: str(item[0])), desc="R4-1 TRAIN+VALIDATION preflight"):
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0 or not math.isfinite(info.duration) or info.duration <= 0:
                errors.append(f"invalid: {path}")
                continue
            if max(row["_end"] for row in path_rows) > info.duration + 0.050:
                errors.append(f"annotation exceeds audio: {path}")
            for row in path_rows:
                center = (row["_start"] + row["_end"]) / 2.0
                overlap = min(info.duration, center + WINDOW_SECONDS / 2) - max(0.0, center - WINDOW_SECONDS / 2)
                if overlap <= 0:
                    errors.append(f"empty crop: {path}")
                padding[(row["relation"], center - WINDOW_SECONDS / 2 < 0 or center + WINDOW_SECONDS / 2 > info.duration)] += 1
            rates[info.samplerate] += 1
        except Exception as exc:
            errors.append(f"unreadable {path}: {exc}")
    if errors:
        raise RuntimeError(f"Audio preflight failed ({len(errors)}):\n" + "\n".join(errors[:20]))
    return {
        "scope": "TRAIN+VALIDATION only", "rows": len(rows), "unique_wav_files": len(by_path),
        "original_sample_rates": dict(sorted(rates.items())), "decoded_sample_rate": SAMPLE_RATE,
        "missing": 0, "unreadable": 0,
        "edge_padding_by_relation": {
            relation: {"padded": padding[(relation, True)], "not_padded": padding[(relation, False)]}
            for relation in ("correct", "substitution", "deletion")
        },
        "test_audio_accessed": False,
    }


class SequentialWaveStore:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.audio: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        if self.path != path:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if not len(audio) or not np.isfinite(audio).all():
                raise RuntimeError(f"Empty/non-finite decoded audio: {path}")
            self.path, self.audio = path, audio
        assert self.audio is not None
        return self.audio


def centered_window(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    center = int(round(((start + end) / 2.0) * SAMPLE_RATE))
    requested_start = center - WINDOW_SAMPLES // 2
    requested_end = requested_start + WINDOW_SAMPLES
    source_start, source_end = max(0, requested_start), min(len(audio), requested_end)
    if source_end <= source_start:
        raise RuntimeError(f"Empty centered crop: {start}, {end}")
    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    destination_start = source_start - requested_start
    output[destination_start : destination_start + source_end - source_start] = audio[source_start:source_end]
    return output


class AudioWindowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.store = SequentialWaveStore()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio = self.store.load(row["_audio_path"])
        return torch.from_numpy(centered_window(audio, row["_start"], row["_end"])), index


class FixedLogMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
            n_mels=N_MELS, window_fn=torch.hann_window, power=2.0, center=True,
            pad_mode="constant", norm="slaney", mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self.mel(waveform)
        log_power = 10.0 * torch.log10(torch.clamp(power, min=1e-10))
        reference = log_power.amax(dim=(-2, -1), keepdim=True)
        return torch.clamp(log_power - reference, min=-80.0).unsqueeze(1)


class ConditionedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, rows: list[dict[str, Any]]) -> None:
        self.features = features
        self.labels = torch.tensor([row["_target"] for row in rows], dtype=torch.long)
        self.phone_ids = torch.tensor([row["_phone_id"] for row in rows], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.phone_ids[index], index


def materialize(rows: list[dict[str, Any]], device: torch.device, name: str) -> ConditionedFeatureDataset:
    source = AudioWindowDataset(rows)
    loader = DataLoader(source, batch_size=PREPROCESS_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    extractor = FixedLogMel().to(device).eval()
    features = torch.empty((len(rows), *EXPECTED_FEATURE_SHAPE), dtype=torch.float32)
    cursor = 0
    with torch.inference_mode():
        for waveforms, indexes in tqdm(loader, desc=f"Materialize {name} log-mel"):
            if not torch.equal(indexes, torch.arange(cursor, cursor + len(indexes))):
                raise RuntimeError(f"Non-sequential feature order: {name}")
            batch = extractor(waveforms.to(device, non_blocking=True)).cpu()
            if tuple(batch.shape[1:]) != EXPECTED_FEATURE_SHAPE:
                raise RuntimeError(f"Feature shape mismatch: {tuple(batch.shape[1:])}")
            features[cursor : cursor + len(batch)].copy_(batch)
            cursor += len(batch)
    if cursor != len(rows) or not torch.isfinite(features).all():
        raise RuntimeError(f"Incomplete/non-finite features: {name}")
    return ConditionedFeatureDataset(features, rows)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        sequence = feature_map.mean(dim=2).transpose(1, 2)
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class ControlledDeletionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.attention = TemporalAttentionPooling(AUDIO_VECTOR_DIM)
        self.phone_embedding = nn.Embedding(len(PHONE_VOCAB), EMBEDDING_DIM)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(AUDIO_VECTOR_DIM + EMBEDDING_DIM, 2))

    def forward(self, features: torch.Tensor, phone_ids: torch.Tensor, mode: str = "full") -> torch.Tensor:
        if mode not in {"full", "no_audio", "no_phone"}:
            raise ValueError(mode)
        audio_vector = self.attention(self.features(features))
        phone_vector = self.phone_embedding(phone_ids)
        if mode == "no_audio":
            audio_vector = torch.zeros_like(audio_vector)
        if mode == "no_phone":
            phone_vector = torch.zeros_like(phone_vector)
        return self.classifier(torch.cat((audio_vector, phone_vector), dim=1))


def binary_metrics(truth: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.int64)
    predicted_array = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(truth_array, predicted_array, labels=[0, 1]).astype(int)
    tn, fp, fn, tp = (int(item) for item in matrix.ravel())
    non_precision = tn / (tn + fn) if tn + fn else 0.0
    non_recall = tn / (tn + fp) if tn + fp else 0.0
    non_f1 = 2 * non_precision * non_recall / (non_precision + non_recall) if non_precision + non_recall else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": len(truth_array), "non_deletion_support": int(np.sum(truth_array == 0)), "deletion_support": int(np.sum(truth_array == 1)),
        "accuracy": float((tn + tp) / len(truth_array)), "balanced_accuracy": float((non_recall + recall) / 2),
        "macro_f1": float((non_f1 + f1) / 2), "deletion_precision": float(precision),
        "deletion_recall": float(recall), "deletion_f1": float(f1), "confusion_matrix": matrix.tolist(),
    }


def add_origin_rates(metrics: dict[str, Any], rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["origin_false_deletion_rate"] = {
        relation: {
            "support": sum(row["relation"] == relation for row in rows),
            "false_deletion_rate": float(np.mean([predictions[i] == 1 for i, row in enumerate(rows) if row["relation"] == relation])),
        }
        for relation in ("correct", "substitution")
    }
    return metrics


def evaluate(
    model: ControlledDeletionModel, dataset: ConditionedFeatureDataset, rows: list[dict[str, Any]],
    criterion: nn.Module, device: torch.device, mode: str,
) -> tuple[dict[str, Any], list[int], list[float]]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    indexes_seen: list[int] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for features, targets, phone_ids, indexes in loader:
            features = features.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            logits = model(features, phone_ids.to(device, non_blocking=True), mode=mode)
            loss_sum += criterion(logits, targets_device).item() * len(targets)
            probability = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(targets.tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            indexes_seen.extend(indexes.tolist())
    if indexes_seen != list(range(len(rows))) or labels != [row["_target"] for row in rows]:
        raise RuntimeError(f"Evaluation order invariant failed: {mode}")
    metrics = add_origin_rates(binary_metrics(labels, predictions), rows, predictions)
    metrics.update({"loss": loss_sum / len(rows), "mode": mode, "threshold": "argmax"})
    return metrics, predictions, probabilities


def subset_metrics(rows: list[dict[str, Any]], predictions: list[int], positions: list[int]) -> dict[str, Any]:
    return binary_metrics([rows[i]["_target"] for i in positions], [predictions[i] for i in positions])


def save_epochs(records: list[dict[str, Any]]) -> None:
    write_json(EXPERIMENT_DIR / "epoch_metrics.json", records)
    fields = [
        "epoch", "train_loss", "validation_loss", "accuracy", "balanced_accuracy", "macro_f1",
        "deletion_precision", "deletion_recall", "deletion_f1", "correct_false_deletion_rate",
        "substitution_false_deletion_rate", "epoch_seconds",
    ]
    with (EXPERIMENT_DIR / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["validation_full"]
            writer.writerow({
                "epoch": record["epoch"], "train_loss": record["train_loss"], "validation_loss": metrics["loss"],
                "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"], "deletion_precision": metrics["deletion_precision"],
                "deletion_recall": metrics["deletion_recall"], "deletion_f1": metrics["deletion_f1"],
                "correct_false_deletion_rate": metrics["origin_false_deletion_rate"]["correct"]["false_deletion_rate"],
                "substitution_false_deletion_rate": metrics["origin_false_deletion_rate"]["substitution"]["false_deletion_rate"],
                "epoch_seconds": record["epoch_seconds"],
            })


def mode_diagnostics(rows: list[dict[str, Any]], predictions: list[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    speakers: dict[str, Any] = {}
    for speaker in VALIDATION_SPEAKERS:
        positions = [i for i, row in enumerate(rows) if row["speaker_id"] == speaker]
        speakers[speaker] = subset_metrics(rows, predictions, positions)
    validation_deletions = Counter(row["expected_phone_canonical"] for row in rows if row["_target"] == 1)
    focus = set(("D", "T", "R", "L", "N", "Z", "V", "K"))
    focus.update(phone for phone, count in validation_deletions.items() if count >= 20)
    phones: dict[str, Any] = {}
    for phone in sorted(focus, key=lambda item: PHONE_TO_ID[item]):
        positions = [i for i, row in enumerate(rows) if row["expected_phone_canonical"] == phone]
        phones[phone] = subset_metrics(rows, predictions, positions)
    dominant = {"D", "T", "R", "L"}
    concentration = {}
    for name, include in (("D_T_R_L", lambda phone: phone in dominant), ("other_phones", lambda phone: phone not in dominant)):
        positions = [i for i, row in enumerate(rows) if include(row["expected_phone_canonical"])]
        concentration[name] = subset_metrics(rows, predictions, positions)
    return speakers, phones, concentration


def config_payload(summary: dict[str, Any], weights: list[float]) -> dict[str, Any]:
    return {
        "experiment": "R4-1 one controlled deletion detector",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "R4_TEST_CLOSED"],
        "dataset": str(V4_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": summary["dataset_sha256"], "matched_control_sha256": summary["matched_control_sha256"],
        "labels": {"non_deletion": 0, "deletion": 1},
        "non_deletion_sources": ["clean correct", "clean substitution"], "positive_source": "DELETION_ELIGIBLE",
        "excluded": ["addition", "non_speech", "unresolved"],
        "speaker_split": {"train": list(TRAIN_SPEAKERS), "validation": list(VALIDATION_SPEAKERS), "test_closed": list(TEST_SPEAKERS)},
        "split_counts": summary["splits"], "phone_vocabulary": {phone: index for index, phone in enumerate(PHONE_VOCAB)},
        "model_inputs": ["audio-derived log-mel", "canonical expected-phone id"],
        "forbidden_model_features": [
            "relation", "raw annotation", "observed phone", "speaker", "duration", "start/end numeric values",
            "filename/path tokens", "gender", "L1", "variable length", "duration-derived mask",
        ],
        "audio": {"sample_rate": SAMPLE_RATE, "mono": True, "window_seconds": WINDOW_SECONDS, "samples": WINDOW_SAMPLES,
                  "crop": "center=(start_time+end_time)/2 uniformly for every relation", "edge_policy": "zero-pad missing utterance-edge samples only"},
        "log_mel": {"n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH,
                    "window": "hann", "center": True, "pad_mode": "constant", "power": 2.0,
                    "mel_scale": "slaney", "norm": "slaney", "power_to_db_ref": "per-sample max", "top_db": 80.0,
                    "feature_shape": list(EXPECTED_FEATURE_SHAPE)},
        "architecture": {"audio_channels": [16, 32, 64, 96], "temporal_attention": True, "audio_vector": 96,
                         "phone_embedding": [40, EMBEDDING_DIM], "fusion": 112, "head": "Dropout(0.2)+Linear(112,2)",
                         "hidden_mlp": False, "random_initialization": True, "pretrained_checkpoint": None},
        "seed": SEED, "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0, "epochs": EPOCHS, "early_stopping": False,
        "loss": "TRAIN-only class-weighted CrossEntropy", "class_weights": {"non_deletion": weights[0], "deletion": weights[1]},
        "sampler": "none", "augmentation": "none", "focal_loss": False, "prediction": "argmax",
        "checkpoint_selection": "highest FULL validation Macro-F1; tie higher deletion F1; tie earlier epoch",
        "duration_baseline_frozen": DURATION_BASELINE, "matched_duration_baseline_frozen": MATCHED_DURATION_BASELINE,
        "full_gates": {**FULL_GATES, "macro_f1_gain_over_duration_min": 0.03}, "matched_gates": MATCHED_GATES,
        "modality_gates": {"full_minus_no_audio_macro_f1": 0.05, "full_minus_no_audio_deletion_f1": 0.05,
                           "full_minus_no_phone_macro_f1_or_deletion_f1": 0.02},
        "speaker_gate": {"minimum_deletion_support": 30, "deletion_recall_min": 0.25},
        "weak_signal_rule": "FULL Macro-F1>=0.55 OR matched balanced accuracy>=0.55 OR FULL-NO_AUDIO matched Macro-F1>=0.02",
        "test_policy": "R4 TEST audio paths are never resolved; no TEST features or inference",
    }


def environment_payload(audio_root: Path, device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "platform": platform.platform(),
        "pytorch": torch.__version__, "torchaudio": torchaudio.__version__, "librosa": librosa.__version__,
        "numpy": np.__version__, "device": str(device), "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(), "l2_arctic_root": str(audio_root), "l2_arctic_root_identity": audio_root.name,
        "deterministic_algorithms": True, "cudnn_benchmark": False, "test_audio_accessed": False,
    }


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite R4-1 experiment: {EXPERIMENT_DIR}")
    set_seed(SEED)
    audio_root = require_audio_root()
    split_rows, summary, matched_sources = load_and_verify(audio_root)
    train_rows, validation_rows = split_rows["train"], split_rows["validation"]
    train_counts = Counter(row["_target"] for row in train_rows)
    weights = [len(train_rows) / (2.0 * train_counts[index]) for index in (0, 1)]
    expected_weights = (0.5140030229851109, 18.35328784119107)
    if any(not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0.0) for actual, expected in zip(weights, expected_weights)):
        raise RuntimeError(f"Class weights changed: {weights}")
    preflight = preflight_audio(train_rows + validation_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic shape/forward preflight before committing experiment artifacts.
    extractor = FixedLogMel().to(device).eval()
    with torch.inference_mode():
        probe_features = extractor(torch.zeros(2, WINDOW_SAMPLES, device=device))
        probe_model = ControlledDeletionModel().to(device).eval()
        probe_logits = probe_model(probe_features, torch.tensor([0, 39], device=device), mode="full")
    if tuple(probe_features.shape[1:]) != EXPECTED_FEATURE_SHAPE or tuple(probe_logits.shape) != (2, 2):
        raise RuntimeError(f"Shape preflight failed: features={tuple(probe_features.shape)}, logits={tuple(probe_logits.shape)}")
    del extractor, probe_features, probe_model, probe_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    EXPERIMENT_DIR.mkdir(parents=True)
    config = config_payload(summary, weights)
    write_json(EXPERIMENT_DIR / "config.json", config)
    write_json(EXPERIMENT_DIR / "environment.json", environment_payload(audio_root, device))
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "status": "PASS", "dataset": summary, "audio": preflight, "binary_class_counts_train": dict(train_counts),
        "class_weights": weights, "feature_shape": list(EXPECTED_FEATURE_SHAPE), "device": str(device),
        "training_started_after_matched_control_pass": True, "test_audio_accessed": False,
    })
    write_json(EXPERIMENT_DIR / "phone_vocab.json", {"tokens": {phone: index for index, phone in enumerate(PHONE_VOCAB)}, "size": 40})
    write_json(EXPERIMENT_DIR / "class_weights.json", {"formula": "N/(2*class_count)", "counts": dict(train_counts), "weights": weights})
    write_json(EXPERIMENT_DIR / "matched_control_verification.json", summary["matched_control"])

    run_started = time.perf_counter()
    train_dataset = materialize(train_rows, device, "R4-1 TRAIN")
    validation_dataset = materialize(validation_rows, device, "R4-1 VALIDATION")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    set_seed(SEED)
    model = ControlledDeletionModel().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    checkpoint_path = EXPERIMENT_DIR / CHECKPOINT_NAME
    records: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_deletion_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        seen = 0
        for features, targets, phone_ids, _ in tqdm(train_loader, desc=f"R4-1 epoch {epoch}/{EPOCHS}"):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            phone_ids = phone_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, phone_ids, mode="full")
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)
            seen += len(targets)
        validation, _, _ = evaluate(model, validation_dataset, validation_rows, criterion, device, "full")
        record = {"epoch": epoch, "train_loss": loss_sum / seen, "validation_full": validation, "epoch_seconds": time.perf_counter() - epoch_started}
        records.append(record)
        save_epochs(records)
        better = validation["macro_f1"] > best_macro_f1 + 1e-12 or (
            abs(validation["macro_f1"] - best_macro_f1) <= 1e-12 and validation["deletion_f1"] > best_deletion_f1 + 1e-12
        )
        if better:
            best_epoch, best_macro_f1, best_deletion_f1 = epoch, validation["macro_f1"], validation["deletion_f1"]
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "validation_macro_f1": best_macro_f1,
                "validation_deletion_f1": best_deletion_f1, "class_to_index": {"non_deletion": 0, "deletion": 1},
                "phone_to_id": PHONE_TO_ID, "config": config,
            }, checkpoint_path)
        print(
            f"epoch={epoch} train_loss={record['train_loss']:.6f} val_loss={validation['loss']:.6f} "
            f"acc={validation['accuracy']:.6f} bal={validation['balanced_accuracy']:.6f} mf1={validation['macro_f1']:.6f} "
            f"del_p={validation['deletion_precision']:.6f} del_r={validation['deletion_recall']:.6f} del_f1={validation['deletion_f1']:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics: dict[str, Any] = {}
    selected_predictions: dict[str, list[int]] = {}
    selected_probabilities: dict[str, list[float]] = {}
    for mode in ("full", "no_audio", "no_phone"):
        metrics, predictions, probabilities = evaluate(model, validation_dataset, validation_rows, criterion, device, mode)
        selected_metrics[mode] = metrics
        selected_predictions[mode] = predictions
        selected_probabilities[mode] = probabilities
        write_json(EXPERIMENT_DIR / f"selected_validation_{mode}.json", {
            "selected_epoch": best_epoch, "checkpoint_selection_used_this_mode": mode == "full", "metrics": metrics,
        })

    matched_positions = [i for i, row in enumerate(validation_rows) if row["_source_index"] in matched_sources]
    if len(matched_positions) != 1_564:
        raise RuntimeError("R4_1_MATCHED_CONTROL_FAIL after inference")
    matched_metrics = {
        mode: subset_metrics(validation_rows, selected_predictions[mode], matched_positions)
        for mode in selected_predictions
    }
    write_json(EXPERIMENT_DIR / "selected_matched_metrics.json", {
        "verification": summary["matched_control"], "metrics": matched_metrics,
        "selection_influence": False, "same_rows_for_all_modes": True,
    })

    speaker_diagnostics: dict[str, Any] = {}
    phone_diagnostics: dict[str, Any] = {}
    concentration_diagnostics: dict[str, Any] = {}
    for mode, predictions in selected_predictions.items():
        speakers, phones, concentration = mode_diagnostics(validation_rows, predictions)
        speaker_diagnostics[mode] = speakers
        phone_diagnostics[mode] = phones
        concentration_diagnostics[mode] = concentration
    write_json(EXPERIMENT_DIR / "per_speaker_metrics.json", speaker_diagnostics)
    write_json(EXPERIMENT_DIR / "phone_diagnostics.json", phone_diagnostics)
    write_json(EXPERIMENT_DIR / "concentration_diagnostics.json", concentration_diagnostics)

    full = selected_metrics["full"]
    full_failures = [f"{name}={full[name]:.6f} < {threshold:.6f}" for name, threshold in FULL_GATES.items() if full[name] < threshold]
    duration_gain = full["macro_f1"] - DURATION_BASELINE["macro_f1"]
    if duration_gain < 0.03:
        full_failures.append(f"macro_f1 gain over duration={duration_gain:.6f} < 0.030000")
    matched_full = matched_metrics["full"]
    matched_failures = [
        f"matched {name}={matched_full[name]:.6f} < {threshold:.6f}"
        for name, threshold in MATCHED_GATES.items() if matched_full[name] < threshold
    ]
    deltas = {
        "full_minus_no_audio": {
            key: matched_metrics["full"][key] - matched_metrics["no_audio"][key]
            for key in ("macro_f1", "balanced_accuracy", "deletion_f1")
        },
        "full_minus_no_phone": {
            key: matched_metrics["full"][key] - matched_metrics["no_phone"][key]
            for key in ("macro_f1", "balanced_accuracy", "deletion_f1")
        },
    }
    modality_failures = []
    if deltas["full_minus_no_audio"]["macro_f1"] < 0.05:
        modality_failures.append("FULL-NO_AUDIO matched Macro-F1 < 0.05")
    if deltas["full_minus_no_audio"]["deletion_f1"] < 0.05:
        modality_failures.append("FULL-NO_AUDIO matched deletion F1 < 0.05")
    if max(deltas["full_minus_no_phone"]["macro_f1"], deltas["full_minus_no_phone"]["deletion_f1"]) < 0.02:
        modality_failures.append("FULL does not beat NO_PHONE by 0.02 on matched Macro-F1 or deletion F1")
    write_json(EXPERIMENT_DIR / "modality_deltas.json", {"strict_matched": deltas, "failures": modality_failures})

    speaker_failures = []
    for speaker, metrics in speaker_diagnostics["full"].items():
        if metrics["deletion_support"] >= 30 and metrics["deletion_recall"] < 0.25:
            speaker_failures.append(f"{speaker} deletion recall={metrics['deletion_recall']:.6f} < 0.250000")

    if not full_failures and not matched_failures and not modality_failures and not speaker_failures:
        status = "R4_1_DELETION_SIGNAL_CONFIRMED"
    elif not full_failures and (matched_failures or modality_failures):
        status = "R4_1_DELETION_SHORTCUT_DOMINATED"
    elif (
        full["macro_f1"] >= 0.55
        or matched_full["balanced_accuracy"] >= 0.55
        or deltas["full_minus_no_audio"]["macro_f1"] >= 0.02
    ):
        status = "R4_1_DELETION_SIGNAL_WEAK"
    else:
        status = "R4_1_DELETION_FAIL"

    epochs = np.asarray([record["epoch"] for record in records[-5:]], dtype=np.float64)
    macro_values = np.asarray([record["validation_full"]["macro_f1"] for record in records[-5:]], dtype=np.float64)
    loss_values = np.asarray([record["validation_full"]["loss"] for record in records[-5:]], dtype=np.float64)
    macro_slope = float(np.polyfit(epochs, macro_values, 1)[0])
    loss_slope = float(np.polyfit(epochs, loss_values, 1)[0])
    budget_flag = bool(best_epoch == EPOCHS and macro_slope > 0.001 and loss_slope < -0.001)
    trend = {
        "selected_epoch": best_epoch, "last5_macro_f1_slope_per_epoch": macro_slope,
        "last5_validation_loss_slope_per_epoch": loss_slope,
        "flag": "TRAINING_BUDGET_POSSIBLY_LIMITING" if budget_flag else "NO_BUDGET_LIMIT_FLAG",
        "no_automatic_extension": True,
    }
    write_json(EXPERIMENT_DIR / "training_trend.json", trend)

    row_fields = [
        "row_index", "source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time",
        "expected_phone_canonical", "relation_origin", "binary_target", "full_prob_deletion", "full_prediction",
        "no_audio_prob_deletion", "no_audio_prediction", "no_phone_prob_deletion", "no_phone_prediction",
        "is_strict_matched_row",
    ]
    row_path = EXPERIMENT_DIR / "validation_row_predictions.csv"
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for index, row in enumerate(validation_rows):
            writer.writerow({
                "row_index": index, "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"], "utterance_id": row["utterance_id"],
                "start_time": row["start_time"], "end_time": row["end_time"],
                "expected_phone_canonical": row["expected_phone_canonical"], "relation_origin": row["relation"],
                "binary_target": row["_target"], "full_prob_deletion": selected_probabilities["full"][index],
                "full_prediction": selected_predictions["full"][index],
                "no_audio_prob_deletion": selected_probabilities["no_audio"][index],
                "no_audio_prediction": selected_predictions["no_audio"][index],
                "no_phone_prob_deletion": selected_probabilities["no_phone"][index],
                "no_phone_prediction": selected_predictions["no_phone"][index],
                "is_strict_matched_row": row["_source_index"] in matched_sources,
            })

    training_seconds = time.perf_counter() - run_started
    final = {
        "status": status, "selected_epoch": best_epoch, "full_gate_pass": not full_failures,
        "matched_gate_pass": not matched_failures, "modality_gate_pass": not modality_failures,
        "speaker_gate_pass": not speaker_failures, "failures": {
            "full": full_failures, "matched": matched_failures, "modality": modality_failures, "speaker": speaker_failures,
        },
        "duration_baseline_gain": duration_gain, "strict_matched_deltas": deltas, "training_trend": trend,
        "training_seconds_including_feature_materialization": training_seconds,
        "checkpoint": CHECKPOINT_NAME, "checkpoint_sha256": sha256_file(checkpoint_path),
        "row_predictions": row_path.name, "row_predictions_sha256": sha256_file(row_path),
        "training_performed": True, "r4_test_audio_accessed": False, "r4_test_inference": False,
        "r3_test_used_for_selection": False, "runtime_modified": False, "production_model": False,
    }
    write_json(EXPERIMENT_DIR / "model_metadata.json", {
        "name": "ControlledDeletionModel", "selected_epoch": best_epoch, "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": final["checkpoint_sha256"], "random_initialization": True,
        "pretrained_checkpoint_loaded": False, "feature_shape": list(EXPECTED_FEATURE_SHAPE),
    })
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    print(json.dumps({
        "status": status, "selected_epoch": best_epoch, "weights": weights, "feature_shape": EXPECTED_FEATURE_SHAPE,
        "device": str(device), "training_seconds": training_seconds, "full": full,
        "matched": matched_metrics, "deltas": deltas, "gates": {
            "full": not full_failures, "matched": not matched_failures,
            "modality": not modality_failures, "speaker": not speaker_failures,
        }, "trend": trend, "test_audio_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
