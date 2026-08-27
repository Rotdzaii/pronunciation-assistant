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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1a_observed_phone_seed42"
EXPECTED_DATASET_SHA256 = "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"

TRAIN_SPEAKERS = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
VALIDATION_SPEAKERS = ("ABA", "HKK", "HQTV", "LXC", "MBMPS", "SVBI")
TEST_SPEAKERS = ("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK")
SPLIT_SPEAKERS = {"train": TRAIN_SPEAKERS, "validation": VALIDATION_SPEAKERS, "test": TEST_SPEAKERS}
EXPECTED_SPLITS = {
    "train": {"rows": 57_559, "correct": 51_400, "substitution": 6_159},
    "validation": {"rows": 28_212, "correct": 25_303, "substitution": 2_909},
    "test": {"rows": 28_216, "correct": 24_923, "substitution": 3_293},
}

PHONE_VOCAB = (
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_ID = {phone: index for index, phone in enumerate(PHONE_VOCAB)}
ALL_LABELS = tuple(range(len(PHONE_VOCAB)))

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 0.50
WINDOW_SAMPLES = 8_000
N_MELS = 64
N_FFT = 2_048
HOP_LENGTH = 512
WIN_LENGTH = 2_048
EXPECTED_FEATURE_SHAPE = (1, 64, 16)

SEED = 42
BATCH_SIZE = 8
PREPROCESS_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 12
NUM_WORKERS = 0
DROPOUT = 0.2

OVERALL_THRESHOLDS = {"macro_f1": 0.35, "balanced_accuracy": 0.40, "accuracy": 0.50}
SUBSTITUTION_THRESHOLDS = {"accuracy": 0.25, "macro_f1": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked research-only R3-1A observed-phone baseline.")
    parser.add_argument("--preflight-only", action="store_true")
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
    if not root.is_absolute() or not root.is_dir() or root.name.lower() != "l2arctic_release_v5.0":
        raise RuntimeError(f"Invalid L2_ARCTIC_ROOT: {root}")
    return root


def resolve_audio_path(metadata_path: str, audio_root: Path) -> Path:
    parts = Path(metadata_path.replace("\\", "/")).parts
    indexes = [index for index, part in enumerate(parts) if part.lower() == "l2arctic_release_v5.0"]
    if not indexes:
        raise ValueError(f"Audio reference lacks corpus-root marker: {metadata_path}")
    return audio_root.joinpath(*parts[indexes[-1] + 1 :])


def load_and_validate_rows(audio_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    actual_hash = sha256_file(METADATA_CSV)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: expected {EXPECTED_DATASET_SHA256}, got {actual_hash}")
    with METADATA_CSV.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {
        "speaker_id", "audio_path", "utterance_id", "start_time", "end_time", "raw_label",
        "relation", "expected_phone_canonical", "observed_phone_canonical", "research_subset", "label_quality",
    }
    missing = required - set(source_rows[0] if source_rows else {})
    if missing:
        raise RuntimeError(f"Missing V4 fields: {sorted(missing)}")
    sets = [set(TRAIN_SPEAKERS), set(VALIDATION_SPEAKERS), set(TEST_SPEAKERS)]
    if len(set.union(*sets)) != 24 or any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Locked speaker split is not disjoint across exactly 24 speakers")

    eligible: list[dict[str, Any]] = []
    for source_index, source in enumerate(source_rows):
        if source["research_subset"] != "PHONE_IDENTIFICATION_ELIGIBLE":
            continue
        if source["label_quality"] != "clean" or source["relation"] not in {"correct", "substitution"}:
            raise RuntimeError(f"Invalid eligible semantics at CSV row {source_index + 2}")
        phone = source["observed_phone_canonical"]
        if phone not in PHONE_TO_ID:
            raise RuntimeError(f"Invalid observed target at CSV row {source_index + 2}: {phone!r}")
        start, end = float(source["start_time"]), float(source["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise RuntimeError(f"Invalid interval at CSV row {source_index + 2}: {start}, {end}")
        speaker = source["speaker_id"]
        if speaker not in set.union(*sets):
            raise RuntimeError(f"Speaker outside locked split: {speaker}")
        row = dict(source)
        row.update({
            "_source_index": source_index,
            "_start": start,
            "_end": end,
            "_target": PHONE_TO_ID[phone],
            "_edge_padded": False,
        })
        # TEST stays metadata-only: no audio path is resolved for its speakers.
        if speaker not in set(TEST_SPEAKERS):
            row["_audio_path"] = resolve_audio_path(source["audio_path"], audio_root)
        eligible.append(row)

    if len(eligible) != 113_987:
        raise RuntimeError(f"Eligible row count mismatch: {len(eligible)}")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, Any] = {}
    for split_name, speakers in SPLIT_SPEAKERS.items():
        speaker_set = set(speakers)
        rows = [row for row in eligible if row["speaker_id"] in speaker_set]
        origins = Counter(row["relation"] for row in rows)
        expected = EXPECTED_SPLITS[split_name]
        observed = {"rows": len(rows), "correct": origins["correct"], "substitution": origins["substitution"]}
        if observed != expected:
            raise RuntimeError(f"{split_name} count mismatch: expected {expected}, got {observed}")
        if {row["speaker_id"] for row in rows} != speaker_set:
            raise RuntimeError(f"{split_name} speaker coverage mismatch")
        split_rows[split_name] = rows
        support = Counter(row["observed_phone_canonical"] for row in rows)
        if set(support) != set(PHONE_VOCAB):
            raise RuntimeError(f"{split_name} does not cover all 40 targets: {set(PHONE_VOCAB) - set(support)}")
        split_summary[split_name] = {**observed, "speakers": list(speakers), "class_support": dict(sorted(support.items()))}

    identities: dict[tuple[Any, ...], str] = {}
    for row in eligible:
        identity = (row["speaker_id"], row["audio_path"], row["start_time"], row["end_time"])
        if identity in identities:
            raise RuntimeError(
                f"Duplicate eligible identity, previous={identities[identity]!r}, current={row['observed_phone_canonical']!r}: {identity}"
            )
        identities[identity] = row["observed_phone_canonical"]

    train_total = Counter(row["_target"] for row in split_rows["train"])
    val_total = Counter(row["_target"] for row in split_rows["validation"])
    train_sub = Counter(row["_target"] for row in split_rows["train"] if row["relation"] == "substitution")
    val_sub = Counter(row["_target"] for row in split_rows["validation"] if row["relation"] == "substitution")
    hard_supported = [index for index in ALL_LABELS if train_total[index] >= 100 and val_total[index] >= 50]
    substitution_supported = [index for index in ALL_LABELS if train_sub[index] >= 20 and val_sub[index] >= 10]
    if len(hard_supported) != 37 or len(substitution_supported) != 32:
        raise RuntimeError(
            f"Locked support sets changed: hard={len(hard_supported)} substitution={len(substitution_supported)}"
        )
    return split_rows, {
        "dataset_sha256": actual_hash,
        "source_rows": len(source_rows),
        "eligible_rows": len(eligible),
        "split_counts": split_summary,
        "hard_supported_classes": [PHONE_VOCAB[i] for i in hard_supported],
        "substitution_supported_classes": [PHONE_VOCAB[i] for i in substitution_supported],
        "test_audio_resolved": False,
        "test_audio_accessed": False,
        "duplicate_identities": 0,
        "conflicting_targets": 0,
    }


def preflight_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_path[row["_audio_path"]].append(row)
    errors: list[str] = []
    sample_rates: Counter[int] = Counter()
    padding = Counter()
    origin_totals = Counter(row["relation"] for row in rows)
    total_seconds = 0.0
    half_window = WINDOW_SECONDS / 2.0
    for path, path_rows in tqdm(sorted(by_path.items(), key=lambda item: str(item[0])), desc="TRAIN+VALIDATION audio preflight"):
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
        raise RuntimeError(f"Audio preflight failed with {len(errors)} error(s):\n" + "\n".join(errors[:20]))
    return {
        "scope": "TRAIN+VALIDATION only; TEST audio not resolved or accessed",
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


class SequentialWaveStore:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.audio: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        if self.path != path:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"Empty/non-finite decoded audio: {path}")
            self.path, self.audio = path, audio
        assert self.audio is not None
        return self.audio


def centered_window(audio: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    center_sample = int(round(((start_time + end_time) / 2.0) * SAMPLE_RATE))
    requested_start = center_sample - WINDOW_SAMPLES // 2
    requested_end = requested_start + WINDOW_SAMPLES
    source_start, source_end = max(0, requested_start), min(len(audio), requested_end)
    if source_end <= source_start:
        raise RuntimeError(f"Centered crop contains no audio: {start_time}, {end_time}")
    output = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    destination_start = source_start - requested_start
    output[destination_start : destination_start + source_end - source_start] = audio[source_start:source_end]
    if output.shape != (WINDOW_SAMPLES,):
        raise RuntimeError(f"Unexpected window shape: {output.shape}")
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
        return torch.from_numpy(window), torch.tensor(row["_target"], dtype=torch.long), index


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
        self.features, self.labels = features, labels

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
                raise RuntimeError(f"Unexpected {split_name} feature shape: {tuple(batch_features.shape[1:])}")
            next_cursor = cursor + len(indexes)
            features[cursor:next_cursor].copy_(batch_features)
            labels[cursor:next_cursor].copy_(batch_labels)
            cursor = next_cursor
    if cursor != len(rows) or not torch.isfinite(features).all():
        raise RuntimeError(f"Incomplete/non-finite features for {split_name}")
    return FeatureDataset(features, labels)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = feature_map.mean(dim=2).transpose(1, 2)
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1), weights


class SmallPronunciationCNNAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.attention = TemporalAttentionPooling(96)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(96, len(PHONE_VOCAB)))

    def forward(self, features: torch.Tensor, return_attention: bool = False):
        pooled, weights = self.attention(self.features(features))
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits


def metric_block(labels: list[int], predictions: list[int], metric_labels: list[int] | tuple[int, ...]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(metric_labels), zero_division=0
    )
    return {
        "rows": len(labels),
        "evaluated_classes": [PHONE_VOCAB[index] for index in metric_labels],
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            PHONE_VOCAB[index]: {
                "precision": float(precision[position]),
                "recall": float(recall[position]),
                "f1": float(f1[position]),
                "support": int(support[position]),
            }
            for position, index in enumerate(metric_labels)
        },
    }


def evaluate(
    model: SmallPronunciationCNNAttention,
    dataset: FeatureDataset,
    rows: list[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    substitution_supported: list[int],
    collect_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    top3_predictions: list[list[int]] = []
    probabilities: list[list[float]] = []
    attentions: list[list[float]] = []
    loss_sum = 0.0
    with torch.no_grad():
        for features, targets, _ in loader:
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if collect_details:
                logits, weights = model(features, return_attention=True)
                attentions.extend(weights.cpu().tolist())
            else:
                logits = model(features)
            loss_sum += criterion(logits, targets).item() * len(targets)
            probs = torch.softmax(logits, dim=1)
            top3 = torch.topk(probs, k=3, dim=1).indices
            labels.extend(targets.cpu().tolist())
            predictions.extend(probs.argmax(dim=1).cpu().tolist())
            top3_predictions.extend(top3.cpu().tolist())
            if collect_details:
                probabilities.extend(probs.cpu().tolist())
    overall = metric_block(labels, predictions, ALL_LABELS)
    overall["loss"] = loss_sum / len(labels)
    overall["top3_accuracy"] = float(np.mean([truth in top for truth, top in zip(labels, top3_predictions)]))
    overall["confusion_matrix"] = confusion_matrix(labels, predictions, labels=ALL_LABELS).astype(int).tolist()

    source_metrics: dict[str, Any] = {}
    for origin in ("correct", "substitution"):
        positions = [index for index, row in enumerate(rows) if row["relation"] == origin]
        local_labels = [labels[index] for index in positions]
        local_predictions = [predictions[index] for index in positions]
        metric_labels = substitution_supported if origin == "substitution" else sorted(set(local_labels))
        source_metrics[origin] = metric_block(local_labels, local_predictions, metric_labels)
        source_metrics[origin]["all_present_class_macro_f1"] = metric_block(
            local_labels, local_predictions, sorted(set(local_labels))
        )["macro_f1"]
        source_metrics[origin]["edge_padding"] = {}
        for padded in (False, True):
            subpositions = [index for index in positions if bool(rows[index]["_edge_padded"]) is padded]
            source_metrics[origin]["edge_padding"]["padded" if padded else "not_padded"] = {
                "rows": len(subpositions),
                "accuracy": float(np.mean([labels[index] == predictions[index] for index in subpositions])) if subpositions else None,
            }

    per_speaker: dict[str, Any] = {}
    for speaker in VALIDATION_SPEAKERS:
        positions = [index for index, row in enumerate(rows) if row["speaker_id"] == speaker]
        local_labels = [labels[index] for index in positions]
        local_predictions = [predictions[index] for index in positions]
        support = Counter(local_labels)
        local_supported = sorted(index for index, count in support.items() if count >= 10)
        per_speaker[speaker] = metric_block(local_labels, local_predictions, local_supported)
        per_speaker[speaker]["locally_supported_class_count"] = len(local_supported)

    metrics = {**overall, "relation_source": source_metrics, "per_speaker": per_speaker}
    details = {
        "labels": labels,
        "predictions": predictions,
        "top3_predictions": top3_predictions,
        "probabilities": probabilities,
        "attentions": attentions,
    }
    return metrics, details


def config_payload(summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    return {
        "experiment": "R3-1A 40-class observed-phone acoustic baseline",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED"],
        "dataset": str(METADATA_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": summary["dataset_sha256"],
        "eligible_subset": "PHONE_IDENTIFICATION_ELIGIBLE",
        "target": "observed_phone_canonical",
        "class_to_index": PHONE_TO_ID,
        "excluded": ["deletion", "addition", "unresolved", "non_speech"],
        "speaker_split": {name: list(speakers) for name, speakers in SPLIT_SPEAKERS.items()},
        "split_counts": summary["split_counts"],
        "model_input": ["audio-derived log-mel only"],
        "forbidden_model_features": [
            "expected_phone", "relation", "correct/substitution origin", "raw_label", "observed_phone except target index",
            "speaker_id", "L1", "gender", "filename/path tokens", "utterance text/id", "duration", "start_time/end_time",
        ],
        "audio": {
            "sample_rate": SAMPLE_RATE, "mono": True, "crop_seconds": WINDOW_SECONDS,
            "crop": "center=(start_time+end_time)/2; center +/- 0.25 s", "samples": WINDOW_SAMPLES,
            "boundary_policy": "clip available audio; zero-pad only missing side",
        },
        "log_mel": {
            "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH,
            "window": "hann", "center": True, "pad_mode": "constant", "power": 2.0,
            "mel_scale": "slaney", "norm": "slaney", "power_to_db_ref": "per-sample max",
            "top_db": 80.0, "feature_shape": list(EXPECTED_FEATURE_SHAPE),
        },
        "model": {
            "name": "SmallPronunciationCNNAttention", "channels": [16, 32, 64, 96],
            "pooling": "temporal attention", "head": ["Dropout(0.2)", "Linear(96,40)"],
            "initialization": "random", "pretrained_checkpoint": None,
        },
        "loss": "class-weighted CrossEntropy",
        "class_weight_formula": "N_train / (40 * train_count[c])",
        "class_weights": {PHONE_VOCAB[i]: class_weights[i] for i in ALL_LABELS},
        "sampler": "none", "augmentation": "none", "seed": SEED, "batch_size": BATCH_SIZE,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": 0.0, "epochs": EPOCHS,
        "prediction": "argmax", "checkpoint_selection": "highest validation Macro-F1; tie substitution-origin top-1; tie earlier epoch",
        "overall_gate": OVERALL_THRESHOLDS, "substitution_gate": SUBSTITUTION_THRESHOLDS,
        "class_coverage_gate": {"hard_supported_recall_at_least_0.10": "32/37", "validation_support_at_least_200_zero_recall": 0},
        "speaker_gate": {"each_macro_f1": 0.25, "median_macro_f1": 0.35},
    }


def environment_payload(audio_root: Path, device: torch.device, dataset_hash: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "pytorch": torch.__version__, "torchaudio": torchaudio.__version__,
        "librosa": librosa.__version__, "numpy": np.__version__, "device": str(device),
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_sha256": dataset_hash, "git_commit": git_commit(), "l2_arctic_root": str(audio_root),
        "l2_arctic_root_identity": audio_root.name, "deterministic_algorithms": True,
        "cudnn_benchmark": False, "num_workers": NUM_WORKERS, "test_audio_accessed": False,
    }


def save_epoch_outputs(records: list[dict[str, Any]]) -> None:
    write_json(EXPERIMENT_DIR / "epoch_metrics.json", records)
    fields = [
        "epoch", "train_loss", "validation_loss", "top1_accuracy", "macro_f1", "balanced_accuracy",
        "macro_precision", "macro_recall", "top3_accuracy", "correct_origin_top1", "correct_origin_macro_f1",
        "substitution_origin_top1", "substitution_origin_macro_f1", "epoch_seconds",
    ] + [f"speaker_{speaker}_macro_f1" for speaker in VALIDATION_SPEAKERS]
    with (EXPERIMENT_DIR / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["validation"]
            row = {
                "epoch": record["epoch"], "train_loss": record["train_loss"], "validation_loss": metrics["loss"],
                "top1_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"], "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"], "top3_accuracy": metrics["top3_accuracy"],
                "correct_origin_top1": metrics["relation_source"]["correct"]["accuracy"],
                "correct_origin_macro_f1": metrics["relation_source"]["correct"]["macro_f1"],
                "substitution_origin_top1": metrics["relation_source"]["substitution"]["accuracy"],
                "substitution_origin_macro_f1": metrics["relation_source"]["substitution"]["macro_f1"],
                "epoch_seconds": record["epoch_seconds"],
            }
            row.update({f"speaker_{speaker}_macro_f1": metrics["per_speaker"][speaker]["macro_f1"] for speaker in VALIDATION_SPEAKERS})
            writer.writerow(row)


def attention_report(attentions: list[list[float]]) -> dict[str, Any]:
    values = np.asarray(attentions, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    sums = values.sum(axis=1)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    normalized = entropy / math.log(values.shape[1]) if values.shape[1] > 1 else np.zeros(len(values))
    return {
        "shape": list(values.shape), "finite": finite, "nan_count": int(np.isnan(values).sum()),
        "row_sum_min": float(sums.min()), "row_sum_max": float(sums.max()),
        "mean_entropy": float(entropy.mean()), "mean_normalized_entropy": float(normalized.mean()),
        "mean_max_weight": float(values.max(axis=1).mean()),
        "degenerate_one_hot_rate": float(np.mean(values.max(axis=1) >= 0.999)),
        "interpretation_warning": "Attention concentration is a diagnostic, not an explanation.",
    }


def downstream_binary(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    truth = [0 if row["relation"] == "correct" else 1 for row in rows]
    predicted = [0 if PHONE_VOCAB[pred] == row["expected_phone_canonical"] else 1 for row, pred in zip(rows, predictions)]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=[0, 1], zero_division=0)
    return {
        "rows": len(rows), "labels": {"correct": 0, "substitution": 1},
        "macro_f1": float(np.mean(f1)), "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist(),
        "correct": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
        "substitution": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        "diagnostic_only": True, "used_for_training_selection_or_gate": False,
    }


def gate_report(metrics: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    overall_failures = [
        f"{name}={metrics[name]:.6f} < {threshold:.6f}"
        for name, threshold in OVERALL_THRESHOLDS.items() if metrics[name] < threshold
    ]
    substitution = metrics["relation_source"]["substitution"]
    overall_failures.extend(
        f"substitution_origin_{name}={substitution[name]:.6f} < {threshold:.6f}"
        for name, threshold in SUBSTITUTION_THRESHOLDS.items() if substitution[name] < threshold
    )
    hard = summary["hard_supported_classes"]
    recalls = metrics["per_class"]
    passing_hard = [phone for phone in hard if recalls[phone]["recall"] >= 0.10]
    high_support_zero = [phone for phone, item in recalls.items() if item["support"] >= 200 and item["recall"] == 0.0]
    class_failures = []
    if len(passing_hard) < 32:
        class_failures.append(f"hard-supported recall>=0.10: {len(passing_hard)}/37 < 32/37")
    if high_support_zero:
        class_failures.append(f"validation support>=200 with zero recall: {high_support_zero}")
    speaker_values = {speaker: metrics["per_speaker"][speaker]["macro_f1"] for speaker in VALIDATION_SPEAKERS}
    median_speaker = float(np.median(list(speaker_values.values())))
    speaker_failures = [f"{speaker} macro_f1={value:.6f} < 0.250000" for speaker, value in speaker_values.items() if value < 0.25]
    if median_speaker < 0.35:
        speaker_failures.append(f"median speaker macro_f1={median_speaker:.6f} < 0.350000")
    return {
        "overall_gate": {"pass": not overall_failures, "failures": overall_failures},
        "class_coverage_gate": {
            "pass": not class_failures, "failures": class_failures, "hard_supported_count": len(hard),
            "recall_at_least_0.10_count": len(passing_hard), "passing_classes": passing_hard,
            "failing_classes": [phone for phone in hard if phone not in passing_hard],
            "high_support_zero_recall": high_support_zero,
        },
        "speaker_gate": {"pass": not speaker_failures, "failures": speaker_failures, "median_macro_f1": median_speaker},
        "all_pass": not (overall_failures or class_failures or speaker_failures),
    }


def save_selected_outputs(
    metrics: dict[str, Any], details: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any],
    selected_epoch: int, training_seconds: float, checkpoint_path: Path,
) -> dict[str, Any]:
    report = {
        "selected_epoch": selected_epoch, "selection_metric": "validation Macro-F1",
        "tie_break": "higher substitution-origin top-1, then earlier epoch", "metrics": metrics,
    }
    write_json(EXPERIMENT_DIR / "selected_validation_report.json", report)
    write_json(EXPERIMENT_DIR / "per_class_metrics.json", metrics["per_class"])
    write_json(EXPERIMENT_DIR / "confusion_matrix.json", {"labels": list(PHONE_VOCAB), "matrix": metrics["confusion_matrix"]})
    with (EXPERIMENT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *PHONE_VOCAB])
        for phone, matrix_row in zip(PHONE_VOCAB, metrics["confusion_matrix"]):
            writer.writerow([phone, *matrix_row])
    write_json(EXPERIMENT_DIR / "per_speaker_metrics.json", metrics["per_speaker"])
    write_json(EXPERIMENT_DIR / "relation_source_metrics.json", metrics["relation_source"])
    write_json(EXPERIMENT_DIR / "attention_diagnostic.json", attention_report(details["attentions"]))
    downstream = downstream_binary(rows, details["predictions"])
    write_json(EXPERIMENT_DIR / "downstream_binary_diagnostic.json", downstream)

    fields = [
        "row_index", "source_csv_row", "speaker_id", "audio_path", "utterance_id", "start_time", "end_time",
        "expected_phone_canonical", "observed_phone_canonical", "relation_origin", "edge_padded",
        "predicted_observed_phone", "top1_probability", "top3_phones", "top3_probabilities",
        "correct_prediction", "downstream_predicted_relation",
    ]
    with (EXPERIMENT_DIR / "validation_row_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (row, predicted, top3, probs) in enumerate(
            zip(rows, details["predictions"], details["top3_predictions"], details["probabilities"])
        ):
            predicted_phone = PHONE_VOCAB[predicted]
            writer.writerow({
                "row_index": index, "source_csv_row": row["_source_index"] + 2, "speaker_id": row["speaker_id"],
                "audio_path": row["audio_path"], "utterance_id": row["utterance_id"], "start_time": row["start_time"],
                "end_time": row["end_time"], "expected_phone_canonical": row["expected_phone_canonical"],
                "observed_phone_canonical": row["observed_phone_canonical"], "relation_origin": row["relation"],
                "edge_padded": int(row["_edge_padded"]), "predicted_observed_phone": predicted_phone,
                "top1_probability": probs[predicted], "top3_phones": " ".join(PHONE_VOCAB[item] for item in top3),
                "top3_probabilities": " ".join(f"{probs[item]:.9f}" for item in top3),
                "correct_prediction": int(predicted == row["_target"]),
                "downstream_predicted_relation": "correct" if predicted_phone == row["expected_phone_canonical"] else "substitution",
            })

    gates = gate_report(metrics, summary)
    status = "R3_1A_PASS_VALIDATION" if gates["all_pass"] else "R3_1A_VALIDATION_FAIL"
    final = {
        "status": status, "test_opened": False, "test_eligible": bool(gates["all_pass"]),
        "selected_epoch": selected_epoch, "training_and_validation_seconds": training_seconds,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"), "gates": gates,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    return {"downstream": downstream, "gates": gates, "final": final}


def main() -> int:
    args = parse_args()
    set_seed(SEED)
    audio_root = require_audio_root()
    split_rows, summary = load_and_validate_rows(audio_root)
    print(json.dumps(summary, indent=2))
    train_validation_rows = split_rows["train"] + split_rows["validation"]
    audio_preflight = preflight_audio(train_validation_rows)
    print("Audio preflight:", json.dumps(audio_preflight, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_row = split_rows["train"][0]
    probe_audio, _ = librosa.load(probe_row["_audio_path"], sr=SAMPLE_RATE, mono=True)
    probe_window = centered_window(np.asarray(probe_audio, dtype=np.float32), probe_row["_start"], probe_row["_end"])
    with torch.no_grad():
        probe_feature = FixedLogMel().to(device)(torch.from_numpy(probe_window).unsqueeze(0).to(device))
    if probe_window.shape != (WINDOW_SAMPLES,) or tuple(probe_feature.shape[1:]) != EXPECTED_FEATURE_SHAPE:
        raise RuntimeError(f"Feature preflight failed: waveform={probe_window.shape}, mel={tuple(probe_feature.shape)}")

    train_counts = Counter(row["_target"] for row in split_rows["train"])
    class_weights = [len(split_rows["train"]) / (len(PHONE_VOCAB) * train_counts[index]) for index in ALL_LABELS]
    if not all(math.isfinite(weight) and weight > 0 for weight in class_weights):
        raise RuntimeError("Class weights must all be finite and positive")
    print("Class weights:", json.dumps({PHONE_VOCAB[i]: class_weights[i] for i in ALL_LABELS}, indent=2))
    print(f"Feature-shape preflight: PASS waveform={probe_window.shape}, feature={tuple(probe_feature.shape)}")
    print(f"Device: {device}")
    if args.preflight_only:
        print("R3-1A preflight PASS; TEST audio not accessed; no training/artifacts.")
        return 0

    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")
    EXPERIMENT_DIR.mkdir(parents=True)
    config = config_payload(summary, class_weights)
    write_json(EXPERIMENT_DIR / "config.json", config)
    write_json(EXPERIMENT_DIR / "environment.json", environment_payload(audio_root, device, summary["dataset_sha256"]))
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "dataset": summary, "audio": audio_preflight, "feature_shape": list(probe_feature.shape),
        "status": "PASS", "test_audio_accessed": False,
    })
    write_json(EXPERIMENT_DIR / "phone_vocab.json", {"class_to_index": PHONE_TO_ID, "index_to_class": list(PHONE_VOCAB)})
    write_json(EXPERIMENT_DIR / "class_weights.json", {
        "formula": "N_train / (40 * train_count[c])", "train_rows": len(split_rows["train"]),
        "weights": {PHONE_VOCAB[i]: class_weights[i] for i in ALL_LABELS},
        "train_support": {PHONE_VOCAB[i]: train_counts[i] for i in ALL_LABELS},
    })

    run_started = time.perf_counter()
    train_dataset = materialize_audio_features(split_rows["train"], device, "train")
    validation_dataset = materialize_audio_features(split_rows["validation"], device, "validation")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = SmallPronunciationCNNAttention().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    substitution_supported = [PHONE_TO_ID[phone] for phone in summary["substitution_supported_classes"]]
    checkpoint_path = EXPERIMENT_DIR / "R3_1A_observed_phone_40class_seed42_best_validation_macro_f1.pt"
    records: list[dict[str, Any]] = []
    best_epoch, best_macro_f1, best_substitution_accuracy = 0, -1.0, -1.0

    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for features, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train"):
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_seen += len(targets)
        validation_metrics, _ = evaluate(
            model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported
        )
        record = {
            "epoch": epoch, "train_loss": train_loss_sum / train_seen, "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        records.append(record)
        save_epoch_outputs(records)
        macro_f1 = validation_metrics["macro_f1"]
        substitution_accuracy = validation_metrics["relation_source"]["substitution"]["accuracy"]
        better = macro_f1 > best_macro_f1 + 1e-12 or (
            abs(macro_f1 - best_macro_f1) <= 1e-12 and substitution_accuracy > best_substitution_accuracy + 1e-12
        )
        if better:
            best_epoch, best_macro_f1, best_substitution_accuracy = epoch, macro_f1, substitution_accuracy
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "validation_macro_f1": macro_f1,
                "validation_substitution_origin_accuracy": substitution_accuracy, "class_to_index": PHONE_TO_ID,
                "config": config,
            }, checkpoint_path)
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} top1={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced={validation_metrics['balanced_accuracy']:.6f} "
            f"top3={validation_metrics['top3_accuracy']:.6f} correct_top1={validation_metrics['relation_source']['correct']['accuracy']:.6f} "
            f"sub_top1={substitution_accuracy:.6f} sub_macro_f1={validation_metrics['relation_source']['substitution']['macro_f1']:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics, selected_details = evaluate(
        model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported, collect_details=True
    )
    elapsed = time.perf_counter() - run_started
    outputs = save_selected_outputs(
        selected_metrics, selected_details, split_rows["validation"], summary, best_epoch, elapsed, checkpoint_path
    )
    write_json(EXPERIMENT_DIR / "model_metadata.json", {
        "name": "SmallPronunciationCNNAttention", "task": "R3-1A observed-phone 40-class recognition",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED"], "random_initialization": True,
        "loaded_checkpoint_for_initialization": False, "selected_epoch": best_epoch, "checkpoint": checkpoint_path.name,
        "validation_macro_f1": selected_metrics["macro_f1"], "class_to_index": PHONE_TO_ID,
        "test_opened": False, "test_eligible": outputs["final"]["test_eligible"],
    })
    print(json.dumps(outputs["final"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
