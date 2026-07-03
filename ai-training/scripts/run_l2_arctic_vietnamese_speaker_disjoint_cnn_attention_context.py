from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import random
import subprocess

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv")
MODEL_DIR = Path("ai-training/models")
EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

RESULTS_JSON = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_results.json"
PER_FOLD_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_per_fold.csv"
SUMMARY_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_summary.csv"
PER_CLASS_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_per_class.csv"
CONFUSION_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_confusion_matrices.csv"
MISCLASSIFIED_CSV = EVALUATION_DIR / "vietnamese_speaker_disjoint_context_misclassified_examples.csv"

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42
DROPOUT = 0.2

LABEL_ORDER = ["addition", "deletion", "substitution"]
VIETNAMESE_SPEAKERS = ["HQTV", "PNV", "THV", "TLV"]
CONTEXT_MODES = {
    "original_segment": 0.0,
    "context_0_05": 0.05,
    "context_0_10": 0.10,
    "context_0_15": 0.15,
}
DEFAULT_CONTEXT_MODES = ["context_0_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vietnamese speaker-disjoint CNN Attention context experiment.")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per fold. Default: 12.")
    parser.add_argument("--max-folds", type=int, default=None, help="Optional maximum number of folds to run.")
    parser.add_argument(
        "--context-modes",
        nargs="+",
        default=DEFAULT_CONTEXT_MODES,
        choices=sorted(CONTEXT_MODES),
        help="Context modes to run. Default: context_0_10.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print fold composition without training.")
    parser.add_argument(
        "--write-from-json",
        action="store_true",
        help="Regenerate CSV outputs from the existing results JSON without retraining.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal", "focal_weighted_sampler"],
        help="Loss function. cross_entropy uses class-weighted CE + WeightedRandomSampler. focal uses FocalLoss(gamma=2) with standard shuffle. focal_weighted_sampler uses FocalLoss(gamma=2) + WeightedRandomSampler. Default: cross_entropy.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (used when --loss-type is focal or focal_weighted_sampler). Default: 2.0.",
    )
    parser.add_argument(
        "--augment-minority",
        action="store_true",
        help="Enable SpecAugment-based data augmentation for the addition class. Default: False.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output file names. Empty = auto-derive from --loss-type. Default: ''.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_telemetry(stage: str) -> None:
    print()
    print(f"GPU telemetry - {stage}")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print(f"memory allocated MB: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"memory reserved MB: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f}")
        print(f"max memory allocated MB: {torch.cuda.max_memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"max memory reserved MB: {torch.cuda.max_memory_reserved(0) / 1024 / 1024:.2f}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi utilization: unavailable ({exc})")
        return

    output = result.stdout.strip()
    print("nvidia-smi utilization:", output if output else "no GPU rows returned")


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_segment(row: dict[str, str]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except (KeyError, ValueError):
        return False

    return (
        row.get("error_type") in LABEL_ORDER
        and Path(row["audio_path"]).exists()
        and start_time >= 0.0
        and end_time > start_time
    )


def spec_augment(mel_spec: np.ndarray, time_mask_width: int = 0.1, freq_mask_width: int = 0.15) -> np.ndarray:
    """Apply SpecAugment to a mel-spectrogram (time and frequency masking)."""
    aug = mel_spec.copy()
    n_mels, n_frames = aug.shape

    # Time masking: mask up to time_mask_width% of frames
    if time_mask_width > 0:
        max_t = max(1, int(n_frames * time_mask_width))
        t_mask_width = np.random.randint(1, max_t + 1)
        t_mask_start = np.random.randint(0, max(1, n_frames - t_mask_width + 1))
        aug[:, t_mask_start:t_mask_start + t_mask_width] = 0.0

    # Frequency masking: mask up to freq_mask_width% of mel bins
    if freq_mask_width > 0:
        max_f = max(1, int(n_mels * freq_mask_width))
        f_mask_width = np.random.randint(1, max_f + 1)
        f_mask_start = np.random.randint(0, max(1, n_mels - f_mask_width + 1))
        aug[f_mask_start:f_mask_start + f_mask_width, :] = 0.0

    return aug


def pitch_shift_audio(audio: np.ndarray, n_steps: int = None) -> np.ndarray:
    """Apply pitch shift to waveform. n_steps in [-2, -1, 0, 1, 2] semitones."""
    if n_steps is None:
        n_steps = np.random.randint(-2, 3)
    if n_steps == 0:
        return audio
    try:
        return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
    except Exception:
        return audio


def time_stretch_audio(audio: np.ndarray, rate: float = None) -> np.ndarray:
    """Apply time stretching to waveform. rate in [0.9, 1.1]."""
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    if abs(rate - 1.0) < 0.01:
        return audio
    try:
        return librosa.effects.time_stretch(audio, rate=rate)
    except Exception:
        return audio


def normalize_audio_length(audio: np.ndarray, target_length: int = MAX_LENGTH) -> np.ndarray:
    """Pad or truncate audio to fixed length."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    else:
        return audio


def load_segment(row: dict[str, str], context_seconds: float) -> np.ndarray:
    start_time = float(row["start_time"])
    end_time = float(row["end_time"])

    if context_seconds <= 0.0:
        audio, _ = librosa.load(
            row["audio_path"],
            sr=SAMPLE_RATE,
            mono=True,
            offset=start_time,
            duration=end_time - start_time,
        )
    else:
        full_audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        audio_duration = len(full_audio) / SAMPLE_RATE
        crop_start = max(0.0, start_time - context_seconds)
        crop_end = min(audio_duration, end_time + context_seconds)
        start_sample = max(0, int(crop_start * SAMPLE_RATE))
        end_sample = min(len(full_audio), int(crop_end * SAMPLE_RATE))
        audio = full_audio[start_sample:end_sample]

    if len(audio) == 0:
        audio = np.zeros(MAX_LENGTH, dtype=np.float32)
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))
    return audio.astype(np.float32, copy=False)


class ErrorSegmentDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_to_index: dict[str, int], context_seconds: float):
        self.rows = rows
        self.label_to_index = label_to_index
        self.context_seconds = context_seconds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio = load_segment(row, self.context_seconds)

        # Apply waveform-level augmentations if needed
        augment_type = row.get("_augment_type")
        if augment_type == "pitch_shift":
            audio = pitch_shift_audio(audio)
        elif augment_type == "time_stretch":
            audio = time_stretch_audio(audio)
            audio = normalize_audio_length(audio)
        elif augment_type == "combo":
            audio = pitch_shift_audio(audio)

        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Apply spectrogram-level augmentations if needed
        if augment_type == "spec_augment":
            log_mel = spec_augment(log_mel, time_mask_width=0.1, freq_mask_width=0.15)
        elif augment_type == "combo":
            log_mel = spec_augment(log_mel, time_mask_width=0.1, freq_mask_width=0.15)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.label_to_index[row["error_type"]], dtype=torch.long)
        return feature, target, index


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
    def __init__(self, num_classes: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(x)
        pooled, _ = self.attention(feature_map)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        pt = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (-((1.0 - pt) ** self.gamma) * log_pt).mean()


def build_sampler(rows: list[dict[str, str]]) -> WeightedRandomSampler:
    counts = Counter(row["error_type"] for row in rows)
    sample_weights = [1.0 / counts[row["error_type"]] for row in rows]
    return WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), True)


def calculate_metrics(labels: list[int], predictions: list[int], index_to_label: dict[int, str]) -> dict:
    label_indexes = list(range(len(index_to_label)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=label_indexes, zero_division=0
    )
    per_class = {}

    for index in label_indexes:
        predicted_count = sum(1 for prediction in predictions if prediction == index)
        correct = sum(1 for label, prediction in zip(labels, predictions) if label == index and prediction == index)
        per_class[index_to_label[index]] = {
            "class_id": index,
            "support": int(support[index]),
            "predicted_count": int(predicted_count),
            "correct": int(correct),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }

    total = len(labels)
    return {
        "total_samples": total,
        "correct": sum(1 for label, prediction in zip(labels, predictions) if label == prediction),
        "accuracy": float(accuracy_score(labels, predictions)) if total else 0.0,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if total else 0.0,
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)) if total else 0.0,
        "per_class": per_class,
    }


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels, _ in tqdm(loader, desc="Training", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def run_predictions(rows, model, label_to_index, device: torch.device, context_seconds: float) -> list[dict]:
    dataset = ErrorSegmentDataset(rows, label_to_index, context_seconds)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    predictions = []
    model.eval()

    with torch.no_grad():
        for features, targets, indexes in tqdm(loader, desc="Evaluating", leave=False):
            features = features.to(device, non_blocking=True)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1).cpu()
            batch_predictions = probs.argmax(dim=1).tolist()

            for target, prediction, row_index, probabilities in zip(
                targets.tolist(), batch_predictions, indexes.tolist(), probs.tolist()
            ):
                predictions.append(
                    {
                        "row": rows[row_index],
                        "label": target,
                        "prediction": prediction,
                        "probabilities": probabilities,
                    }
                )
    return predictions


def summarize_predictions(context_mode, fold, split_name, predictions, index_to_label) -> dict:
    labels = [item["label"] for item in predictions]
    predicted = [item["prediction"] for item in predictions]
    metrics = calculate_metrics(labels, predicted, index_to_label)
    confusion = defaultdict(Counter)
    misclassified_examples = []
    class_names = [index_to_label[index] for index in sorted(index_to_label)]

    for item in predictions:
        row = item["row"]
        actual = index_to_label[item["label"]]
        prediction = index_to_label[item["prediction"]]
        probabilities = item["probabilities"]
        confusion[actual][prediction] += 1

        if actual != prediction:
            misclassified_examples.append(
                {
                    "context_mode": context_mode,
                    "held_out_speaker": fold,
                    "split": split_name,
                    "speaker_id": row["speaker_id"],
                    "l1": row.get("l1", ""),
                    "utterance_id": row["utterance_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "label": row["label"],
                    "actual_error_type": actual,
                    "predicted_error_type": prediction,
                    "confidence": probabilities[item["prediction"]],
                    "audio_path": row["audio_path"],
                    "target_text": row["target_text"],
                }
            )

    return {
        "metrics": metrics,
        "confusion_matrix": {
            actual: {predicted_name: confusion[actual][predicted_name] for predicted_name in class_names}
            for actual in class_names
        },
        "misclassified_examples": misclassified_examples,
    }


def build_fold_rows(rows: list[dict[str, str]], held_out_speaker: str) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = [row for row in rows if row["speaker_id"] != held_out_speaker and row["split"] == "train"]
    val_rows = [row for row in rows if row["speaker_id"] != held_out_speaker and row["split"] == "val"]
    test_rows = [row for row in rows if row["speaker_id"] == held_out_speaker]
    return train_rows, val_rows, test_rows


def print_distribution(rows: list[dict], title: str) -> None:
    counts = Counter(row["error_type"] for row in rows)
    print(title)
    for label in LABEL_ORDER:
        print(f"- {label}: {counts[label]}")


def checkpoint_path(context_mode: str, held_out_speaker: str, suffix: str = "") -> Path:
    suffix_part = f"_{suffix}" if suffix else ""
    return MODEL_DIR / f"l2_arctic_cnn_attention_speaker_disjoint_context_{context_mode}_{held_out_speaker}{suffix_part}.pt"


def training_config(epochs: int, context_modes: list[str], loss_type: str = "cross_entropy") -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "random_seed": RANDOM_SEED,
        "dropout": DROPOUT,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "max_seconds": MAX_SECONDS,
        "context_modes": context_modes,
        "context_seconds": {mode: CONTEXT_MODES[mode] for mode in context_modes},
        "sampler": "shuffle" if loss_type == "focal" else "inverse-frequency weighted random sampler, same as baseline speaker-disjoint run",
        "loss": "focal_loss_gamma2_no_alpha" if loss_type in ("focal", "focal_weighted_sampler") else "cross_entropy_class_weighted",
    }


def run_fold(context_mode, held_out_speaker, rows, epochs, label_to_index, index_to_label, device, output_suffix: str = "", loss_type: str = "cross_entropy", focal_gamma: float = 2.0, augment_minority: bool = False) -> dict:
    set_seed(RANDOM_SEED)
    context_seconds = CONTEXT_MODES[context_mode]
    train_rows, val_rows, test_rows = build_fold_rows(rows, held_out_speaker)

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(f"Fold {context_mode}/{held_out_speaker} has an empty train/val/test partition.")
    if any(row["speaker_id"] == held_out_speaker for row in train_rows + val_rows):
        raise RuntimeError(f"Held-out speaker leaked into train/val: {held_out_speaker}")

    print()
    print("=" * 72)
    print(f"Context mode: {context_mode} ({context_seconds:.2f}s each side)")
    print(f"Fold: hold out Vietnamese speaker {held_out_speaker}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print_distribution(train_rows, "Train error distribution:")
    print_distribution(val_rows, "Validation error distribution:")
    print_distribution(test_rows, "Held-out test error distribution:")

    # Apply diverse augmentations to addition class samples (training only)
    if augment_minority:
        addition_rows = [row for row in train_rows if row["error_type"] == "addition"]
        augmented_rows = []
        augment_types = ["pitch_shift", "time_stretch", "spec_augment", "combo"]
        for row in addition_rows:
            for aug_idx, aug_type in enumerate(augment_types):
                augmented_row = row.copy()
                augmented_row["_augment_type"] = aug_type
                augmented_rows.append(augmented_row)
        train_rows = train_rows + augmented_rows
        print(f"Augmented {len(addition_rows)} addition samples x4 = {len(augmented_rows)} augmented samples added")
        print(f"  Augmentation types: pitch_shift, time_stretch, spec_augment, combo")
        print_distribution(train_rows, "Train error distribution after augmentation:")

    if loss_type == "focal":
        train_loader = DataLoader(
            ErrorSegmentDataset(train_rows, label_to_index, context_seconds),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            ErrorSegmentDataset(train_rows, label_to_index, context_seconds),
            batch_size=BATCH_SIZE,
            sampler=build_sampler(train_rows),
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    model = SmallPronunciationCNNAttention(num_classes=len(LABEL_ORDER)).to(device)
    if loss_type in ("focal", "focal_weighted_sampler"):
        criterion = FocalLoss(gamma=focal_gamma).to(device)
    else:
        train_counts = Counter(row["error_type"] for row in train_rows)
        total_train = sum(train_counts.values())
        class_weights = torch.tensor(
            [total_train / (len(LABEL_ORDER) * train_counts[label]) for label in LABEL_ORDER],
            dtype=torch.float32,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    output_path = checkpoint_path(context_mode, held_out_speaker, output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_val_metrics = None

    for epoch in range(epochs):
        print()
        print(f"Context {context_mode} fold {held_out_speaker} epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_predictions = run_predictions(val_rows, model, label_to_index, device, context_seconds)
        val_result = summarize_predictions(context_mode, held_out_speaker, "val", val_predictions, index_to_label)
        val_metrics = val_result["metrics"]
        improved = val_metrics["macro_f1"] > best_val_macro_f1
        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"checkpoint_improved={improved}"
        )

        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "error_to_label": label_to_index,
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "max_seconds": MAX_SECONDS,
                    "max_length": MAX_LENGTH,
                    "context_mode": context_mode,
                    "context_seconds": context_seconds,
                    "task": "l2_arctic_vietnamese_speaker_disjoint_cnn_attention_context",
                    "held_out_speaker": held_out_speaker,
                    "metadata_csv": str(METADATA_CSV),
                    "training_strategy": (
                        "focal_loss_standard_sampler" if loss_type == "focal"
                        else "focal_loss_weighted_random_sampler" if loss_type == "focal_weighted_sampler"
                        else "weighted_random_sampler_with_context_window"
                    ),
                    "loss": "focal_loss_gamma2_no_alpha" if loss_type in ("focal", "focal_weighted_sampler") else "cross_entropy_class_weighted",
                    "pooling": "temporal_attention",
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                    "label_order": LABEL_ORDER,
                    "config": training_config(epochs, [context_mode], loss_type),
                    "note": "Confidence is classifier confidence, not pronunciation correctness.",
                },
                output_path,
            )
            print(f"Saved checkpoint: {output_path}")

    checkpoint = torch.load(output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_predictions = run_predictions(test_rows, model, label_to_index, device, context_seconds)
    test_result = summarize_predictions(context_mode, held_out_speaker, "test", test_predictions, index_to_label)

    print()
    print(f"Context {context_mode} fold {held_out_speaker} test metrics:")
    metrics = test_result["metrics"]
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}")
    for label in LABEL_ORDER:
        item = metrics["per_class"][label]
        print(f"- {label}: precision={item['precision']:.4f} recall={item['recall']:.4f} f1={item['f1']:.4f} support={item['support']}")

    return {
        "context_mode": context_mode,
        "context_seconds": context_seconds,
        "held_out_speaker": held_out_speaker,
        "checkpoint_path": str(output_path),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_distribution": dict(Counter(row["error_type"] for row in train_rows)),
        "validation_distribution": dict(Counter(row["error_type"] for row in val_rows)),
        "test_distribution": dict(Counter(row["error_type"] for row in test_rows)),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "validation_metrics": best_val_metrics,
        "test": test_result,
    }


def nested_metric(fold: dict, metric_path: list[str]) -> float:
    current = fold
    for key in metric_path:
        current = current[key]
    return float(current)


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def build_summary(folds: list[dict], context_modes: list[str]) -> dict:
    summary_rows = []
    for mode in context_modes:
        mode_folds = [fold for fold in folds if fold["context_mode"] == mode]
        if not mode_folds:
            continue

        row = {
            "context_mode": mode,
            "context_seconds": CONTEXT_MODES[mode],
            "fold_count": len(mode_folds),
        }
        for output_name, path in [
            ("accuracy", ["test", "metrics", "accuracy"]),
            ("macro_f1", ["test", "metrics", "macro_f1"]),
            ("weighted_f1", ["test", "metrics", "weighted_f1"]),
        ]:
            mean_value, std_value = mean_std([nested_metric(fold, path) for fold in mode_folds])
            row[f"mean_{output_name}"] = mean_value
            row[f"std_{output_name}"] = std_value

        for label in LABEL_ORDER:
            mean_value, std_value = mean_std(
                [nested_metric(fold, ["test", "metrics", "per_class", label, "f1"]) for fold in mode_folds]
            )
            row[f"mean_{label}_f1"] = mean_value
            row[f"std_{label}_f1"] = std_value

        ranked = sorted(mode_folds, key=lambda fold: fold["test"]["metrics"]["macro_f1"], reverse=True)
        row["best_fold"] = ranked[0]["held_out_speaker"]
        row["best_fold_macro_f1"] = ranked[0]["test"]["metrics"]["macro_f1"]
        row["worst_fold"] = ranked[-1]["held_out_speaker"]
        row["worst_fold_macro_f1"] = ranked[-1]["test"]["metrics"]["macro_f1"]
        summary_rows.append(row)

    best_by_addition = max(summary_rows, key=lambda row: row["mean_addition_f1"]) if summary_rows else None
    best_by_macro = max(summary_rows, key=lambda row: row["mean_macro_f1"]) if summary_rows else None
    return {
        "by_context_mode": summary_rows,
        "best_context_mode_by_addition_f1": best_by_addition["context_mode"] if best_by_addition else None,
        "best_context_mode_by_macro_f1": best_by_macro["context_mode"] if best_by_macro else None,
    }


def output_paths(suffix: str = "") -> tuple[Path, Path, Path, Path, Path, Path]:
    s = f"_{suffix}" if suffix else ""
    base = "vietnamese_speaker_disjoint_context"
    return (
        EVALUATION_DIR / f"{base}{s}_results.json",
        EVALUATION_DIR / f"{base}{s}_per_fold.csv",
        EVALUATION_DIR / f"{base}{s}_summary.csv",
        EVALUATION_DIR / f"{base}{s}_per_class.csv",
        EVALUATION_DIR / f"{base}{s}_confusion_matrices.csv",
        EVALUATION_DIR / f"{base}{s}_misclassified_examples.csv",
    )


def write_outputs(results: dict, suffix: str = "") -> None:
    results_json, per_fold_csv, summary_csv, per_class_csv, confusion_csv, misclassified_csv = output_paths(suffix)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with per_fold_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "context_mode",
            "context_seconds",
            "held_out_speaker",
            "train_rows",
            "validation_rows",
            "test_rows",
            "best_epoch",
            "best_val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            *[f"test_{label}_f1" for label in LABEL_ORDER],
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            metrics = fold["test"]["metrics"]
            row = {
                "context_mode": fold["context_mode"],
                "context_seconds": fold["context_seconds"],
                "held_out_speaker": fold["held_out_speaker"],
                "train_rows": fold["train_rows"],
                "validation_rows": fold["validation_rows"],
                "test_rows": fold["test_rows"],
                "best_epoch": fold["best_epoch"],
                "best_val_macro_f1": fold["best_val_macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
            }
            for label in LABEL_ORDER:
                row[f"test_{label}_f1"] = metrics["per_class"][label]["f1"]
            writer.writerow(row)

    summary_rows = results["summary"]["by_context_mode"]
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["context_mode"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with per_class_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["context_mode", "held_out_speaker", "error_type", "class_id", "support", "predicted_count", "correct", "precision", "recall", "f1"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            for label, metrics in fold["test"]["metrics"]["per_class"].items():
                writer.writerow({"context_mode": fold["context_mode"], "held_out_speaker": fold["held_out_speaker"], "error_type": label, **metrics})

    with confusion_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["context_mode", "held_out_speaker", "actual_error_type", *LABEL_ORDER])
        writer.writeheader()
        for fold in results["folds"]:
            for actual in LABEL_ORDER:
                row = {"context_mode": fold["context_mode"], "held_out_speaker": fold["held_out_speaker"], "actual_error_type": actual}
                row.update(fold["test"]["confusion_matrix"][actual])
                writer.writerow(row)

    with misclassified_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "context_mode",
            "held_out_speaker",
            "split",
            "speaker_id",
            "l1",
            "utterance_id",
            "start_time",
            "end_time",
            "label",
            "actual_error_type",
            "predicted_error_type",
            "confidence",
            "audio_path",
            "target_text",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            writer.writerows(fold["test"]["misclassified_examples"][:500])


def print_dry_run(rows: list[dict], speakers: list[str], context_modes: list[str]) -> None:
    print("Dry run: no training will be performed.")
    print("Context modes:", {mode: CONTEXT_MODES[mode] for mode in context_modes})
    print("Protocol:")
    print("- Test: all rows from one held-out Vietnamese speaker.")
    print("- Train: original train split rows from every other speaker, including non-Vietnamese L1 groups.")
    print("- Validation: original val split rows from every other speaker.")
    print("- Held-out speaker is excluded from both train and validation.")
    for speaker in speakers:
        train_rows, val_rows, test_rows = build_fold_rows(rows, speaker)
        print()
        print(f"Fold {speaker}")
        print(f"train_rows={len(train_rows)} val_rows={len(val_rows)} test_rows={len(test_rows)}")
        print_distribution(test_rows, "held-out test distribution:")


def main() -> None:
    args = parse_args()
    output_suffix = args.output_suffix or (
        "focal_loss" if args.loss_type == "focal"
        else "focal_weighted_sampler" if args.loss_type == "focal_weighted_sampler"
        else "weighted_loss"
    )
    if args.focal_gamma != 2.0 and args.loss_type in ("focal", "focal_weighted_sampler"):
        output_suffix = f"{output_suffix}_g{args.focal_gamma}"
    if args.augment_minority:
        output_suffix = f"{output_suffix}_aug"

    if args.write_from_json:
        results_json = output_paths(output_suffix)[0]
        if not results_json.exists():
            raise FileNotFoundError(f"Results JSON not found: {results_json}")
        results = json.loads(results_json.read_text(encoding="utf-8"))
        write_outputs(results, output_suffix)
        print("Regenerated CSV outputs from:", results_json)
        return

    set_seed(RANDOM_SEED)
    rows = [row for row in read_rows() if valid_segment(row)]
    speakers = VIETNAMESE_SPEAKERS[: args.max_folds] if args.max_folds else VIETNAMESE_SPEAKERS
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    print("Vietnamese speaker-disjoint CNN Attention context-window experiment")
    print("Metadata:", METADATA_CSV)
    print("Speakers:", speakers)
    print("Context modes:", {mode: CONTEXT_MODES[mode] for mode in args.context_modes})
    print("Loss type:", args.loss_type)
    print("Output suffix:", output_suffix)
    print("Training config:", training_config(args.epochs, args.context_modes, args.loss_type))
    print("Note: confidence is model confidence, not pronunciation correctness.")
    print_gpu_telemetry("before folds")

    if args.dry_run:
        print_dry_run(rows, speakers, args.context_modes)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("Using device:", device)

    folds = []
    for context_mode in args.context_modes:
        for speaker in speakers:
            folds.append(run_fold(context_mode, speaker, rows, args.epochs, label_to_index, index_to_label, device, output_suffix, args.loss_type, args.focal_gamma, args.augment_minority))
            print_gpu_telemetry(f"after context {context_mode} fold {speaker}")

    summary = build_summary(folds, args.context_modes)
    results = {
        "metadata_csv": str(METADATA_CSV),
        "protocol": {
            "held_out_speakers": speakers,
            "train_rule": "Rows where speaker_id != held_out_speaker and split == train.",
            "validation_rule": "Rows where speaker_id != held_out_speaker and split == val.",
            "test_rule": "All rows where speaker_id == held_out_speaker.",
            "non_vietnamese_training": "Included through original train/validation splits when not held out.",
            "held_out_leakage_rule": "Held-out Vietnamese speaker is excluded from train and validation.",
        },
        "label_order": LABEL_ORDER,
        "config": training_config(args.epochs, args.context_modes, args.loss_type),
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
        "folds": folds,
        "summary": summary,
    }
    write_outputs(results, output_suffix)

    print()
    print("Speaker-disjoint context summary:")
    for row in summary["by_context_mode"]:
        print(
            f"- {row['context_mode']}: macro_f1={row['mean_macro_f1']:.4f} +/- {row['std_macro_f1']:.4f}, "
            f"addition_f1={row['mean_addition_f1']:.4f} +/- {row['std_addition_f1']:.4f}, "
            f"accuracy={row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}"
        )
    print("Generated files:")
    for path in output_paths(output_suffix):
        print(f"- {path}")


if __name__ == "__main__":
    main()
