from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
MODEL_OUTPUT = Path("ai-training/models/l2_arctic_error_type_cnn_v2.pt")

SAMPLE_RATE = 16000
MAX_SECONDS = 1
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
USE_FOCAL_LOSS = True
USE_WEIGHTED_SAMPLER = True
USE_AUGMENTATION = True
FOCAL_GAMMA = 2.0
RANDOM_SEED = 42
IMBALANCE_WEIGHT_POWER = 0.4

ERROR_TYPES = ("addition", "deletion", "substitution")


class ErrorTypeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        error_to_label: dict[str, int],
        augment: bool = False,
    ):
        self.rows = rows
        self.error_to_label = error_to_label
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio = load_segment(row)

        if self.augment:
            audio = augment_audio(audio)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.error_to_label[row["error_type"]], dtype=torch.long)

        return feature, target


class SmallPronunciationCNNV2(nn.Module):
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Classification CSV not found: {METADATA_CSV}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_segment(row: dict[str, str]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except (KeyError, ValueError):
        return False

    return (
        row.get("error_type") in ERROR_TYPES
        and Path(row["audio_path"]).exists()
        and start_time >= 0
        and end_time > start_time
    )


def load_segment(row: dict[str, str]) -> np.ndarray:
    start_time = float(row["start_time"])
    end_time = float(row["end_time"])
    duration = end_time - start_time

    audio, _ = librosa.load(
        row["audio_path"],
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_time,
        duration=duration,
    )

    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]

    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))

    return audio.astype(np.float32, copy=False)


def augment_audio(audio: np.ndarray) -> np.ndarray:
    augmented = audio.copy()

    gain = np.random.uniform(0.85, 1.15)
    augmented *= gain

    noise_std = np.random.uniform(0.0005, 0.003)
    augmented += np.random.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)

    if np.random.random() < 0.5:
        max_shift = int(0.02 * SAMPLE_RATE)
        shift = np.random.randint(-max_shift, max_shift + 1)
        augmented = np.roll(augmented, shift)
        if shift > 0:
            augmented[:shift] = 0.0
        elif shift < 0:
            augmented[shift:] = 0.0

    return np.clip(augmented, -1.0, 1.0)


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
    label_to_error: dict[int, str],
) -> dict:
    classes = sorted(label_to_error)
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    total = len(labels)
    per_class = {}

    for class_id in classes:
        true_positive = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label == class_id and prediction == class_id
        )
        false_positive = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label != class_id and prediction == class_id
        )
        false_negative = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label == class_id and prediction != class_id
        )

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label_to_error[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1),
        "per_class": per_class,
    }


def build_class_weights(rows: list[dict[str, str]], error_to_label: dict[str, int]) -> torch.Tensor:
    counts = Counter(row["error_type"] for row in rows)
    total = sum(counts.values())
    weights = torch.ones(len(error_to_label), dtype=torch.float32)

    for error_type, label in error_to_label.items():
        weights[label] = (total / (len(error_to_label) * counts[error_type])) ** IMBALANCE_WEIGHT_POWER

    return weights


def build_sampler(rows: list[dict[str, str]], error_to_label: dict[str, int]) -> WeightedRandomSampler:
    counts = Counter(row["error_type"] for row in rows)
    sample_weights = [
        1.0 / (counts[row["error_type"]] ** IMBALANCE_WEIGHT_POWER)
        for row in rows
        if row["error_type"] in error_to_label
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(loader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, label_to_error):
    model.eval()
    total_loss = 0.0
    labels = []
    predictions = []

    with torch.no_grad():
        for features, targets in tqdm(loader, desc="Validating"):
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            labels.extend(targets.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = calculate_metrics(labels, predictions, label_to_error)
    return total_loss / max(len(loader), 1), metrics


def print_distribution(rows: list[dict[str, str]], title: str) -> None:
    counts = Counter(row["error_type"] for row in rows)
    print(title)
    for error_type in ERROR_TYPES:
        print(f"- {error_type}: {counts[error_type]}")


def main():
    set_seed(RANDOM_SEED)
    rows = [row for row in read_rows() if valid_segment(row)]
    rows = [row for row in rows if row["error_type"] in ERROR_TYPES]

    error_to_label = {error_type: index for index, error_type in enumerate(ERROR_TYPES)}
    label_to_error = {label: error_type for error_type, label in error_to_label.items()}

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]

    if not train_rows or not val_rows:
        print("Skipping training: train or validation split is empty.")
        return

    print_distribution(rows, "Rows by error_type:")
    print_distribution(train_rows, "Train rows by error_type:")
    print_distribution(val_rows, "Val rows by error_type:")
    print("Error labels:")
    for error_type, label in error_to_label.items():
        print(f"- {error_type}: {label}")

    class_weights = build_class_weights(train_rows, error_to_label)
    print("Class weights:")
    for error_type, label in error_to_label.items():
        print(f"- {error_type}: {class_weights[label].item():.4f}")

    sampler = build_sampler(train_rows, error_to_label) if USE_WEIGHTED_SAMPLER else None
    train_loader = DataLoader(
        ErrorTypeDataset(train_rows, error_to_label, augment=USE_AUGMENTATION),
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        ErrorTypeDataset(val_rows, error_to_label, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = SmallPronunciationCNNV2(num_classes=len(error_to_label)).to(device)
    loss_weights = class_weights.to(device)
    criterion: nn.Module
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(weight=loss_weights, gamma=FOCAL_GAMMA)
        print(f"Loss: focal loss with weighted cross entropy, gamma={FOCAL_GAMMA}")
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        print("Loss: weighted cross entropy")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_macro_f1 = -1.0

    for epoch in range(EPOCHS):
        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, label_to_error)
        val_accuracy = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        per_class_f1 = {
            error_type: metrics["f1"]
            for error_type, metrics in val_metrics["per_class"].items()
        }

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )
        print("Per-class F1:")
        for error_type in ERROR_TYPES:
            print(f"- {error_type}: {per_class_f1[error_type]:.4f}")

        if val_macro_f1 >= best_val_macro_f1:
            previous = best_val_macro_f1
            best_val_macro_f1 = val_macro_f1
            MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "error_to_label": error_to_label,
                    "sample_rate": SAMPLE_RATE,
                    "max_seconds": MAX_SECONDS,
                    "max_length": MAX_LENGTH,
                    "n_mels": N_MELS,
                    "task": "l2_arctic_phone_error_type_classification_v2",
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "num_workers": NUM_WORKERS,
                        "use_focal_loss": USE_FOCAL_LOSS,
                        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
                        "use_augmentation": USE_AUGMENTATION,
                        "focal_gamma": FOCAL_GAMMA,
                        "random_seed": RANDOM_SEED,
                        "imbalance_weight_power": IMBALANCE_WEIGHT_POWER,
                    },
                    "class_weights": class_weights.tolist(),
                    "best_epoch": epoch + 1,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_accuracy": val_accuracy,
                    "best_val_per_class_f1": per_class_f1,
                },
                MODEL_OUTPUT,
            )
            print(
                "Saved checkpoint: "
                f"val_macro_f1 improved from {previous:.4f} to {best_val_macro_f1:.4f}"
            )
        else:
            print(
                "Checkpoint not updated: "
                f"val_macro_f1={val_macro_f1:.4f} did not improve best={best_val_macro_f1:.4f}"
            )

    print()
    print("Training done.")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
