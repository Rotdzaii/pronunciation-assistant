from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import random
import subprocess

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
MODEL_OUTPUT = Path("ai-training/models/l2_arctic_error_type_cnn_attention.pt")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)

BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42
DROPOUT = 0.2

LABEL_ORDER = ["addition", "deletion", "substitution"]


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


class ErrorSegmentDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_to_index: dict[str, int]):
        self.rows = rows
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        audio = load_segment(row)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.label_to_index[row["error_type"]], dtype=torch.long)
        return feature, target


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
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(x)
        pooled, _ = self.attention(feature_map)
        return self.classifier(pooled)


def build_sampler(rows: list[dict[str, str]]) -> WeightedRandomSampler:
    counts = Counter(row["error_type"] for row in rows)
    sample_weights = [1.0 / counts[row["error_type"]] for row in rows]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def calculate_metrics(labels: list[int], predictions: list[int], index_to_label: dict[int, str]) -> dict:
    label_indexes = list(range(len(index_to_label)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indexes,
        zero_division=0,
    )

    per_class = {}
    for index in label_indexes:
        per_class[index_to_label[index]] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "per_class": per_class,
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, index_to_label: dict[int, str]) -> dict:
    model.eval()
    labels = []
    predictions = []

    with torch.no_grad():
        for features, targets in tqdm(loader, desc="Validating", leave=False):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(features)
            labels.extend(targets.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    return calculate_metrics(labels, predictions, index_to_label)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(loader, desc="Training"):
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


def print_distribution(rows: list[dict[str, str]], title: str) -> None:
    counts = Counter(row["error_type"] for row in rows)
    print(title)
    for label in LABEL_ORDER:
        print(f"- {label}: {counts[label]}")


def main() -> None:
    set_seed(RANDOM_SEED)

    rows = [row for row in read_rows() if valid_segment(row)]
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]

    if not train_rows or not val_rows:
        raise RuntimeError("Train or validation split is empty.")

    print("Metadata:", METADATA_CSV)
    print_distribution(rows, "All rows by error_type:")
    print_distribution(train_rows, "Train rows by error_type:")
    print_distribution(val_rows, "Validation rows by error_type:")
    print("Training strategy: WeightedRandomSampler only")
    print("Loss: CrossEntropyLoss without class weights")
    print("Attention: temporal attention over CNN feature map after frequency averaging")

    train_loader = DataLoader(
        ErrorSegmentDataset(train_rows, label_to_index),
        batch_size=BATCH_SIZE,
        sampler=build_sampler(train_rows),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ErrorSegmentDataset(val_rows, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = SmallPronunciationCNNAttention(num_classes=len(LABEL_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print_gpu_telemetry("before training")

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, index_to_label)

        improved = val_metrics["macro_f1"] > best_val_macro_f1
        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_f1_addition={val_metrics['per_class']['addition']['f1']:.4f} "
            f"val_f1_deletion={val_metrics['per_class']['deletion']['f1']:.4f} "
            f"val_f1_substitution={val_metrics['per_class']['substitution']['f1']:.4f} "
            f"checkpoint_improved={improved}"
        )
        print_gpu_telemetry(f"after epoch {epoch + 1}")

        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_metrics = val_metrics
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
                    "task": "l2_arctic_phone_error_type_cnn_attention",
                    "training_strategy": "weighted_random_sampler_only",
                    "loss": "cross_entropy_unweighted",
                    "pooling": "temporal_attention",
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_metrics,
                    "label_order": LABEL_ORDER,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "num_workers": NUM_WORKERS,
                        "random_seed": RANDOM_SEED,
                        "dropout": DROPOUT,
                    },
                    "note": "Confidence is classifier confidence, not pronunciation correctness.",
                },
                MODEL_OUTPUT,
            )
            print(f"Saved checkpoint: {MODEL_OUTPUT}")

    print()
    print("Training done.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    if best_metrics is not None:
        print(
            "Best validation per-class F1: "
            f"addition={best_metrics['per_class']['addition']['f1']:.4f}, "
            f"deletion={best_metrics['per_class']['deletion']['f1']:.4f}, "
            f"substitution={best_metrics['per_class']['substitution']['f1']:.4f}"
        )
    print(f"Checkpoint: {MODEL_OUTPUT}")
    print_gpu_telemetry("after training")


if __name__ == "__main__":
    main()
