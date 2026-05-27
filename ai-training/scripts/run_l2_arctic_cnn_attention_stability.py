from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import random
import statistics
import subprocess

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
MODEL_DIR = Path("ai-training/models")
EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

RUNS_CSV = EVALUATION_DIR / "cnn_attention_stability_runs.csv"
SUMMARY_CSV = EVALUATION_DIR / "cnn_attention_stability_summary.csv"
METRICS_JSON = EVALUATION_DIR / "cnn_attention_stability_metrics.json"

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
DROPOUT = 0.2

SEEDS = [42, 123, 2026]
LABEL_ORDER = ["addition", "deletion", "substitution"]

CNN_V2_TEST_MACRO_F1 = 0.4835
CNN_V2_TEST_ADDITION_F1 = 0.1240


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_telemetry(stage: str) -> dict:
    telemetry = {
        "stage": stage,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "memory_allocated_mb": None,
        "memory_reserved_mb": None,
        "max_memory_allocated_mb": None,
        "max_memory_reserved_mb": None,
        "nvidia_smi": None,
    }

    print()
    print(f"GPU telemetry - {stage}")
    print("CUDA available:", telemetry["cuda_available"])
    print("Note: low GPU utilization can happen because audio loading/preprocessing is CPU-bound and this CNN is small.")

    if torch.cuda.is_available():
        telemetry["gpu_name"] = torch.cuda.get_device_name(0)
        telemetry["memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
        telemetry["memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
        telemetry["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(0) / 1024 / 1024
        telemetry["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved(0) / 1024 / 1024

        print("GPU name:", telemetry["gpu_name"])
        print(f"memory allocated MB: {telemetry['memory_allocated_mb']:.2f}")
        print(f"memory reserved MB: {telemetry['memory_reserved_mb']:.2f}")
        print(f"max memory allocated MB: {telemetry['max_memory_allocated_mb']:.2f}")
        print(f"max memory reserved MB: {telemetry['max_memory_reserved_mb']:.2f}")

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
        telemetry["nvidia_smi"] = f"unavailable ({exc})"
        print("nvidia-smi utilization:", telemetry["nvidia_smi"])
        return telemetry

    telemetry["nvidia_smi"] = result.stdout.strip() or "no GPU rows returned"
    print("nvidia-smi utilization:", telemetry["nvidia_smi"])
    return telemetry


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
    audio, _ = librosa.load(
        row["audio_path"],
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_time,
        duration=end_time - start_time,
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
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
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


def build_sampler(rows: list[dict[str, str]], seed: int) -> WeightedRandomSampler:
    counts = Counter(row["error_type"] for row in rows)
    sample_weights = [1.0 / counts[row["error_type"]] for row in rows]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
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
        for features, targets in tqdm(loader, desc="Evaluating", leave=False):
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

    for features, labels in tqdm(loader, desc="Training", leave=False):
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


def metric_row(seed: int, best_epoch: int, val_metrics: dict, test_metrics: dict, checkpoint_path: Path) -> dict:
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_addition_f1": val_metrics["per_class"]["addition"]["f1"],
        "val_deletion_f1": val_metrics["per_class"]["deletion"]["f1"],
        "val_substitution_f1": val_metrics["per_class"]["substitution"]["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_addition_f1": test_metrics["per_class"]["addition"]["f1"],
        "test_deletion_f1": test_metrics["per_class"]["deletion"]["f1"],
        "test_substitution_f1": test_metrics["per_class"]["substitution"]["f1"],
        "checkpoint_path": str(checkpoint_path),
    }


def run_seed(
    seed: int,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
    device: torch.device,
) -> dict:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_loader = DataLoader(
        ErrorSegmentDataset(train_rows, label_to_index),
        batch_size=BATCH_SIZE,
        sampler=build_sampler(train_rows, seed),
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
    test_loader = DataLoader(
        ErrorSegmentDataset(test_rows, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallPronunciationCNNAttention(num_classes=len(LABEL_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = MODEL_DIR / f"l2_arctic_error_type_cnn_attention_seed_{seed}.pt"

    best_epoch = 0
    best_val_metrics = None
    best_val_macro_f1 = -1.0

    print()
    print(f"Seed {seed}")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device, index_to_label)

        improved = val_metrics["macro_f1"] > best_val_macro_f1
        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_addition_f1={val_metrics['per_class']['addition']['f1']:.4f} "
            f"checkpoint_improved={improved}"
        )

        if improved:
            best_epoch = epoch + 1
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
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
                    "task": "l2_arctic_phone_error_type_cnn_attention_stability",
                    "training_strategy": "weighted_random_sampler_only",
                    "loss": "cross_entropy_unweighted",
                    "pooling": "temporal_attention",
                    "seed": seed,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                    "label_order": LABEL_ORDER,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "num_workers": NUM_WORKERS,
                        "dropout": DROPOUT,
                    },
                    "note": "Confidence is classifier confidence, not pronunciation correctness.",
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate_model(model, val_loader, device, index_to_label)
    test_metrics = evaluate_model(model, test_loader, device, index_to_label)

    print(
        f"Seed {seed} result: "
        f"best_epoch={best_epoch} "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} "
        f"test_addition_f1={test_metrics['per_class']['addition']['f1']:.4f} "
        f"test_deletion_f1={test_metrics['per_class']['deletion']['f1']:.4f} "
        f"test_substitution_f1={test_metrics['per_class']['substitution']['f1']:.4f}"
    )

    telemetry = print_gpu_telemetry(f"after seed {seed}")
    row = metric_row(seed, best_epoch, val_metrics, test_metrics, checkpoint_path)

    return {
        "row": row,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "gpu_telemetry": telemetry,
    }


def mean_std(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "metric": key,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_outputs(results: list[dict], summary_rows: list[dict], payload: dict) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "best_epoch",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_addition_f1",
        "val_deletion_f1",
        "val_substitution_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_addition_f1",
        "test_deletion_f1",
        "test_substitution_f1",
        "checkpoint_path",
    ]

    with RUNS_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    rows = [row for row in read_rows() if valid_segment(row)]
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Train, validation, or test split is empty.")

    print("Metadata:", METADATA_CSV)
    print("Seeds:", SEEDS)
    print("Training strategy: WeightedRandomSampler only")
    print("Loss: CrossEntropyLoss without class weights")
    print("Note: confidence is classifier confidence, not pronunciation correctness.")
    print("Split sizes:")
    print("- train:", len(train_rows), Counter(row["error_type"] for row in train_rows))
    print("- val:", len(val_rows), Counter(row["error_type"] for row in val_rows))
    print("- test:", len(test_rows), Counter(row["error_type"] for row in test_rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    initial_telemetry = print_gpu_telemetry("before stability check")

    results = []
    detailed_results = []

    for seed in SEEDS:
        result = run_seed(seed, train_rows, val_rows, test_rows, label_to_index, index_to_label, device)
        results.append(result["row"])
        detailed_results.append(result)

    summary_metrics = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_addition_f1",
        "test_deletion_f1",
        "test_substitution_f1",
        "val_macro_f1",
        "val_addition_f1",
    ]
    summary_rows = [mean_std(results, metric) for metric in summary_metrics]
    summary_by_metric = {row["metric"]: row for row in summary_rows}

    mean_test_macro = summary_by_metric["test_macro_f1"]["mean"]
    mean_test_addition = summary_by_metric["test_addition_f1"]["mean"]
    beats_macro = mean_test_macro > CNN_V2_TEST_MACRO_F1
    beats_addition = mean_test_addition > CNN_V2_TEST_ADDITION_F1

    payload = {
        "metadata_csv": str(METADATA_CSV),
        "seeds": SEEDS,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "max_seconds": MAX_SECONDS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "num_workers": NUM_WORKERS,
            "dropout": DROPOUT,
            "training_strategy": "weighted_random_sampler_only",
            "loss": "cross_entropy_unweighted",
            "pooling": "temporal_attention",
        },
        "cnn_v2_baseline": {
            "test_macro_f1": CNN_V2_TEST_MACRO_F1,
            "test_addition_f1": CNN_V2_TEST_ADDITION_F1,
        },
        "runs": results,
        "summary": summary_rows,
        "mean_test_macro_f1_beats_cnn_v2": beats_macro,
        "mean_test_addition_f1_beats_cnn_v2": beats_addition,
        "initial_gpu_telemetry": initial_telemetry,
        "gpu_telemetry_by_seed": [result["gpu_telemetry"] for result in detailed_results],
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
    }

    write_outputs(results, summary_rows, payload)

    print()
    print("Final stability summary")
    print(
        f"test_macro_f1 mean={mean_test_macro:.4f} "
        f"std={summary_by_metric['test_macro_f1']['std']:.4f} "
        f"beats_cnn_v2={beats_macro}"
    )
    print(
        f"test_addition_f1 mean={mean_test_addition:.4f} "
        f"std={summary_by_metric['test_addition_f1']['std']:.4f} "
        f"beats_cnn_v2={beats_addition}"
    )
    print("Generated files:")
    print("-", RUNS_CSV)
    print("-", SUMMARY_CSV)
    print("-", METRICS_JSON)


if __name__ == "__main__":
    main()
