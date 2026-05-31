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
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
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


def checkpoint_path(context_mode: str, held_out_speaker: str) -> Path:
    return MODEL_DIR / f"l2_arctic_cnn_attention_speaker_disjoint_context_{context_mode}_{held_out_speaker}.pt"


def training_config(epochs: int, context_modes: list[str]) -> dict:
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
        "sampler": "inverse-frequency weighted random sampler, same as baseline speaker-disjoint run",
        "loss": "cross_entropy_unweighted",
    }


def run_fold(context_mode, held_out_speaker, rows, epochs, label_to_index, index_to_label, device) -> dict:
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

    train_loader = DataLoader(
        ErrorSegmentDataset(train_rows, label_to_index, context_seconds),
        batch_size=BATCH_SIZE,
        sampler=build_sampler(train_rows),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallPronunciationCNNAttention(num_classes=len(LABEL_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    output_path = checkpoint_path(context_mode, held_out_speaker)
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
                    "training_strategy": "weighted_random_sampler_with_context_window",
                    "loss": "cross_entropy_unweighted",
                    "pooling": "temporal_attention",
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_val_metrics,
                    "label_order": LABEL_ORDER,
                    "config": training_config(epochs, [context_mode]),
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


def write_outputs(results: dict) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with PER_FOLD_CSV.open("w", encoding="utf-8", newline="") as file:
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
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["context_mode"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with PER_CLASS_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["context_mode", "held_out_speaker", "error_type", "class_id", "support", "predicted_count", "correct", "precision", "recall", "f1"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for fold in results["folds"]:
            for label, metrics in fold["test"]["metrics"]["per_class"].items():
                writer.writerow({"context_mode": fold["context_mode"], "held_out_speaker": fold["held_out_speaker"], "error_type": label, **metrics})

    with CONFUSION_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["context_mode", "held_out_speaker", "actual_error_type", *LABEL_ORDER])
        writer.writeheader()
        for fold in results["folds"]:
            for actual in LABEL_ORDER:
                row = {"context_mode": fold["context_mode"], "held_out_speaker": fold["held_out_speaker"], "actual_error_type": actual}
                row.update(fold["test"]["confusion_matrix"][actual])
                writer.writerow(row)

    with MISCLASSIFIED_CSV.open("w", encoding="utf-8", newline="") as file:
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

    if args.write_from_json:
        if not RESULTS_JSON.exists():
            raise FileNotFoundError(f"Results JSON not found: {RESULTS_JSON}")
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        write_outputs(results)
        print("Regenerated CSV outputs from:", RESULTS_JSON)
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
    print("Training config:", training_config(args.epochs, args.context_modes))
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
            folds.append(run_fold(context_mode, speaker, rows, args.epochs, label_to_index, index_to_label, device))
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
        "config": training_config(args.epochs, args.context_modes),
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
        "folds": folds,
        "summary": summary,
    }
    write_outputs(results)

    print()
    print("Speaker-disjoint context summary:")
    for row in summary["by_context_mode"]:
        print(
            f"- {row['context_mode']}: macro_f1={row['mean_macro_f1']:.4f} +/- {row['std_macro_f1']:.4f}, "
            f"addition_f1={row['mean_addition_f1']:.4f} +/- {row['std_addition_f1']:.4f}, "
            f"accuracy={row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}"
        )
    print("Generated files:")
    for path in [RESULTS_JSON, PER_FOLD_CSV, SUMMARY_CSV, PER_CLASS_CSV, CONFUSION_CSV, MISCLASSIFIED_CSV]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
