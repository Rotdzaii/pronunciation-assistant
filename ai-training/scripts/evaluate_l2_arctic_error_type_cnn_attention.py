from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_cnn_attention.pt")
EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

METRICS_JSON = EVALUATION_DIR / "cnn_attention_eval_metrics.json"
CONFUSION_MATRIX_CSV = EVALUATION_DIR / "cnn_attention_confusion_matrix.csv"
PER_CLASS_CSV = EVALUATION_DIR / "cnn_attention_per_class_metrics.csv"
PER_SPEAKER_CSV = EVALUATION_DIR / "cnn_attention_per_speaker_metrics.csv"
MISCLASSIFIED_CSV = EVALUATION_DIR / "cnn_attention_misclassified_examples.csv"

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 8
NUM_WORKERS = 0
LABEL_ORDER = ["addition", "deletion", "substitution"]


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_segment(row: dict[str, str], label_to_index: dict[str, int]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except (KeyError, ValueError):
        return False
    return (
        row.get("error_type") in label_to_index
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
    def __init__(self, num_classes: int, dropout: float = 0.2):
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
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(x)
        pooled, _ = self.attention(feature_map)
        return self.classifier(pooled)


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def calculate_metrics(labels: list[int], predictions: list[int], index_to_label: dict[int, str]) -> dict:
    label_indexes = list(range(len(index_to_label)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_indexes,
        zero_division=0,
    )
    total = len(labels)
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    per_class = {}

    for index in label_indexes:
        predicted_count = sum(1 for prediction in predictions if prediction == index)
        true_positive = sum(
            1
            for label, prediction in zip(labels, predictions)
            if label == index and prediction == index
        )
        per_class[index_to_label[index]] = {
            "class_id": index,
            "support": int(support[index]),
            "predicted_count": int(predicted_count),
            "correct": int(true_positive),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "per_class": per_class,
    }


def evaluate_split(
    split: str,
    rows: list[dict[str, str]],
    model: nn.Module,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
    device: torch.device,
) -> dict:
    dataset = ErrorSegmentDataset(rows, label_to_index)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    labels = []
    predictions = []
    probabilities = []
    evaluated_rows = []

    model.eval()
    with torch.no_grad():
        for features, targets, indexes in loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1).cpu()
            batch_predictions = probs.argmax(dim=1).tolist()

            labels.extend(targets.tolist())
            predictions.extend(batch_predictions)
            probabilities.extend(probs.tolist())
            for row_index in indexes.tolist():
                evaluated_rows.append(rows[row_index])

    metrics = calculate_metrics(labels, predictions, index_to_label)
    class_distribution = Counter(row["error_type"] for row in rows)
    speaker_counts = Counter(row["speaker_id"] for row in rows)
    speaker_correct = Counter()
    class_correct = Counter()
    class_counts = Counter()
    confusion = defaultdict(Counter)
    misclassified_examples = []

    for row, label, prediction, probs in zip(evaluated_rows, labels, predictions, probabilities):
        actual = index_to_label[label]
        predicted = index_to_label[prediction]
        speaker_id = row["speaker_id"]
        is_correct = actual == predicted

        confusion[actual][predicted] += 1
        class_counts[actual] += 1

        if is_correct:
            speaker_correct[speaker_id] += 1
            class_correct[actual] += 1
        else:
            misclassified_examples.append(
                {
                    "split": split,
                    "speaker_id": row["speaker_id"],
                    "utterance_id": row["utterance_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "label": row["label"],
                    "actual_error_type": actual,
                    "predicted_error_type": predicted,
                    "confidence": probs[prediction],
                    "audio_path": row["audio_path"],
                    "target_text": row["target_text"],
                }
            )

    per_speaker = {}
    for speaker_id, count in sorted(speaker_counts.items()):
        per_speaker[speaker_id] = {
            "count": count,
            "correct": speaker_correct[speaker_id],
            "accuracy": speaker_correct[speaker_id] / count if count else 0.0,
        }

    by_error_type = {}
    for error_type, count in sorted(class_counts.items()):
        by_error_type[error_type] = {
            "count": count,
            "correct": class_correct[error_type],
            "accuracy": class_correct[error_type] / count if count else 0.0,
        }

    class_names = [index_to_label[index] for index in sorted(index_to_label)]
    return {
        "split": split,
        "class_distribution": dict(sorted(class_distribution.items())),
        "metrics": metrics,
        "per_speaker": per_speaker,
        "by_error_type": by_error_type,
        "confusion_matrix": {
            actual: {predicted: confusion[actual][predicted] for predicted in class_names}
            for actual in class_names
        },
        "misclassified_examples": misclassified_examples,
    }


def write_outputs(results: dict, index_to_label: dict[int, str]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    class_names = [index_to_label[index] for index in sorted(index_to_label)]

    with CONFUSION_MATRIX_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "actual_error_type", *class_names])
        writer.writeheader()
        for split, result in results["splits"].items():
            for actual in class_names:
                row = {"split": split, "actual_error_type": actual}
                row.update(result["confusion_matrix"][actual])
                writer.writerow(row)

    with PER_CLASS_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "split",
            "error_type",
            "class_id",
            "support",
            "predicted_count",
            "correct",
            "precision",
            "recall",
            "f1",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split, result in results["splits"].items():
            for error_type, metrics in result["metrics"]["per_class"].items():
                writer.writerow({"split": split, "error_type": error_type, **metrics})

    with PER_SPEAKER_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "speaker_id", "count", "correct", "accuracy"])
        writer.writeheader()
        for split, result in results["splits"].items():
            for speaker_id, metrics in result["per_speaker"].items():
                writer.writerow({"split": split, "speaker_id": speaker_id, **metrics})

    with MISCLASSIFIED_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "split",
            "speaker_id",
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
        for result in results["splits"].values():
            writer.writerows(result["misclassified_examples"][:500])


def print_split_summary(split: str, result: dict) -> None:
    metrics = result["metrics"]
    print(f"{split} samples: {metrics['total_samples']}")
    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )
    print("Per-class precision/recall/F1:")
    for error_type, item in metrics["per_class"].items():
        print(
            f"- {error_type}: precision={item['precision']:.4f} "
            f"recall={item['recall']:.4f} f1={item['f1']:.4f} support={item['support']}"
        )
    print("Confusion matrix:")
    for actual, predicted_counts in result["confusion_matrix"].items():
        print(f"- actual {actual}: {predicted_counts}")
    print("Per-speaker accuracy:")
    for speaker_id, item in result["per_speaker"].items():
        print(f"- {speaker_id}: accuracy={item['accuracy']:.4f} count={item['count']}")


def main() -> None:
    checkpoint = load_checkpoint()
    label_to_index = checkpoint.get("label_to_index", checkpoint.get("error_to_label"))
    index_to_label = {int(index): label for index, label in checkpoint.get("index_to_label", {}).items()}
    if not index_to_label:
        index_to_label = {index: label for label, index in label_to_index.items()}

    rows = [row for row in read_rows() if valid_segment(row, label_to_index)]
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("val", "test")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = checkpoint.get("config", {}).get("dropout", 0.2)
    model = SmallPronunciationCNNAttention(num_classes=len(label_to_index), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Using device:", device)
    print("Note: confidence is classifier confidence, not pronunciation correctness.")
    print()

    results = {
        "metadata_csv": str(METADATA_CSV),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "sample_rate": checkpoint.get("sample_rate", SAMPLE_RATE),
        "max_seconds": checkpoint.get("max_seconds", MAX_SECONDS),
        "n_mels": checkpoint.get("n_mels", N_MELS),
        "task": checkpoint.get("task", "l2_arctic_phone_error_type_cnn_attention"),
        "training_strategy": checkpoint.get("training_strategy", "weighted_random_sampler_only"),
        "loss": checkpoint.get("loss", "cross_entropy_unweighted"),
        "pooling": checkpoint.get("pooling", "temporal_attention"),
        "config": checkpoint.get("config", {}),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
        "label_to_index": label_to_index,
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
        "splits": {},
    }

    for split, items in split_rows.items():
        if not items:
            continue
        result = evaluate_split(split, items, model, label_to_index, index_to_label, device)
        results["splits"][split] = result
        print_split_summary(split, result)
        print()

    write_outputs(results, index_to_label)
    print("Generated files:")
    for path in [METRICS_JSON, CONFUSION_MATRIX_CSV, PER_CLASS_CSV, PER_SPEAKER_CSV, MISCLASSIFIED_CSV]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
