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
from torch.utils.data import DataLoader, Dataset


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_cnn.pt")
EVALUATION_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

METRICS_JSON = EVALUATION_DIR / "error_type_eval_metrics.json"
CONFUSION_MATRIX_CSV = EVALUATION_DIR / "error_type_confusion_matrix.csv"
PER_CLASS_CSV = EVALUATION_DIR / "error_type_per_class_metrics.csv"
PER_SPEAKER_CSV = EVALUATION_DIR / "error_type_per_speaker_metrics.csv"
MISCLASSIFIED_CSV = EVALUATION_DIR / "error_type_misclassified_examples.csv"
CONFUSION_MATRIX_PNG = EVALUATION_DIR / "error_type_confusion_matrix.png"

SAMPLE_RATE = 16000
MAX_SECONDS = 1
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64
BATCH_SIZE = 8
NUM_WORKERS = 0


class ErrorTypeDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], error_to_label: dict[str, int]):
        self.rows = rows
        self.error_to_label = error_to_label

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
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

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self.error_to_label[row["error_type"]], dtype=torch.long)

        return feature, target, index


class SmallPronunciationCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_segment(row: dict[str, str]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except ValueError:
        return False

    return Path(row["audio_path"]).exists() and start_time >= 0 and end_time > start_time


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
    label_to_error: dict[int, str],
) -> dict:
    classes = sorted(label_to_error)
    total = len(labels)
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    per_class = {}

    for class_id in classes:
        true_positive = sum(
            1 for label, prediction in zip(labels, predictions)
            if label == class_id and prediction == class_id
        )
        false_positive = sum(
            1 for label, prediction in zip(labels, predictions)
            if label != class_id and prediction == class_id
        )
        false_negative = sum(
            1 for label, prediction in zip(labels, predictions)
            if label == class_id and prediction != class_id
        )
        support = sum(1 for label in labels if label == class_id)
        predicted_count = sum(1 for prediction in predictions if prediction == class_id)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_class[label_to_error[class_id]] = {
            "class_id": class_id,
            "support": support,
            "predicted_count": predicted_count,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_precision = sum(item["precision"] for item in per_class.values()) / max(len(per_class), 1)
    macro_recall = sum(item["recall"] for item in per_class.values()) / max(len(per_class), 1)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1)

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def evaluate_split(
    split: str,
    rows: list[dict[str, str]],
    model: nn.Module,
    error_to_label: dict[str, int],
    label_to_error: dict[int, str],
    device: torch.device,
) -> dict:
    dataset = ErrorTypeDataset(rows, error_to_label)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    labels = []
    predictions = []
    evaluated_rows = []

    model.eval()
    with torch.no_grad():
        for features, targets, indexes in loader:
            features = features.to(device)
            outputs = model(features)
            batch_predictions = outputs.argmax(dim=1).cpu().tolist()
            batch_targets = targets.tolist()

            labels.extend(batch_targets)
            predictions.extend(batch_predictions)

            for row_index in indexes.tolist():
                evaluated_rows.append(rows[row_index])

    metrics = calculate_metrics(labels, predictions, label_to_error)
    class_distribution = Counter(row["error_type"] for row in rows)
    speaker_counts = Counter(row["speaker_id"] for row in rows)
    speaker_correct = Counter()
    class_correct = Counter()
    class_counts = Counter()
    confusion = defaultdict(Counter)
    misclassified_examples = []
    error_examples_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row, label, prediction in zip(evaluated_rows, labels, predictions):
        actual = label_to_error[label]
        predicted = label_to_error[prediction]
        speaker_id = row["speaker_id"]
        is_correct = actual == predicted

        confusion[actual][predicted] += 1
        class_counts[actual] += 1

        if is_correct:
            speaker_correct[speaker_id] += 1
            class_correct[actual] += 1
        else:
            example = {
                "split": split,
                "speaker_id": row["speaker_id"],
                "utterance_id": row["utterance_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "label": row["label"],
                "actual_error_type": actual,
                "predicted_error_type": predicted,
                "audio_path": row["audio_path"],
                "target_text": row["target_text"],
            }
            misclassified_examples.append(example)

            if len(error_examples_by_class[actual]) < 10:
                error_examples_by_class[actual].append(example)

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
            "error_examples": error_examples_by_class.get(error_type, []),
        }

    return {
        "split": split,
        "class_distribution": dict(sorted(class_distribution.items())),
        "metrics": metrics,
        "per_speaker": per_speaker,
        "by_error_type": by_error_type,
        "confusion_matrix": {
            actual: {predicted: confusion[actual][predicted] for predicted in label_to_error.values()}
            for actual in label_to_error.values()
        },
        "misclassified_examples": misclassified_examples,
    }


def write_outputs(results: dict[str, dict], label_to_error: dict[int, str]):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    class_names = [label_to_error[index] for index in sorted(label_to_error)]

    with CONFUSION_MATRIX_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["split", "actual_error_type", *class_names]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for split, result in results["splits"].items():
            for actual in class_names:
                row = {
                    "split": split,
                    "actual_error_type": actual,
                }
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
        fieldnames = ["split", "speaker_id", "count", "correct", "accuracy"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
            "audio_path",
            "target_text",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results["splits"].values():
            writer.writerows(result["misclassified_examples"][:500])


def maybe_write_confusion_matrix_png(results: dict[str, dict], label_to_error: dict[int, str]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if "val" not in results["splits"]:
        return False

    class_names = [label_to_error[index] for index in sorted(label_to_error)]
    matrix = np.array(
        [
            [results["splits"]["val"]["confusion_matrix"][actual][predicted] for predicted in class_names]
            for actual in class_names
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation Confusion Matrix")

    for row_index in range(len(class_names)):
        for col_index in range(len(class_names)):
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=160)
    plt.close(fig)
    return True


def print_split_summary(split: str, result: dict):
    metrics = result["metrics"]
    print(f"{split} samples: {metrics['total_samples']}")
    print("Class distribution:")
    for error_type, count in result["class_distribution"].items():
        print(f"- {error_type}: {count}")

    print(
        f"accuracy={metrics['accuracy']:.4f} "
        f"macro_precision={metrics['macro_precision']:.4f} "
        f"macro_recall={metrics['macro_recall']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print("Per-class metrics:")
    for error_type, item in metrics["per_class"].items():
        print(
            f"- {error_type}: "
            f"precision={item['precision']:.4f} "
            f"recall={item['recall']:.4f} "
            f"f1={item['f1']:.4f} "
            f"support={item['support']}"
        )
    print("Confusion matrix:")
    for actual, predicted_counts in result["confusion_matrix"].items():
        print(f"- actual {actual}: {predicted_counts}")
    print("Per-speaker accuracy:")
    for speaker_id, item in result["per_speaker"].items():
        print(f"- {speaker_id}: accuracy={item['accuracy']:.4f} count={item['count']}")


def main():
    checkpoint = load_checkpoint()
    error_to_label = checkpoint["error_to_label"]
    label_to_error = {label: error_type for error_type, label in error_to_label.items()}

    rows = [row for row in read_rows() if valid_segment(row) and row["error_type"] in error_to_label]
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("val", "test")
    }
    split_rows = {split: items for split, items in split_rows.items() if items}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPronunciationCNN(num_classes=len(error_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Using device:", device)
    print("Error labels:")
    for error_type, label in sorted(error_to_label.items(), key=lambda item: item[1]):
        print(f"- {label}: {error_type}")
    print()

    results = {
        "metadata_csv": str(METADATA_CSV),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "sample_rate": checkpoint.get("sample_rate", SAMPLE_RATE),
        "max_seconds": checkpoint.get("max_seconds", MAX_SECONDS),
        "n_mels": checkpoint.get("n_mels", N_MELS),
        "task": checkpoint.get("task", "l2_arctic_phone_error_type_classification"),
        "error_to_label": error_to_label,
        "splits": {},
    }

    for split, items in split_rows.items():
        result = evaluate_split(
            split=split,
            rows=items,
            model=model,
            error_to_label=error_to_label,
            label_to_error=label_to_error,
            device=device,
        )
        results["splits"][split] = result
        print_split_summary(split, result)
        print()

    write_outputs(results, label_to_error)
    png_created = maybe_write_confusion_matrix_png(results, label_to_error)

    print("Generated files:")
    for path in [
        METRICS_JSON,
        CONFUSION_MATRIX_CSV,
        PER_CLASS_CSV,
        PER_SPEAKER_CSV,
        MISCLASSIFIED_CSV,
    ]:
        print(f"- {path}")

    if png_created:
        print(f"- {CONFUSION_MATRIX_PNG}")
    else:
        print("- confusion matrix PNG not generated; matplotlib unavailable or validation split missing")


if __name__ == "__main__":
    main()
