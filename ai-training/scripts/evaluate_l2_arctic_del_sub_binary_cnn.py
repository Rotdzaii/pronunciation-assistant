from pathlib import Path
import csv
import json

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_del_sub_binary_cnn.pt")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 8
NUM_WORKERS = 0
LABEL_ORDER = ["deletion", "substitution"]


def print_gpu_memory(stage: str):
    print()
    print(f"GPU memory - {stage}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"CUDA memory allocated MB: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"CUDA memory reserved MB: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f}")


class ErrorSegmentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_to_index: dict[str, int]):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        label = self.label_to_index[row["error_type"]]
        audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        start_sample = max(0, int(float(row["start_time"]) * SAMPLE_RATE))
        end_sample = min(len(audio), int(float(row["end_time"]) * SAMPLE_RATE))
        segment = audio[start_sample:end_sample]
        if len(segment) == 0:
            segment = np.zeros(MAX_LENGTH, dtype=np.float32)
        if len(segment) > MAX_LENGTH:
            segment = segment[:MAX_LENGTH]
        if len(segment) < MAX_LENGTH:
            segment = np.pad(segment, (0, MAX_LENGTH - len(segment)))
        mel = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long), index


class SmallCNN(nn.Module):
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


def evaluate_split(model, dataframe: pd.DataFrame, label_to_index: dict[str, int], device):
    loader = DataLoader(
        ErrorSegmentDataset(dataframe, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_indices = []

    with torch.no_grad():
        for features, labels, indices in tqdm(loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            probabilities = torch.softmax(model(features), dim=1)
            predictions = probabilities.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_indices.extend(indices.cpu().numpy().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )
    per_class = {
        label_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label_name in enumerate(LABEL_ORDER)
    }
    matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(len(LABEL_ORDER))))
    index_to_label = {index: label for label, index in label_to_index.items()}
    rows = []
    for local_row_index, true_label, predicted_label, probabilities in zip(
        all_indices, all_labels, all_predictions, all_probabilities
    ):
        row = dataframe.iloc[local_row_index].to_dict()
        true_name = index_to_label[true_label]
        predicted_name = index_to_label[predicted_label]
        rows.append(
            {
                "speaker_id": row.get("speaker_id", ""),
                "utterance_id": row.get("utterance_id", ""),
                "audio_path": row.get("audio_path", ""),
                "start_time": row.get("start_time", ""),
                "end_time": row.get("end_time", ""),
                "true_error_type": true_name,
                "predicted_error_type": predicted_name,
                "confidence": float(max(probabilities)),
                "is_correct": true_name == predicted_name,
                "prob_deletion": float(probabilities[label_to_index["deletion"]]),
                "prob_substitution": float(probabilities[label_to_index["substitution"]]),
            }
        )

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_predictions, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "rows": rows,
    }


def save_confusion_matrix_csv(matrix: list[list[int]], output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + LABEL_ORDER)
        for label_name, row in zip(LABEL_ORDER, matrix):
            writer.writerow([label_name] + row)


def save_per_class_csv(metrics: dict, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["split", "class", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ["validation", "test"]:
            for class_name, class_metrics in metrics[split_name]["per_class"].items():
                writer.writerow({"split": split_name, "class": class_name, **class_metrics})


def save_misclassified_examples(rows: list[dict], output_path: Path, limit: int = 200):
    misclassified = [row for row in rows if not row["is_correct"]]
    misclassified = sorted(misclassified, key=lambda row: row["confidence"], reverse=True)[:limit]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        if not misclassified:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["No misclassified examples found"])
            return
        writer = csv.DictWriter(file, fieldnames=list(misclassified[0].keys()))
        writer.writeheader()
        writer.writerows(misclassified)


def print_summary(split_name: str, metrics: dict):
    print()
    print(f"{split_name} metrics:")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    for label_name in LABEL_ORDER:
        class_metrics = metrics["per_class"][label_name]
        print(
            f"{label_name}: precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print()
    print("Validation distribution:")
    print(val_df["error_type"].value_counts())
    print()
    print("Test distribution:")
    print(test_df["error_type"].value_counts())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SmallCNN(num_classes=len(LABEL_ORDER)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print_gpu_memory("before evaluation")
    validation_metrics = evaluate_split(model, val_df, label_to_index, device)
    test_metrics = evaluate_split(model, test_df, label_to_index, device)
    print_gpu_memory("after evaluation")

    print_summary("Validation", validation_metrics)
    print_summary("Test", test_metrics)

    output_metrics = {
        "model": "l2_arctic_del_sub_binary_cnn",
        "checkpoint_path": str(CHECKPOINT_PATH).replace("\\", "/"),
        "training_strategy": checkpoint.get("training_strategy", "weighted_random_sampler"),
        "loss": "cross_entropy_unweighted",
        "validation": {key: value for key, value in validation_metrics.items() if key != "rows"},
        "test": {key: value for key, value in test_metrics.items() if key != "rows"},
        "note": "Confidence is model confidence for predicted class, not pronunciation correctness.",
    }

    metrics_path = OUTPUT_DIR / "del_sub_binary_eval_metrics.json"
    confusion_matrix_path = OUTPUT_DIR / "del_sub_binary_confusion_matrix.csv"
    per_class_path = OUTPUT_DIR / "del_sub_binary_per_class_metrics.csv"
    misclassified_path = OUTPUT_DIR / "del_sub_binary_misclassified_examples.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output_metrics, file, indent=2, ensure_ascii=False)
    save_confusion_matrix_csv(test_metrics["confusion_matrix"], confusion_matrix_path)
    save_per_class_csv({"validation": validation_metrics, "test": test_metrics}, per_class_path)
    save_misclassified_examples(test_metrics["rows"], misclassified_path)

    print()
    print("Saved evaluation outputs:")
    for path in [metrics_path, confusion_matrix_path, per_class_path, misclassified_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
