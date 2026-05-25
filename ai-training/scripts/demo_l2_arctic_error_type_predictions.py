from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import random
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_cnn_v2.pt")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/evaluation/v2_inference_demo_predictions.csv")

SAMPLE_RATE = 16000
MAX_SECONDS = 1
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64
SAMPLES_PER_CLASS = 5
RANDOM_SEED = 42
ERROR_TYPES = ("addition", "deletion", "substitution")


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


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_row(row: dict[str, str]) -> bool:
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


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        print("Checkpoint not found. Please train V2 first.")
        sys.exit(1)
    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def sample_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    random.seed(RANDOM_SEED)
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_class[row["error_type"]].append(row)

    sampled = []
    for error_type in ERROR_TYPES:
        candidates = rows_by_class[error_type]
        sampled.extend(random.sample(candidates, min(SAMPLES_PER_CLASS, len(candidates))))
    return sampled


def row_to_feature(row: dict[str, str]) -> torch.Tensor:
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
        y=audio.astype(np.float32, copy=False),
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def build_model(checkpoint: dict, device: torch.device) -> tuple[nn.Module, dict[int, str]]:
    error_to_label = checkpoint["error_to_label"]
    label_to_error = {label: error_type for error_type, label in error_to_label.items()}
    model = SmallPronunciationCNNV2(num_classes=len(error_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_to_error


def infer_row(row: dict[str, str], model: nn.Module, label_to_error: dict[int, str], device: torch.device) -> dict:
    feature = row_to_feature(row).to(device)
    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_label = int(torch.argmax(probabilities).item())
    predicted_error_type = label_to_error[predicted_label]
    confidence = float(probabilities[predicted_label].item())

    return {
        "dataset": row["dataset"],
        "speaker_id": row["speaker_id"],
        "utterance_id": row["utterance_id"],
        "audio_path": row["audio_path"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "ground_truth_error_type": row["error_type"],
        "predicted_error_type": predicted_error_type,
        "confidence": confidence,
        "is_correct": row["error_type"] == predicted_error_type,
    }


def write_predictions(predictions: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "dataset",
            "speaker_id",
            "utterance_id",
            "audio_path",
            "start_time",
            "end_time",
            "ground_truth_error_type",
            "predicted_error_type",
            "confidence",
            "is_correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def print_summary(predictions: list[dict]) -> None:
    total = len(predictions)
    correct = sum(1 for prediction in predictions if prediction["is_correct"])
    print(f"total demo samples: {total}")
    print(f"correct count: {correct}")
    print(f"accuracy on demo subset: {correct / total if total else 0.0:.4f}")
    print("per-class demo results:")
    counts = Counter(prediction["ground_truth_error_type"] for prediction in predictions)
    correct_by_class = Counter(
        prediction["ground_truth_error_type"]
        for prediction in predictions
        if prediction["is_correct"]
    )
    for error_type in ERROR_TYPES:
        count = counts[error_type]
        class_correct = correct_by_class[error_type]
        accuracy = class_correct / count if count else 0.0
        print(f"- {error_type}: correct={class_correct} total={count} accuracy={accuracy:.4f}")
    print(f"saved predictions: {OUTPUT_CSV}")
    print("note: confidence is model confidence for the predicted class, not pronunciation correctness.")


def main() -> None:
    checkpoint = load_checkpoint()
    rows = [row for row in read_rows() if valid_row(row)]
    sampled_rows = sample_rows(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_to_error = build_model(checkpoint, device)

    print(f"Using device: {device}")
    predictions = [infer_row(row, model, label_to_error, device) for row in sampled_rows]
    write_predictions(predictions)
    print_summary(predictions)


if __name__ == "__main__":
    main()
