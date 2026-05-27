from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_cnn_attention.pt")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CNN attention phone error type inference.")
    parser.add_argument("--row-index", type=int, required=True, help="Metadata CSV row index to infer.")
    return parser.parse_args()


def read_rows() -> list[dict[str, str]]:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def row_from_index(row_index: int) -> dict[str, str]:
    rows = read_rows()
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row-index {row_index} out of range; metadata has {len(rows)} rows")
    return rows[row_index]


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def row_to_feature(row: dict[str, str]) -> torch.Tensor:
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

    mel = librosa.feature.melspectrogram(y=audio.astype(np.float32, copy=False), sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict(row: dict[str, str], checkpoint: dict) -> dict:
    label_to_index = checkpoint.get("label_to_index", checkpoint.get("error_to_label"))
    index_to_label = {int(index): label for index, label in checkpoint.get("index_to_label", {}).items()}
    if not index_to_label:
        index_to_label = {index: label for label, index in label_to_index.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = checkpoint.get("config", {}).get("dropout", 0.2)
    model = SmallPronunciationCNNAttention(num_classes=len(label_to_index), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature = row_to_feature(row).to(device)
    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_error_type = index_to_label[predicted_index]
    class_probabilities = {
        index_to_label[index]: float(probabilities[index].item())
        for index in sorted(index_to_label)
    }

    return {
        "device": str(device),
        "predicted_error_type": predicted_error_type,
        "confidence": class_probabilities[predicted_error_type],
        "class_probabilities": class_probabilities,
    }


def print_result(row: dict[str, str], result: dict) -> None:
    ground_truth = row.get("error_type", "")
    print(f"speaker_id: {row.get('speaker_id', '')}")
    print(f"utterance_id: {row.get('utterance_id', '')}")
    print(f"ground_truth_error_type: {ground_truth}")
    print(f"predicted_error_type: {result['predicted_error_type']}")
    print(f"confidence: {result['confidence']:.6f}")
    print("class_probabilities:")
    print(json.dumps(result["class_probabilities"], indent=2, ensure_ascii=False))
    print(f"is_correct: {ground_truth == result['predicted_error_type']}")
    print("Note: confidence is model confidence, not pronunciation correctness.")


def main() -> None:
    args = parse_args()
    row = row_from_index(args.row_index)
    checkpoint = load_checkpoint()
    result = predict(row, checkpoint)
    print_result(row, result)


if __name__ == "__main__":
    main()
