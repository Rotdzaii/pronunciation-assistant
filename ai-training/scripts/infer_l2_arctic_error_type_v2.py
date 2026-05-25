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


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_cnn_v2.pt")

SAMPLE_RATE = 16000
MAX_SECONDS = 1
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local V2 phone error type inference for one audio segment.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--row-index", type=int, help="Metadata CSV row index to infer.")
    mode.add_argument("--audio-path", type=Path, help="Audio path to infer.")
    parser.add_argument("--start-time", type=float, default=None, help="Optional segment start time in seconds.")
    parser.add_argument("--end-time", type=float, default=None, help="Optional segment end time in seconds.")
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


def row_from_audio_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "dataset": "",
        "speaker_id": "",
        "utterance_id": "",
        "audio_path": str(args.audio_path),
        "start_time": "" if args.start_time is None else str(args.start_time),
        "end_time": "" if args.end_time is None else str(args.end_time),
        "error_type": "",
    }


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        print("Checkpoint not found. Please train V2 first.")
        sys.exit(1)
    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def load_audio_segment(audio_path: Path, start_time: float | None, end_time: float | None) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    offset = start_time if start_time is not None else 0.0
    duration = None
    if end_time is not None:
        if start_time is None:
            raise ValueError("--end-time requires --start-time")
        if end_time <= start_time:
            raise ValueError("--end-time must be greater than --start-time")
        duration = end_time - start_time

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=offset,
        duration=duration,
    )

    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]

    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))

    return audio.astype(np.float32, copy=False)


def parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def row_to_feature(row: dict[str, str]) -> torch.Tensor:
    start_time = parse_optional_float(row.get("start_time", ""))
    end_time = parse_optional_float(row.get("end_time", ""))
    audio = load_audio_segment(Path(row["audio_path"]), start_time, end_time)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict(row: dict[str, str], checkpoint: dict) -> dict:
    error_to_label = checkpoint["error_to_label"]
    label_to_error = {label: error_type for error_type, label in error_to_label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPronunciationCNNV2(num_classes=len(error_to_label)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature = row_to_feature(row).to(device)
    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_label = int(torch.argmax(probabilities).item())
    predicted_error_type = label_to_error[predicted_label]
    class_probabilities = {
        label_to_error[index]: float(probabilities[index].item())
        for index in sorted(label_to_error)
    }

    return {
        "device": str(device),
        "predicted_error_type": predicted_error_type,
        "confidence": class_probabilities[predicted_error_type],
        "class_probabilities": class_probabilities,
    }


def print_result(row: dict[str, str], result: dict) -> None:
    ground_truth = row.get("error_type", "")
    print(f"dataset: {row.get('dataset', '')}")
    print(f"speaker_id: {row.get('speaker_id', '')}")
    print(f"utterance_id: {row.get('utterance_id', '')}")
    print(f"audio_path: {row.get('audio_path', '')}")
    print(f"start_time: {row.get('start_time', '')}")
    print(f"end_time: {row.get('end_time', '')}")
    print(f"ground_truth_error_type: {ground_truth}")
    print(f"predicted_error_type: {result['predicted_error_type']}")
    print(f"confidence: {result['confidence']:.6f}")
    print("class_probabilities:")
    print(json.dumps(result["class_probabilities"], indent=2, ensure_ascii=False))
    print("interpretation: confidence is model confidence for the predicted class, not pronunciation correctness.")
    if ground_truth:
        print(f"is_correct: {ground_truth == result['predicted_error_type']}")


def main() -> None:
    args = parse_args()
    row = row_from_index(args.row_index) if args.row_index is not None else row_from_audio_args(args)
    checkpoint = load_checkpoint()
    result = predict(row, checkpoint)
    print_result(row, result)


if __name__ == "__main__":
    main()
