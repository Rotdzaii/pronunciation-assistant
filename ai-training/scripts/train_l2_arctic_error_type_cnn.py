from collections import Counter
from pathlib import Path
import csv

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv")
MODEL_OUTPUT = Path("ai-training/models/l2_arctic_error_type_cnn.pt")

SAMPLE_RATE = 16000
MAX_SECONDS = 1
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
MIN_ROWS_PER_CLASS = 50


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

        return feature, target


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
        raise FileNotFoundError(f"Classification CSV not found: {METADATA_CSV}")

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def valid_segment(row: dict[str, str]) -> bool:
    try:
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
    except ValueError:
        return False

    return Path(row["audio_path"]).exists() and start_time >= 0 and end_time > start_time


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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main():
    rows = [row for row in read_rows() if valid_segment(row)]
    counts = Counter(row["error_type"] for row in rows)
    eligible_classes = sorted(
        error_type for error_type, count in counts.items() if count >= MIN_ROWS_PER_CLASS
    )

    print("Rows by error_type:")
    for error_type, count in sorted(counts.items()):
        print(f"- {error_type}: {count}")

    if len(eligible_classes) < 2:
        print("Skipping training: fewer than 2 error classes have at least 50 valid rows.")
        return

    rows = [row for row in rows if row["error_type"] in eligible_classes]
    error_to_label = {error_type: index for index, error_type in enumerate(eligible_classes)}

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]

    if not train_rows or not val_rows:
        print("Skipping training: train or validation split is empty.")
        return

    print("Error labels:")
    for error_type, label in error_to_label.items():
        print(f"- {error_type}: {label}")

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")

    train_loader = DataLoader(
        ErrorTypeDataset(train_rows, error_to_label),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        ErrorTypeDataset(val_rows, error_to_label),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = SmallPronunciationCNN(num_classes=len(error_to_label)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "error_to_label": error_to_label,
                    "sample_rate": SAMPLE_RATE,
                    "max_seconds": MAX_SECONDS,
                    "n_mels": N_MELS,
                    "task": "l2_arctic_phone_error_type_classification",
                },
                MODEL_OUTPUT,
            )
            print(f"Saved model to {MODEL_OUTPUT}")

    print()
    print("Training done.")
    print("Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()
