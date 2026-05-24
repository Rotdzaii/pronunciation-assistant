from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_speakers_metadata.csv")
MODEL_OUTPUT = Path("ai-training/models/l2_arctic_vietnamese_speaker_cnn.pt")

SAMPLE_RATE = 16000
MAX_SECONDS = 5
MAX_LENGTH = SAMPLE_RATE * MAX_SECONDS
N_MELS = 64

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


class L2ArcticSpeakerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, speaker_to_label: dict[str, int]):
        self.dataframe = dataframe.reset_index(drop=True)
        self.speaker_to_label = speaker_to_label

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        audio_path = row["audio_path"]
        speaker_id = row["speaker_id"]
        label = self.speaker_to_label[speaker_id]

        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]

        if len(audio) < MAX_LENGTH:
            pad_width = MAX_LENGTH - len(audio)
            audio = np.pad(audio, (0, pad_width))

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(label, dtype=torch.long)

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
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def validate_audio_files(df: pd.DataFrame):
    missing_files = []

    for path in df["audio_path"]:
        if not Path(path).exists():
            missing_files.append(path)

    if missing_files:
        preview = "\n".join(f"- {path}" for path in missing_files[:20])
        raise FileNotFoundError(
            "Some audio files are missing.\n"
            f"Missing count: {len(missing_files)}\n"
            f"{preview}"
        )


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

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


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

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def main():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)

    validate_audio_files(df)

    speakers = sorted(df["speaker_id"].unique().tolist())
    speaker_to_label = {speaker_id: index for index, speaker_id in enumerate(speakers)}

    print("Speakers:")
    for speaker_id, label in speaker_to_label.items():
        print(f"- {speaker_id}: {label}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    print()
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    train_dataset = L2ArcticSpeakerDataset(train_df, speaker_to_label)
    val_dataset = L2ArcticSpeakerDataset(val_df, speaker_to_label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = SmallPronunciationCNN(num_classes=len(speakers)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

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
                    "speaker_to_label": speaker_to_label,
                    "sample_rate": SAMPLE_RATE,
                    "max_seconds": MAX_SECONDS,
                    "n_mels": N_MELS,
                    "task": "l2_arctic_vietnamese_speaker_classification",
                },
                MODEL_OUTPUT,
            )

            print(f"Saved model to {MODEL_OUTPUT}")

    print()
    print("Training done.")
    print("Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()