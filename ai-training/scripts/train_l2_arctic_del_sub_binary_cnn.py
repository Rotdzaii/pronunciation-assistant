from pathlib import Path
import random
import subprocess

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
MODEL_OUTPUT = Path("ai-training/models/l2_arctic_del_sub_binary_cnn.pt")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)

BATCH_SIZE = 8
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
RANDOM_SEED = 42

LABEL_ORDER = ["deletion", "substitution"]
USE_WEIGHTED_SAMPLER = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_telemetry(stage: str):
    print()
    print(f"GPU telemetry - {stage}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Low VRAM usage can be normal for this small CNN; check CUDA availability and GPU utilization, not VRAM alone.")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"CUDA memory allocated MB: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"CUDA memory reserved MB: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f}")
        print(f"CUDA max memory allocated MB: {torch.cuda.max_memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"CUDA max memory reserved MB: {torch.cuda.max_memory_reserved(0) / 1024 / 1024:.2f}")

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
        print(f"nvidia-smi warning: unavailable ({exc})")
        return

    output = result.stdout.strip()
    print("nvidia-smi:", output if output else "no GPU rows returned")


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
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)


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


def build_sampler(train_df: pd.DataFrame):
    counts = train_df["error_type"].value_counts().to_dict()
    sample_weights = [1.0 / counts[row["error_type"]] for _, row in train_df.iterrows()]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating", leave=False):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(features).argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

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
    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(loader, desc="Training"):
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


def print_metrics(prefix: str, metrics: dict):
    print(f"{prefix}_acc={metrics['accuracy']:.4f} {prefix}_macro_f1={metrics['macro_f1']:.4f}")
    for label_name in LABEL_ORDER:
        print(f"  {prefix}_f1_{label_name}={metrics['per_class'][label_name]['f1']:.4f}")


def validate_audio_files(df: pd.DataFrame):
    missing = [path for path in df["audio_path"] if not Path(path).exists()]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        raise FileNotFoundError(f"Missing audio files: {len(missing)}\n{preview}")


def main():
    set_seed(RANDOM_SEED)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()

    required_columns = {"audio_path", "error_type", "split", "start_time", "end_time"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    validate_audio_files(df)

    print("Dataset shape:", df.shape)
    print()
    print("Error distribution:")
    print(df["error_type"].value_counts())
    print()
    print("Split distribution:")
    print(df["split"].value_counts())

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print()
        print(f"{split_name} distribution:")
        print(split_df["error_type"].value_counts())

    sampler = build_sampler(train_df) if USE_WEIGHTED_SAMPLER else None
    train_loader = DataLoader(
        ErrorSegmentDataset(train_df, label_to_index),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ErrorSegmentDataset(val_df, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ErrorSegmentDataset(test_df, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    model = SmallCNN(num_classes=len(LABEL_ORDER)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model device:", next(model.parameters()).device)
    print(f"Training strategy: {'WeightedRandomSampler + ' if USE_WEIGHTED_SAMPLER else ''}normal CrossEntropyLoss")
    print_gpu_telemetry("before training starts")

    best_val_macro_f1 = -1.0
    best_epoch = 0
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print_metrics("val", val_metrics)
        print_gpu_telemetry(f"after epoch {epoch + 1}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "max_seconds": MAX_SECONDS,
                    "task": "l2_arctic_del_sub_binary_cnn",
                    "training_strategy": "weighted_random_sampler" if USE_WEIGHTED_SAMPLER else "shuffle_only",
                    "loss": "cross_entropy_unweighted",
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "label_order": LABEL_ORDER,
                },
                MODEL_OUTPUT,
            )
            print(f"Saved best checkpoint to {MODEL_OUTPUT}")

    print()
    print("Loading best checkpoint for final evaluation...")
    checkpoint = torch.load(MODEL_OUTPUT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    print()
    print("Final validation metrics:")
    print_metrics("val", val_metrics)
    print()
    print("Final test metrics:")
    print_metrics("test", test_metrics)
    print_gpu_telemetry("after final evaluation")

    print()
    print("Training done.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Saved checkpoint: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
