from pathlib import Path
import csv
import json
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
MODEL_DIR = Path("ai-training/models")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)

BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42

LABEL_ORDER = ["addition", "deletion", "substitution"]


CONFIGS = [
    {
        "name": "no_balancing",
        "use_weighted_loss": False,
        "use_weighted_sampler": False,
    },
    {
        "name": "weighted_loss_only",
        "use_weighted_loss": True,
        "use_weighted_sampler": False,
    },
    {
        "name": "weighted_sampler_only",
        "use_weighted_loss": False,
        "use_weighted_sampler": True,
    },
    {
        "name": "weighted_loss_and_sampler",
        "use_weighted_loss": True,
        "use_weighted_sampler": True,
    },
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ErrorSegmentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_to_index: dict):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        audio_path = row["audio_path"]
        error_type = row["error_type"]
        label = self.label_to_index[error_type]

        start_time = float(row["start_time"])
        end_time = float(row["end_time"])

        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        start_sample = max(0, int(start_time * SAMPLE_RATE))
        end_sample = min(len(audio), int(end_time * SAMPLE_RATE))

        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            segment = np.zeros(MAX_LENGTH, dtype=np.float32)

        if len(segment) > MAX_LENGTH:
            segment = segment[:MAX_LENGTH]

        if len(segment) < MAX_LENGTH:
            segment = np.pad(segment, (0, MAX_LENGTH - len(segment)))

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(label, dtype=torch.long)

        return feature, target


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
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_class_weights(train_df: pd.DataFrame):
    counts = train_df["error_type"].value_counts().to_dict()
    total = len(train_df)
    num_classes = len(LABEL_ORDER)

    weights = []

    for label_name in LABEL_ORDER:
        count = counts.get(label_name, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(train_df: pd.DataFrame):
    counts = train_df["error_type"].value_counts().to_dict()

    sample_weights = []

    for _, row in train_df.iterrows():
        error_type = row["error_type"]
        sample_weights.append(1.0 / counts[error_type])

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
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )

    per_class = {}

    for index, label_name in enumerate(LABEL_ORDER):
        per_class[label_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def train_one_config(df: pd.DataFrame, config: dict):
    print("=" * 80)
    print(f"Balancing config: {config['name']}")
    print("=" * 80)

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Train distribution:")
    print(train_df["error_type"].value_counts())
    print()

    train_dataset = ErrorSegmentDataset(train_df, label_to_index)
    val_dataset = ErrorSegmentDataset(val_df, label_to_index)
    test_dataset = ErrorSegmentDataset(test_df, label_to_index)

    sampler = build_sampler(train_df) if config["use_weighted_sampler"] else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    class_weights = build_class_weights(train_df)

    print("Class weights:")
    for label_name, weight in zip(LABEL_ORDER, class_weights.tolist()):
        print(f"- {label_name}: {weight:.4f}")

    model = SmallCNN(num_classes=len(LABEL_ORDER)).to(device)

    if config["use_weighted_loss"]:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    checkpoint_path = MODEL_DIR / f"ablation_balance_{config['name']}.pt"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        for features, labels in tqdm(train_loader, desc="Training"):
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

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)

        val_metrics = evaluate_model(model, val_loader, device)

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        for label_name in LABEL_ORDER:
            print(f"  val_f1_{label_name}={val_metrics['per_class'][label_name]['f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "config": config,
                    "sample_rate": SAMPLE_RATE,
                    "n_mels": N_MELS,
                    "max_seconds": MAX_SECONDS,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    return {
        "config": config["name"],
        "best_epoch": best_epoch,
        "validation": val_metrics,
        "test": test_metrics,
        "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
    }


def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "ablation_balancing_results.json"
    csv_path = OUTPUT_DIR / "ablation_balancing_comparison.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    rows = []

    for result in results:
        row = {
            "config": result["config"],
            "best_epoch": result["best_epoch"],
            "val_accuracy": result["validation"]["accuracy"],
            "val_macro_f1": result["validation"]["macro_f1"],
            "test_accuracy": result["test"]["accuracy"],
            "test_macro_f1": result["test"]["macro_f1"],
        }

        for label_name in LABEL_ORDER:
            row[f"val_f1_{label_name}"] = result["validation"]["per_class"][label_name]["f1"]
            row[f"test_f1_{label_name}"] = result["test"]["per_class"][label_name]["f1"]

        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved CSV comparison to: {csv_path}")


def main():
    set_seed(RANDOM_SEED)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()

    print("Dataset shape:", df.shape)
    print("Error distribution:")
    print(df["error_type"].value_counts())
    print()
    print("Split distribution:")
    print(df["split"].value_counts())
    print()

    missing_audio = df[~df["audio_path"].apply(lambda path: Path(path).exists())]

    if len(missing_audio) > 0:
        raise FileNotFoundError(f"Missing audio rows: {len(missing_audio)}")

    results = []

    for config in CONFIGS:
        set_seed(RANDOM_SEED)
        result = train_one_config(df, config)
        results.append(result)

    save_results(results)

    print()
    print("Balancing ablation summary:")

    for result in results:
        print("-" * 60)
        print("config:", result["config"])
        print("val_macro_f1:", round(result["validation"]["macro_f1"], 4))
        print("test_macro_f1:", round(result["test"]["macro_f1"], 4))
        print("test addition F1:", round(result["test"]["per_class"]["addition"]["f1"], 4))
        print("test deletion F1:", round(result["test"]["per_class"]["deletion"]["f1"], 4))
        print("test substitution F1:", round(result["test"]["per_class"]["substitution"]["f1"], 4))


if __name__ == "__main__":
    main()