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
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
MODEL_DIR = Path("ai-training/models")
MODEL_NAME = "facebook/wav2vec2-base-960h"

SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 4
EPOCHS = 6
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42
FREEZE_ENCODER = True
POOLING = "mean"

LABEL_ORDER = ["addition", "deletion", "substitution"]
CROP_MODES = {
    "original_segment": 0.0,
    "context_0_10": 0.10,
    "context_0_15": 0.15,
}


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
    print("Low GPU utilization can happen because audio loading/preprocessing is CPU-bound and the encoder is frozen.")
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


def checkpoint_path(crop_mode: str) -> Path:
    if crop_mode == "original_segment":
        return MODEL_DIR / "l2_arctic_error_type_wav2vec2_context_original_segment.pt"
    return MODEL_DIR / f"l2_arctic_error_type_wav2vec2_{crop_mode}.pt"


def crop_segment(row: pd.Series, context_seconds: float):
    audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
    audio_duration = len(audio) / SAMPLE_RATE
    crop_start = max(0.0, float(row["start_time"]) - context_seconds)
    crop_end = min(audio_duration, float(row["end_time"]) + context_seconds)
    start_sample = max(0, int(crop_start * SAMPLE_RATE))
    end_sample = min(len(audio), int(crop_end * SAMPLE_RATE))
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        segment = np.zeros(MAX_LENGTH, dtype=np.float32)
    if len(segment) > MAX_LENGTH:
        segment = segment[:MAX_LENGTH]
    if len(segment) < MAX_LENGTH:
        segment = np.pad(segment, (0, MAX_LENGTH - len(segment)))
    return segment.astype(np.float32, copy=False)


class ErrorSegmentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_to_index: dict[str, int], context_seconds: float):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.context_seconds = context_seconds

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        return {
            "audio": crop_segment(row, self.context_seconds),
            "label": torch.tensor(self.label_to_index[row["error_type"]], dtype=torch.long),
        }


class Wav2Vec2EncoderClassifier(nn.Module):
    def __init__(self, encoder: Wav2Vec2Model, num_classes: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.encoder.config.hidden_size, num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


def collate_batch(batch: list[dict], feature_extractor: Wav2Vec2FeatureExtractor):
    inputs = feature_extractor(
        [item["audio"] for item in batch],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    return inputs["input_values"], inputs.get("attention_mask"), torch.stack([item["label"] for item in batch])


def build_sampler(train_df: pd.DataFrame):
    counts = train_df["error_type"].value_counts().to_dict()
    weights = [1.0 / counts[row["error_type"]] for _, row in train_df.iterrows()]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)


def evaluate_model(model, loader, device):
    model.eval()
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for input_values, attention_mask, labels in tqdm(loader, desc="Evaluating", leave=False):
            input_values = input_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            preds = model(input_values, attention_mask=attention_mask).argmax(dim=1)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_all, preds_all, labels=list(range(len(LABEL_ORDER))), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABEL_ORDER)
    }
    return {
        "accuracy": float(accuracy_score(labels_all, preds_all)),
        "macro_f1": float(f1_score(labels_all, preds_all, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_all, preds_all, average="weighted", zero_division=0)),
        "per_class": per_class,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for input_values, attention_mask, labels in tqdm(loader, desc="Training"):
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(input_values, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix}_accuracy={metrics['accuracy']:.4f} "
        f"{prefix}_macro_f1={metrics['macro_f1']:.4f} "
        f"{prefix}_weighted_f1={metrics['weighted_f1']:.4f}"
    )
    for label in LABEL_ORDER:
        print(f"  {prefix}_f1_{label}={metrics['per_class'][label]['f1']:.4f}")


def validate_audio_files(df: pd.DataFrame):
    missing = [path for path in df["audio_path"] if not Path(path).exists()]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        raise FileNotFoundError(f"Missing audio files: {len(missing)}\n{preview}")


def train_crop_mode(crop_mode: str, context_seconds: float, df: pd.DataFrame, feature_extractor, device):
    print()
    print("=" * 72)
    print("Crop mode:", crop_mode)
    print("Context seconds:", context_seconds)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print()
        print(f"{name} distribution:")
        print(split_df["error_type"].value_counts())

    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    collate_fn = lambda batch: collate_batch(batch, feature_extractor)
    train_loader = DataLoader(
        ErrorSegmentDataset(train_df, label_to_index, context_seconds),
        batch_size=BATCH_SIZE,
        sampler=build_sampler(train_df),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ErrorSegmentDataset(val_df, label_to_index, context_seconds),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    encoder = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model = Wav2Vec2EncoderClassifier(encoder, len(LABEL_ORDER), FREEZE_ENCODER).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_macro_f1 = -1.0
    best_epoch = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_path(crop_mode)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_gpu_telemetry(f"before training {crop_mode}")

    try:
        for epoch in range(EPOCHS):
            print()
            print(f"Epoch {epoch + 1}/{EPOCHS} - {crop_mode}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, device)
            print(f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
            print_metrics("val", val_metrics)
            improved = val_metrics["macro_f1"] > best_val_macro_f1
            print("checkpoint_improved:", improved)
            print_gpu_telemetry(f"after epoch {epoch + 1} {crop_mode}")
            if improved:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "classifier_state_dict": model.classifier.state_dict(),
                        "label_to_index": label_to_index,
                        "index_to_label": index_to_label,
                        "label_order": LABEL_ORDER,
                        "model_name": MODEL_NAME,
                        "sample_rate": SAMPLE_RATE,
                        "max_seconds": MAX_SECONDS,
                        "crop_mode": crop_mode,
                        "context_seconds": context_seconds,
                        "freeze_encoder": FREEZE_ENCODER,
                        "pooling": POOLING,
                        "task": "l2_arctic_phone_error_type_wav2vec2_context",
                        "best_val_macro_f1": best_val_macro_f1,
                        "best_epoch": best_epoch,
                        "config": {
                            "batch_size": BATCH_SIZE,
                            "epochs": EPOCHS,
                            "learning_rate": LEARNING_RATE,
                            "num_workers": NUM_WORKERS,
                            "random_seed": RANDOM_SEED,
                            "training_strategy": "weighted_random_sampler_only",
                            "loss": "cross_entropy_unweighted",
                        },
                    },
                    output_path,
                )
                print(f"Saved best checkpoint to {output_path}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA out of memory while training Wav2Vec2 context classifier.")
            print("Suggested next run: reduce BATCH_SIZE to 2 and keep FREEZE_ENCODER = True.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        raise

    print_gpu_telemetry(f"after crop mode {crop_mode}")
    print(f"Best epoch for {crop_mode}: {best_epoch}")
    print(f"Best validation macro F1 for {crop_mode}: {best_val_macro_f1:.4f}")
    print(f"Saved checkpoint: {output_path}")


def main():
    set_seed(RANDOM_SEED)
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()
    required = {"audio_path", "error_type", "split", "start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    validate_audio_files(df)
    print("Dataset shape:", df.shape)
    print("Error distribution:")
    print(df["error_type"].value_counts())
    print("Split distribution:")
    print(df["split"].value_counts())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    for crop_mode, context_seconds in CROP_MODES.items():
        train_crop_mode(crop_mode, context_seconds, df, feature_extractor, device)


if __name__ == "__main__":
    main()
