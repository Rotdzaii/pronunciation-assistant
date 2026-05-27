from pathlib import Path
import csv
import json
import subprocess

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_wav2vec2_attention.pt")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 4
NUM_WORKERS = 0
FREEZE_ENCODER = True
CONTEXT_SECONDS = 0.15
DROPOUT = 0.2
LABEL_ORDER = ["addition", "deletion", "substitution"]


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
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi warning: unavailable ({exc})")
        return
    print("nvidia-smi:", result.stdout.strip() or "no GPU rows returned")


def crop_segment(row: pd.Series):
    audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    crop_start = max(0.0, float(row["start_time"]) - CONTEXT_SECONDS)
    crop_end = min(duration, float(row["end_time"]) + CONTEXT_SECONDS)
    segment = audio[int(crop_start * SAMPLE_RATE): min(len(audio), int(crop_end * SAMPLE_RATE))]
    if len(segment) == 0:
        segment = np.zeros(MAX_LENGTH, dtype=np.float32)
    if len(segment) > MAX_LENGTH:
        segment = segment[:MAX_LENGTH]
    if len(segment) < MAX_LENGTH:
        segment = np.pad(segment, (0, MAX_LENGTH - len(segment)))
    return segment.astype(np.float32, copy=False)


class ErrorSegmentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, label_to_index: dict[str, int]):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        return {
            "audio": crop_segment(row),
            "label": torch.tensor(self.label_to_index[row["error_type"]], dtype=torch.long),
            "index": index,
        }


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor):
        scores = self.attention_score(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class Wav2Vec2AttentionClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        hidden_size = self.encoder.config.hidden_size
        self.pooling = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(hidden_size, num_classes))

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
        return self.classifier(self.pooling(outputs.last_hidden_state))


def collate_batch(batch: list[dict], feature_extractor: Wav2Vec2FeatureExtractor):
    inputs = feature_extractor(
        [item["audio"] for item in batch],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    labels = torch.stack([item["label"] for item in batch])
    indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    return inputs["input_values"], inputs.get("attention_mask"), labels, indices


def evaluate_split(model, dataframe, label_to_index, feature_extractor, device):
    loader = DataLoader(
        ErrorSegmentDataset(dataframe, label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: collate_batch(batch, feature_extractor),
    )
    model.eval()
    all_labels, all_preds, all_probs, all_indices = [], [], [], []
    with torch.no_grad():
        for input_values, attention_mask, labels, indices in tqdm(loader, desc="Evaluating"):
            input_values = input_values.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            probs = torch.softmax(model(input_values, attention_mask=attention_mask), dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_indices.extend(indices.cpu().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(LABEL_ORDER))), zero_division=0
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
    matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABEL_ORDER))))
    index_to_label = {index: label for label, index in label_to_index.items()}
    rows = []
    for row_index, true_id, pred_id, probs in zip(all_indices, all_labels, all_preds, all_probs):
        source = dataframe.iloc[row_index].to_dict()
        true_name = index_to_label[true_id]
        pred_name = index_to_label[pred_id]
        rows.append(
            {
                "speaker_id": source.get("speaker_id", ""),
                "utterance_id": source.get("utterance_id", ""),
                "audio_path": source.get("audio_path", ""),
                "start_time": source.get("start_time", ""),
                "end_time": source.get("end_time", ""),
                "true_error_type": true_name,
                "predicted_error_type": pred_name,
                "confidence": float(max(probs)),
                "is_correct": true_name == pred_name,
                "prob_addition": float(probs[label_to_index["addition"]]),
                "prob_deletion": float(probs[label_to_index["deletion"]]),
                "prob_substitution": float(probs[label_to_index["substitution"]]),
            }
        )
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "rows": rows,
    }


def save_confusion_matrix_csv(matrix: list[list[int]], path: Path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + LABEL_ORDER)
        for label, row in zip(LABEL_ORDER, matrix):
            writer.writerow([label] + row)


def save_per_class_csv(metrics: dict, path: Path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for split in ["validation", "test"]:
            for label, item in metrics[split]["per_class"].items():
                writer.writerow({"split": split, "class": label, **item})


def save_per_speaker(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    summaries = []
    for speaker_id, group in df.groupby("speaker_id"):
        total = len(group)
        correct = int(group["is_correct"].sum())
        summaries.append({"speaker_id": speaker_id, "total": total, "correct": correct, "accuracy": correct / max(total, 1)})
    pd.DataFrame(summaries).to_csv(path, index=False, encoding="utf-8")


def save_misclassified(rows: list[dict], path: Path, limit: int = 200):
    misses = sorted([row for row in rows if not row["is_correct"]], key=lambda row: row["confidence"], reverse=True)[:limit]
    with path.open("w", encoding="utf-8", newline="") as file:
        if not misses:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["No misclassified examples found"])
            return
        writer = csv.DictWriter(file, fieldnames=list(misses[0].keys()))
        writer.writeheader()
        writer.writerows(misses)


def print_summary(name: str, metrics: dict):
    print()
    print(f"{name} metrics:")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print("confusion_matrix:")
    print(pd.DataFrame(metrics["confusion_matrix"], index=LABEL_ORDER, columns=LABEL_ORDER))
    for label in LABEL_ORDER:
        item = metrics["per_class"][label]
        print(f"{label}: precision={item['precision']:.4f} recall={item['recall']:.4f} f1={item['f1']:.4f} support={item['support']}")


def print_per_speaker(rows: list[dict]):
    print()
    print("Per-speaker accuracy:")
    df = pd.DataFrame(rows)
    for speaker_id, group in df.groupby("speaker_id"):
        total = len(group)
        correct = int(group["is_correct"].sum())
        print(f"{speaker_id}: accuracy={correct / max(total, 1):.4f} correct={correct} total={total}")


def main():
    print("Note: confidence is classifier confidence, not pronunciation correctness.")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2AttentionClassifier(model_name, len(LABEL_ORDER), checkpoint.get("freeze_encoder", FREEZE_ENCODER)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print_gpu_telemetry("before evaluation")
    validation = evaluate_split(model, val_df, label_to_index, feature_extractor, device)
    test = evaluate_split(model, test_df, label_to_index, feature_extractor, device)
    print_gpu_telemetry("after evaluation")
    print_summary("Validation", validation)
    print_per_speaker(validation["rows"])
    print_summary("Test", test)
    print_per_speaker(test["rows"])
    output = {
        "model": "l2_arctic_error_type_wav2vec2_attention",
        "checkpoint_path": str(CHECKPOINT_PATH).replace("\\", "/"),
        "model_name": model_name,
        "context_seconds": checkpoint.get("context_seconds", CONTEXT_SECONDS),
        "freeze_encoder": checkpoint.get("freeze_encoder", FREEZE_ENCODER),
        "pooling": checkpoint.get("pooling", "attention"),
        "validation": {key: value for key, value in validation.items() if key != "rows"},
        "test": {key: value for key, value in test.items() if key != "rows"},
        "note": "Confidence is classifier confidence, not pronunciation correctness.",
    }
    metrics_path = OUTPUT_DIR / "wav2vec2_attention_eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    save_confusion_matrix_csv(test["confusion_matrix"], OUTPUT_DIR / "wav2vec2_attention_confusion_matrix.csv")
    save_per_class_csv({"validation": validation, "test": test}, OUTPUT_DIR / "wav2vec2_attention_per_class_metrics.csv")
    save_per_speaker(test["rows"], OUTPUT_DIR / "wav2vec2_attention_per_speaker_metrics.csv")
    save_misclassified(test["rows"], OUTPUT_DIR / "wav2vec2_attention_misclassified_examples.csv")
    print()
    print("Saved evaluation outputs:")
    for path in [
        metrics_path,
        OUTPUT_DIR / "wav2vec2_attention_confusion_matrix.csv",
        OUTPUT_DIR / "wav2vec2_attention_per_class_metrics.csv",
        OUTPUT_DIR / "wav2vec2_attention_per_speaker_metrics.csv",
        OUTPUT_DIR / "wav2vec2_attention_misclassified_examples.csv",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
