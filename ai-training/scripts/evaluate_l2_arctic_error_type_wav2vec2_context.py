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
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")
MODEL_DIR = Path("ai-training/models")
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 4
NUM_WORKERS = 0
FREEZE_ENCODER = True
LABEL_ORDER = ["addition", "deletion", "substitution"]
CROP_MODES = {"original_segment": 0.0, "context_0_10": 0.10, "context_0_15": 0.15}


def checkpoint_path(crop_mode: str) -> Path:
    if crop_mode == "original_segment":
        return MODEL_DIR / "l2_arctic_error_type_wav2vec2_context_original_segment.pt"
    return MODEL_DIR / f"l2_arctic_error_type_wav2vec2_{crop_mode}.pt"


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


def crop_segment(row: pd.Series, context_seconds: float):
    audio, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    crop_start = max(0.0, float(row["start_time"]) - context_seconds)
    crop_end = min(duration, float(row["end_time"]) + context_seconds)
    segment = audio[int(crop_start * SAMPLE_RATE): min(len(audio), int(crop_end * SAMPLE_RATE))]
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
            "index": index,
        }


class Wav2Vec2EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
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
        return self.classifier(outputs.last_hidden_state.mean(dim=1))


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


def evaluate_split(model, dataframe, label_to_index, context_seconds, feature_extractor, device):
    loader = DataLoader(
        ErrorSegmentDataset(dataframe, label_to_index, context_seconds),
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


def save_confusion_matrix_csv(crop_mode: str, matrix: list[list[int]]):
    filename_mode = crop_mode.replace("context_", "", 1) if crop_mode.startswith("context_") else crop_mode
    path = OUTPUT_DIR / f"wav2vec2_context_{filename_mode}_confusion_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + LABEL_ORDER)
        for label, row in zip(LABEL_ORDER, matrix):
            writer.writerow([label] + row)
    return path


def save_per_class_csv(results: dict):
    path = OUTPUT_DIR / "wav2vec2_context_per_class_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["crop_mode", "split", "class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for crop_mode, run in results["runs"].items():
            for split in ["validation", "test"]:
                for label, metrics in run[split]["per_class"].items():
                    writer.writerow({"crop_mode": crop_mode, "split": split, "class": label, **metrics})


def save_comparison_csv(results: dict):
    path = OUTPUT_DIR / "wav2vec2_context_comparison.csv"
    fieldnames = ["crop_mode", "val_accuracy", "val_macro_f1", "val_weighted_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
    fieldnames += [f"test_{label}_f1" for label in LABEL_ORDER]
    rows = []
    for crop_mode, run in results["runs"].items():
        row = {
            "crop_mode": crop_mode,
            "val_accuracy": run["validation"]["accuracy"],
            "val_macro_f1": run["validation"]["macro_f1"],
            "val_weighted_f1": run["validation"]["weighted_f1"],
            "test_accuracy": run["test"]["accuracy"],
            "test_macro_f1": run["test"]["macro_f1"],
            "test_weighted_f1": run["test"]["weighted_f1"],
        }
        for label in LABEL_ORDER:
            row[f"test_{label}_f1"] = run["test"]["per_class"][label]["f1"]
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_misclassified(results: dict):
    path = OUTPUT_DIR / "wav2vec2_context_misclassified_examples.csv"
    rows = []
    for crop_mode, run in results["runs"].items():
        for row in run["test"]["rows"]:
            if not row["is_correct"]:
                rows.append({"crop_mode": crop_mode, **row})
    rows = sorted(rows, key=lambda row: row["confidence"], reverse=True)[:300]
    with path.open("w", encoding="utf-8", newline="") as file:
        if not rows:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["No misclassified examples found"])
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(crop_mode: str, split: str, metrics: dict):
    print()
    print(f"{crop_mode} {split}:")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    for label in LABEL_ORDER:
        item = metrics["per_class"][label]
        print(f"{label}: precision={item['precision']:.4f} recall={item['recall']:.4f} f1={item['f1']:.4f}")


def main():
    print("Note: confidence is classifier confidence, not pronunciation correctness.")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    label_to_index = {label: index for index, label in enumerate(LABEL_ORDER)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    results = {"model": "l2_arctic_error_type_wav2vec2_context", "runs": {}, "note": "Confidence is classifier confidence, not pronunciation correctness."}
    print_gpu_telemetry("before evaluation")
    for crop_mode, context_seconds in CROP_MODES.items():
        path = checkpoint_path(crop_mode)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device)
        model = Wav2Vec2EncoderClassifier(checkpoint.get("model_name", MODEL_NAME), len(LABEL_ORDER), checkpoint.get("freeze_encoder", FREEZE_ENCODER)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = evaluate_split(model, val_df, label_to_index, context_seconds, feature_extractor, device)
        test = evaluate_split(model, test_df, label_to_index, context_seconds, feature_extractor, device)
        results["runs"][crop_mode] = {
            "checkpoint_path": str(path).replace("\\", "/"),
            "context_seconds": context_seconds,
            "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
            "best_epoch": checkpoint.get("best_epoch"),
            "validation": {key: value for key, value in validation.items() if key != "rows"},
            "test": {key: value for key, value in test.items() if key != "rows"},
        }
        results["runs"][crop_mode]["validation"]["rows"] = validation["rows"]
        results["runs"][crop_mode]["test"]["rows"] = test["rows"]
        print_summary(crop_mode, "validation", validation)
        print_summary(crop_mode, "test", test)
        save_confusion_matrix_csv(crop_mode, test["confusion_matrix"])
    print_gpu_telemetry("after evaluation")
    best_macro = max(results["runs"], key=lambda mode: results["runs"][mode]["test"]["macro_f1"])
    best_addition = max(results["runs"], key=lambda mode: results["runs"][mode]["test"]["per_class"]["addition"]["f1"])
    results["best_by_test_macro_f1"] = best_macro
    results["best_by_test_addition_f1"] = best_addition
    print()
    print("Best crop mode by test macro F1:", best_macro)
    print("Best crop mode by test addition F1:", best_addition)
    metrics_path = OUTPUT_DIR / "wav2vec2_context_eval_metrics.json"
    compact = json.loads(json.dumps(results))
    for run in compact["runs"].values():
        run["validation"].pop("rows", None)
        run["test"].pop("rows", None)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(compact, file, indent=2, ensure_ascii=False)
    save_comparison_csv(compact)
    save_per_class_csv(compact)
    save_misclassified(results)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
