from pathlib import Path
import csv
import json

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
ADDITION_CHECKPOINT = Path("ai-training/models/l2_arctic_addition_binary_cnn.pt")
DEL_SUB_CHECKPOINT = Path("ai-training/models/l2_arctic_del_sub_binary_cnn.pt")
OUTPUT_DIR = Path("ai-training/datasets/l2-arctic/evaluation")

SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
BATCH_SIZE = 8
NUM_WORKERS = 0

STAGE1_LABEL_ORDER = ["addition", "non_addition"]
STAGE2_LABEL_ORDER = ["deletion", "substitution"]
FINAL_LABEL_ORDER = ["addition", "deletion", "substitution"]


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
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long), index


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


def print_gpu_memory(stage: str):
    print()
    print(f"GPU memory - {stage}")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"CUDA memory allocated MB: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f}")
        print(f"CUDA memory reserved MB: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f}")


def load_model(checkpoint_path: Path, num_classes: int, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SmallCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_split(stage1_model, stage2_model, dataframe: pd.DataFrame, device):
    final_label_to_index = {label: index for index, label in enumerate(FINAL_LABEL_ORDER)}
    final_index_to_label = {index: label for label, index in final_label_to_index.items()}
    loader = DataLoader(
        ErrorSegmentDataset(dataframe, final_label_to_index),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_labels = []
    all_predictions = []
    rows = []

    with torch.no_grad():
        for features, labels, indices in tqdm(loader, desc="Evaluating binary-stage pipeline"):
            features = features.to(device)
            stage1_probs = torch.softmax(stage1_model(features), dim=1)
            stage1_preds = stage1_probs.argmax(dim=1)
            stage2_probs = torch.softmax(stage2_model(features), dim=1)
            stage2_preds = stage2_probs.argmax(dim=1)

            labels_list = labels.cpu().numpy().tolist()
            indices_list = indices.cpu().numpy().tolist()
            stage1_probs_list = stage1_probs.cpu().numpy().tolist()
            stage1_preds_list = stage1_preds.cpu().numpy().tolist()
            stage2_probs_list = stage2_probs.cpu().numpy().tolist()
            stage2_preds_list = stage2_preds.cpu().numpy().tolist()

            for true_label, local_row_index, s1_probs, s1_pred, s2_probs, s2_pred in zip(
                labels_list,
                indices_list,
                stage1_probs_list,
                stage1_preds_list,
                stage2_probs_list,
                stage2_preds_list,
            ):
                stage1_label = STAGE1_LABEL_ORDER[s1_pred]
                if stage1_label == "addition":
                    final_label = "addition"
                    final_confidence = float(s1_probs[s1_pred])
                else:
                    final_label = STAGE2_LABEL_ORDER[s2_pred]
                    final_confidence = float(s2_probs[s2_pred])

                final_prediction = final_label_to_index[final_label]
                all_labels.append(true_label)
                all_predictions.append(final_prediction)

                row = dataframe.iloc[local_row_index].to_dict()
                true_name = final_index_to_label[true_label]
                rows.append(
                    {
                        "speaker_id": row.get("speaker_id", ""),
                        "utterance_id": row.get("utterance_id", ""),
                        "audio_path": row.get("audio_path", ""),
                        "start_time": row.get("start_time", ""),
                        "end_time": row.get("end_time", ""),
                        "true_error_type": true_name,
                        "stage1_prediction": stage1_label,
                        "stage1_confidence": float(s1_probs[s1_pred]),
                        "stage2_prediction": STAGE2_LABEL_ORDER[s2_pred] if stage1_label != "addition" else "",
                        "stage2_confidence": float(s2_probs[s2_pred]) if stage1_label != "addition" else "",
                        "predicted_error_type": final_label,
                        "confidence": final_confidence,
                        "is_correct": true_name == final_label,
                        "prob_stage1_addition": float(s1_probs[0]),
                        "prob_stage1_non_addition": float(s1_probs[1]),
                        "prob_stage2_deletion": float(s2_probs[0]) if stage1_label != "addition" else "",
                        "prob_stage2_substitution": float(s2_probs[1]) if stage1_label != "addition" else "",
                    }
                )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=list(range(len(FINAL_LABEL_ORDER))),
        zero_division=0,
    )
    per_class = {
        label_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label_name in enumerate(FINAL_LABEL_ORDER)
    }
    matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(len(FINAL_LABEL_ORDER))))

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_predictions, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "rows": rows,
    }


def save_confusion_matrix_csv(matrix: list[list[int]], output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + FINAL_LABEL_ORDER)
        for label_name, row in zip(FINAL_LABEL_ORDER, matrix):
            writer.writerow([label_name] + row)


def save_per_class_csv(metrics: dict, output_path: Path):
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["split", "class", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ["validation", "test"]:
            for class_name, class_metrics in metrics[split_name]["per_class"].items():
                writer.writerow({"split": split_name, "class": class_name, **class_metrics})


def save_misclassified_examples(rows: list[dict], output_path: Path, limit: int = 200):
    misclassified = [row for row in rows if not row["is_correct"]]
    misclassified = sorted(misclassified, key=lambda row: row["confidence"], reverse=True)[:limit]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        if not misclassified:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["No misclassified examples found"])
            return
        writer = csv.DictWriter(file, fieldnames=list(misclassified[0].keys()))
        writer.writeheader()
        writer.writerows(misclassified)


def print_summary(split_name: str, metrics: dict):
    print()
    print(f"{split_name} metrics:")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    for label_name in FINAL_LABEL_ORDER:
        class_metrics = metrics["per_class"][label_name]
        print(f"{label_name}_f1={class_metrics['f1']:.4f}")


def main():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(FINAL_LABEL_ORDER)].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print()
    print("Validation distribution:")
    print(val_df["error_type"].value_counts())
    print()
    print("Test distribution:")
    print(test_df["error_type"].value_counts())

    print()
    print("Note: Stage confidence is model confidence, not pronunciation correctness.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    stage1_model, stage1_checkpoint = load_model(ADDITION_CHECKPOINT, len(STAGE1_LABEL_ORDER), device)
    stage2_model, stage2_checkpoint = load_model(DEL_SUB_CHECKPOINT, len(STAGE2_LABEL_ORDER), device)

    print_gpu_memory("before evaluation")
    validation_metrics = evaluate_split(stage1_model, stage2_model, val_df, device)
    test_metrics = evaluate_split(stage1_model, stage2_model, test_df, device)
    print_gpu_memory("after evaluation")

    print_summary("Validation", validation_metrics)
    print_summary("Test", test_metrics)

    output_metrics = {
        "model": "l2_arctic_binary_stage_pipeline",
        "stage1_checkpoint_path": str(ADDITION_CHECKPOINT).replace("\\", "/"),
        "stage2_checkpoint_path": str(DEL_SUB_CHECKPOINT).replace("\\", "/"),
        "stage1_best_val_macro_f1": stage1_checkpoint.get("best_val_macro_f1"),
        "stage2_best_val_macro_f1": stage2_checkpoint.get("best_val_macro_f1"),
        "validation": {key: value for key, value in validation_metrics.items() if key != "rows"},
        "test": {key: value for key, value in test_metrics.items() if key != "rows"},
        "note": "Stage confidence is model confidence, not pronunciation correctness.",
    }

    metrics_path = OUTPUT_DIR / "binary_stage_pipeline_eval_metrics.json"
    confusion_matrix_path = OUTPUT_DIR / "binary_stage_pipeline_confusion_matrix.csv"
    per_class_path = OUTPUT_DIR / "binary_stage_pipeline_per_class_metrics.csv"
    misclassified_path = OUTPUT_DIR / "binary_stage_pipeline_misclassified_examples.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output_metrics, file, indent=2, ensure_ascii=False)
    save_confusion_matrix_csv(test_metrics["confusion_matrix"], confusion_matrix_path)
    save_per_class_csv({"validation": validation_metrics, "test": test_metrics}, per_class_path)
    save_misclassified_examples(test_metrics["rows"], misclassified_path)

    print()
    print("Saved evaluation outputs:")
    for path in [metrics_path, confusion_matrix_path, per_class_path, misclassified_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
