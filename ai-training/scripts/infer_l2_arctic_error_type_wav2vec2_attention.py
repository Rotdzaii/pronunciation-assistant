from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_wav2vec2_attention.pt")
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
CONTEXT_SECONDS = 0.15
DROPOUT = 0.2
LABEL_ORDER = ["addition", "deletion", "substitution"]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor):
        weights = torch.softmax(self.attention_score(hidden_states).squeeze(-1), dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class Wav2Vec2AttentionClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        self.pooling = AttentionPooling(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(self.encoder.config.hidden_size, num_classes))

    def forward(self, input_values, attention_mask=None):
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        return self.classifier(self.pooling(outputs.last_hidden_state))


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
    return segment.astype(np.float32, copy=False), crop_start, crop_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, required=True)
    args = parser.parse_args()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].reset_index(drop=True)
    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"row-index must be between 0 and {len(df) - 1}")
    row = df.iloc[args.row_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    label_to_index = checkpoint.get("label_to_index", {label: index for index, label in enumerate(LABEL_ORDER)})
    index_to_label = {int(index): label for index, label in checkpoint.get("index_to_label", {}).items()} or {
        index: label for label, index in label_to_index.items()
    }
    model_name = checkpoint.get("model_name", MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2AttentionClassifier(model_name, len(label_to_index), checkpoint.get("freeze_encoder", True)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    segment, crop_start, crop_end = crop_segment(row, checkpoint.get("context_seconds", CONTEXT_SECONDS))
    inputs = feature_extractor([segment], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, return_attention_mask=True)
    attention_mask = inputs.get("attention_mask")
    with torch.no_grad():
        probs = torch.softmax(
            model(
                inputs["input_values"].to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            ),
            dim=1,
        )[0]
    pred_id = int(probs.argmax().item())
    pred = index_to_label[pred_id]
    print("speaker_id:", row.get("speaker_id", ""))
    print("utterance_id:", row.get("utterance_id", ""))
    print("crop_start_time:", f"{crop_start:.6f}")
    print("crop_end_time:", f"{crop_end:.6f}")
    print("ground_truth_error_type:", row["error_type"])
    print("predicted_error_type:", pred)
    print("confidence:", f"{float(probs[pred_id]):.6f}")
    print("class probabilities:")
    for label in LABEL_ORDER:
        print(f"- {label}: {float(probs[label_to_index[label]]):.6f}")
    print("is_correct:", pred == row["error_type"])
    print("Note: confidence is model confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
