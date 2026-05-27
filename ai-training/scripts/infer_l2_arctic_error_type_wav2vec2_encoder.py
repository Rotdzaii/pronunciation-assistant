from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


METADATA_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv")
CHECKPOINT_PATH = Path("ai-training/models/l2_arctic_error_type_wav2vec2_encoder.pt")

MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
LABEL_ORDER = ["addition", "deletion", "substitution"]


class Wav2Vec2EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.encoder.config.hidden_size, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        else:
            encoder_outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        return self.classifier(encoder_outputs.last_hidden_state.mean(dim=1))


def load_segment(row: pd.Series):
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
    return segment.astype(np.float32, copy=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    df = df[df["error_type"].isin(LABEL_ORDER)].reset_index(drop=True)
    if args.row_index < 0 or args.row_index >= len(df):
        raise IndexError(f"row-index must be between 0 and {len(df) - 1}")

    row = df.iloc[args.row_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    label_to_index = checkpoint.get("label_to_index", {label: index for index, label in enumerate(LABEL_ORDER)})
    index_to_label = {int(index): label for index, label in checkpoint.get("index_to_label", {}).items()}
    if not index_to_label:
        index_to_label = {index: label for label, index in label_to_index.items()}

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2EncoderClassifier(
        model_name=model_name,
        num_classes=len(label_to_index),
        freeze_encoder=checkpoint.get("freeze_encoder", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    segment = load_segment(row)
    inputs = feature_extractor(
        [segment],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_values, attention_mask=attention_mask), dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    predicted_error_type = index_to_label[predicted_index]
    ground_truth = row["error_type"]
    print("speaker_id:", row.get("speaker_id", ""))
    print("utterance_id:", row.get("utterance_id", ""))
    print("ground_truth_error_type:", ground_truth)
    print("predicted_error_type:", predicted_error_type)
    print("confidence:", f"{float(probabilities[predicted_index]):.6f}")
    print("class probabilities:")
    for label in LABEL_ORDER:
        print(f"- {label}: {float(probabilities[label_to_index[label]]):.6f}")
    print("is_correct:", predicted_error_type == ground_truth)
    print("Note: confidence is model confidence, not pronunciation correctness.")


if __name__ == "__main__":
    main()
