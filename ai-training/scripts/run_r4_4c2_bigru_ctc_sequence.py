"""Frozen R4-4C2 CNN+BiGRU model contract and synthetic/static checks.

This source intentionally contains no training invocation. The future locked
R4-4C2 runner must reuse this model without changing its architecture.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PHONE_COUNT = 40
CTC_BLANK_INDEX = 40
CLASS_COUNT = 41
INPUT_FEATURES = 96
GRU_HIDDEN_SIZE = 96


class WordBiGRUCTCModel(nn.Module):
    """R4-4B CNN with exactly one packed bidirectional GRU and CTC head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.bigru = nn.GRU(
            input_size=96,
            hidden_size=96,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.20)
        self.classifier = nn.Linear(192, CLASS_COUNT)

    @staticmethod
    def encoder_output_lengths(feature_lengths: torch.Tensor) -> torch.Tensor:
        # Only MaxPool2d((2, 2)) downsamples time; MaxPool2d((2, 1)) preserves it.
        return torch.div(feature_lengths, 2, rounding_mode="floor")

    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        sequence = encoded.mean(dim=2).transpose(1, 2)  # [B,T,96]
        output_lengths = self.encoder_output_lengths(feature_lengths)
        if torch.any(output_lengths <= 0):
            raise ValueError("All packed encoder lengths must be positive")
        if int(output_lengths.max()) > sequence.shape[1]:
            raise ValueError("Encoder output length exceeds the padded sequence")
        packed = pack_padded_sequence(
            sequence,
            output_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bigru(packed)
        recurrent, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        logits = self.classifier(self.dropout(recurrent))
        return logits, output_lengths


def parameter_report(model: nn.Module) -> dict[str, int]:
    groups = {
        "cnn_encoder": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "bigru": sum(parameter.numel() for parameter in model.bigru.parameters()),
        "ctc_head": sum(parameter.numel() for parameter in model.classifier.parameters()),
    }
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    groups["trainable"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return groups


def synthetic_static_checks() -> dict[str, Any]:
    torch.manual_seed(42)
    model = WordBiGRUCTCModel().eval()
    feature_lengths = torch.tensor([41, 31, 17], dtype=torch.long)
    features = torch.randn(3, 1, 64, 41)
    for index, length in enumerate(feature_lengths.tolist()):
        features[index, :, :, length:] = 0.0
    with torch.no_grad():
        logits, output_lengths = model(features, feature_lengths)
        single_logits, single_lengths = model(features[2:3, :, :, :17], feature_lengths[2:3])
        encoded = model.encoder(features).mean(dim=2).transpose(1, 2)
        packed = pack_padded_sequence(encoded, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = model.bigru(packed)
        recurrent, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=encoded.shape[1]
        )
        padded_values = torch.cat(
            [recurrent[index, length:] for index, length in enumerate(output_lengths.tolist()) if length < recurrent.shape[1]]
        )
    expected_lengths = [20, 15, 8]
    checks = {
        "frequency_mean_input_shape": list(encoded.shape),
        "bigru_output_shape": list(recurrent.shape),
        "logit_shape": list(logits.shape),
        "expected_logit_shape": [3, 20, 41],
        "encoder_output_lengths": output_lengths.tolist(),
        "expected_encoder_output_lengths": expected_lengths,
        "single_item_logit_shape": list(single_logits.shape),
        "single_item_output_lengths": single_lengths.tolist(),
        "packed_recurrent_padding_max_abs": float(padded_values.abs().max().item()),
        "parameter_count": parameter_report(model),
    }
    checks["status"] = "PASS" if (
        checks["logit_shape"] == checks["expected_logit_shape"]
        and checks["frequency_mean_input_shape"] == [3, 20, 96]
        and checks["bigru_output_shape"] == [3, 20, 192]
        and checks["encoder_output_lengths"] == expected_lengths
        and checks["single_item_output_lengths"] == [8]
        and checks["packed_recurrent_padding_max_abs"] == 0.0
        and checks["parameter_count"]["total"] == 198_761
    ) else "FAIL"
    if checks["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run synthetic/static checks only")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This frozen source is preregistration-only; use --self-test")
    print(json.dumps(synthetic_static_checks(), indent=2))


if __name__ == "__main__":
    main()
