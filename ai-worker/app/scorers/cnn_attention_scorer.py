from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.contracts.ai_result_contract import build_ai_result, estimate_demo_score


SAMPLE_RATE = 16000
N_MELS = 64
MAX_SECONDS = 1.0
MAX_LENGTH = int(SAMPLE_RATE * MAX_SECONDS)
LABEL_ORDER = ["addition", "deletion", "substitution"]
SCORER_METADATA = {
    "name": "cnn_attention",
    "type": "phone_error_classifier",
    "version": "cnn_attention_selected_baseline",
}
DEFAULT_METADATA = {
    "model_output_is_scoring": False,
    "alignment_used": False,
    "gop_used": False,
    "hybrid_used": False,
}


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[3] / "ai-training" / "models" / "l2_arctic_error_type_cnn_attention.pt"


class CNNAttentionScorerError(RuntimeError):
    """Raised when CNN Attention inference cannot produce a result."""


def _load_torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention scorer requires torch. Install the AI training/inference dependencies before using "
            "SCORER_MODE=cnn_attention."
        ) from exc
    return torch, nn


torch, nn = _load_torch_modules()


class TemporalAttentionPooling(nn.Module):
    """Matches ai-training/scripts/train_l2_arctic_error_type_cnn_attention.py."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, feature_map: Any) -> tuple[Any, Any]:
        sequence = feature_map.mean(dim=2).transpose(1, 2)
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SmallPronunciationCNNAttention(nn.Module):
    """Minimal worker copy of the selected CNN Attention architecture."""

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.attention = TemporalAttentionPooling(channels=96)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(96, num_classes))

    def forward(self, x: Any) -> Any:
        feature_map = self.features(x)
        pooled, _ = self.attention(feature_map)
        return self.classifier(pooled)


def _resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    configured = checkpoint_path or os.getenv("CNN_ATTENTION_CHECKPOINT_PATH")
    return Path(configured).expanduser() if configured else default_checkpoint_path()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise CNNAttentionScorerError(
            "CNN Attention checkpoint not found. Expected local checkpoint at "
            f"{checkpoint_path}. Set CNN_ATTENTION_CHECKPOINT_PATH to override. "
            "Checkpoint files are local artifacts and must not be committed."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CNNAttentionScorerError(f"Invalid CNN Attention checkpoint format: {checkpoint_path}")
    return checkpoint


def _index_to_label(checkpoint: dict[str, Any]) -> dict[int, str]:
    raw_index_to_label = checkpoint.get("index_to_label") or {}
    index_to_label = {int(index): str(label) for index, label in raw_index_to_label.items()}
    if index_to_label:
        return index_to_label

    label_to_index = checkpoint.get("label_to_index") or checkpoint.get("error_to_label") or {}
    if label_to_index:
        return {int(index): str(label) for label, index in label_to_index.items()}

    return {index: label for index, label in enumerate(LABEL_ORDER)}


def _download_audio_to_temp(audio_url: str) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise CNNAttentionScorerError("Downloading audio_url requires requests.") from exc

    response = requests.get(audio_url, timeout=30)
    response.raise_for_status()
    suffix = Path(urlparse(audio_url).path).suffix or ".audio"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    try:
        temp_file.write(response.content)
    finally:
        temp_file.close()
    return temp_path


def _audio_path_from_job(job: dict[str, Any]) -> tuple[Path, bool]:
    audio_path = str(job.get("audio_path") or "").strip()
    if audio_path:
        return Path(audio_path), False

    audio_url = str(job.get("audio_url") or "").strip()
    if not audio_url:
        raise CNNAttentionScorerError("CNN Attention scorer requires job.audio_path or job.audio_url.")

    parsed = urlparse(audio_url)
    if parsed.scheme in {"http", "https"}:
        return _download_audio_to_temp(audio_url), True
    return Path(audio_url), False


def _feature_from_audio_path(audio_path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention scorer requires librosa and numpy for log-mel preprocessing."
        ) from exc

    audio, sample_rate = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    original_samples = int(len(audio))
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    if len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))

    audio = audio.astype(np.float32, copy=False)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    feature = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return feature, {
        "sample_rate": sample_rate,
        "target_sample_rate": SAMPLE_RATE,
        "original_duration_seconds": round(original_samples / SAMPLE_RATE, 3),
        "max_seconds": MAX_SECONDS,
        "n_mels": N_MELS,
        "clip_level_inference": True,
        "segment_source": "full_audio_first_second",
    }


def predict_audio(audio_path: str | Path, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    checkpoint_file = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    index_to_label = _index_to_label(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    model = SmallPronunciationCNNAttention(num_classes=len(index_to_label), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature, audio_metadata = _feature_from_audio_path(Path(audio_path))
    with torch.no_grad():
        logits = model(feature.to(device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_error_type = index_to_label[predicted_index]
    class_probabilities = {
        index_to_label[index]: float(probabilities[index].item())
        for index in sorted(index_to_label)
    }

    return {
        "predicted_error_type": predicted_error_type,
        "class_probabilities": class_probabilities,
        "diagnosis_confidence": class_probabilities[predicted_error_type],
        "device": str(device),
        "checkpoint_path": str(checkpoint_file),
        "audio": audio_metadata,
    }


def score_pronunciation(job: dict[str, Any], confidence_threshold: float | None = None) -> dict[str, Any]:
    temp_audio_path: Path | None = None
    try:
        audio_path, is_temp = _audio_path_from_job(job)
        temp_audio_path = audio_path if is_temp else None
        prediction = predict_audio(audio_path)
    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)

    demo_score = estimate_demo_score(
        prediction["predicted_error_type"],
        prediction["diagnosis_confidence"],
    )
    result = build_ai_result(
        score=demo_score["score"],
        problem_phonemes=[],
        predicted_error_type=prediction["predicted_error_type"],
        class_probabilities=prediction["class_probabilities"],
        diagnosis_confidence=prediction["diagnosis_confidence"],
        scorer=SCORER_METADATA,
        metadata={
            **DEFAULT_METADATA,
            "is_demo_score": demo_score["is_demo_score"],
            "score_note": demo_score["score_note"],
            "confidence_threshold": confidence_threshold,
            "device": prediction["device"],
            "checkpoint_path": prediction["checkpoint_path"],
            "audio": prediction["audio"],
            "limitation": (
                "No forced alignment is used yet. CNN Attention is currently run as clip-level demo "
                "inference over the first/whole submitted audio segment, not final phone-localized diagnosis."
            ),
        },
    )
    result["feedback"]["diagnosis"] = result["diagnosis"]
    result["feedback"]["scorer"] = result["scorer"]
    result["feedback"]["metadata"] = result["metadata"]
    return result
