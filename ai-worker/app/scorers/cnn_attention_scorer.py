from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.alignment.audio_preparation import PreparedMfaAudio, prepare_audio_for_mfa, validate_prepared_mfa_wav
from app.contracts.ai_result_contract import build_ai_result, estimate_demo_score
from app.contracts.alignment_contract import AlignmentError, FALLBACK_ALIGNMENT_NOTE, get_alignment_segments
from app.phonetics.canonicalization import (
    build_phone_results as _build_phone_results,
    canonicalize_phones as _canonicalize_phones,
    check_inventory_guard as _check_inventory_guard,
    compute_reliability as _compute_reliability,
)


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
CONTEXT_SCORER_METADATA = {
    "name": "cnn_attention_context",
    "type": "phone_error_classifier",
    "version": "cnn_attention_context_0_10_phase2_candidate",
}
DEFAULT_METADATA = {
    "model_output_is_scoring": False,
    "alignment_used": False,
    "gop_used": False,
    "hybrid_used": False,
}
SENSITIVE_ALIGNMENT_METADATA_KEYS = {
    "audio_path",
    "checkpoint_path",
    "local_path",
    "mfa_output_dir",
    "mfa_temp_dir",
    "temp_dir",
    "textgrid_path",
    "debug_artifact_dir",
}


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[3] / "ai-training" / "models" / "l2_arctic_error_type_cnn_attention.pt"


def default_context_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[3] / "ai-training" / "models" / "l2_arctic_cnn_attention_context_0_10.pt"


class CNNAttentionScorerError(RuntimeError):
    """Raised when CNN Attention inference cannot produce a result."""


def _load_torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention scorer requires torch. Install the AI training/inference dependencies before using "
            "SCORER_MODE=cnn_attention or SCORER_MODE=cnn_attention_context."
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


def _resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    *,
    env_var_name: str = "CNN_ATTENTION_CHECKPOINT_PATH",
    default_path: Path | None = None,
) -> Path:
    configured = checkpoint_path or os.getenv(env_var_name)
    return Path(configured).expanduser() if configured else default_path or default_checkpoint_path()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise CNNAttentionScorerError(
            "CNN Attention checkpoint not found. Set CNN_ATTENTION_CHECKPOINT_PATH to a valid local checkpoint. "
            "Checkpoint files are local artifacts and must not be committed."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CNNAttentionScorerError(
            "Invalid CNN Attention checkpoint format. Set CNN_ATTENTION_CHECKPOINT_PATH to a valid local checkpoint."
        )
    return checkpoint


def _load_model(checkpoint_path: str | Path | None = None) -> tuple[Any, dict[int, str], Path, Any]:
    checkpoint_file = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    index_to_label = _index_to_label(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    model = SmallPronunciationCNNAttention(num_classes=len(index_to_label), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, index_to_label, checkpoint_file, device


def _load_context_model(checkpoint_path: str | Path | None = None) -> tuple[Any, dict[int, str], Path, Any]:
    checkpoint_file = _resolve_checkpoint_path(
        checkpoint_path,
        env_var_name="CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH",
        default_path=default_context_checkpoint_path(),
    )
    try:
        checkpoint = _load_checkpoint(checkpoint_file)
    except CNNAttentionScorerError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention context checkpoint not found or invalid. Set CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH to a "
            "valid local checkpoint. "
            "Checkpoint files are local artifacts and must not be committed."
        ) from exc

    index_to_label = _index_to_label(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    model = SmallPronunciationCNNAttention(num_classes=len(index_to_label), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, index_to_label, checkpoint_file, device


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


def _load_prepared_wav(audio_path: str | Path) -> tuple[Any, int]:
    """Load a verified MFA WAV once, without allowing codec fallback decoding."""

    try:
        import numpy as np
    except ImportError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention scorer requires numpy for prepared WAV loading."
        ) from exc

    path = Path(audio_path).expanduser()
    try:
        validate_prepared_mfa_wav(path)
        with wave.open(str(path), "rb") as wav_file:
            raw_frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
    except AlignmentError as exc:
        raise CNNAttentionScorerError("CNN segment inference requires a prepared 16 kHz mono PCM WAV.") from exc
    except (OSError, wave.Error) as exc:
        raise CNNAttentionScorerError("Prepared WAV could not be loaded for CNN segment inference.") from exc

    audio = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise CNNAttentionScorerError("Prepared WAV has no samples for CNN segment inference.")
    return audio, sample_rate


def _feature_from_waveform(
    waveform: Any,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise CNNAttentionScorerError(
            "CNN Attention scorer requires librosa and numpy for log-mel preprocessing."
        ) from exc

    audio = np.asarray(waveform, dtype=np.float32)
    full_duration_seconds = len(audio) / SAMPLE_RATE if SAMPLE_RATE else 0.0
    segment_start = max(float(start_time or 0.0), 0.0)
    segment_end = float(end_time) if end_time is not None else full_duration_seconds
    segment_end = max(segment_end, segment_start)
    if start_time is not None or end_time is not None:
        start_sample = min(int(segment_start * SAMPLE_RATE), len(audio))
        end_sample = min(int(segment_end * SAMPLE_RATE), len(audio))
        audio = audio[start_sample:end_sample]

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
        "sample_rate": SAMPLE_RATE,
        "target_sample_rate": SAMPLE_RATE,
        "original_duration_seconds": round(original_samples / SAMPLE_RATE, 3),
        "full_duration_seconds": round(full_duration_seconds, 3),
        "max_seconds": MAX_SECONDS,
        "n_mels": N_MELS,
        "clip_level_inference": start_time is None and end_time is None,
        "segment_source": "bounded_segment" if start_time is not None or end_time is not None else "full_audio_first_second",
        "start_time": round(segment_start, 3) if start_time is not None or end_time is not None else None,
        "end_time": round(segment_end, 3) if start_time is not None or end_time is not None else None,
    }


def _feature_from_audio_path(
    audio_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    waveform, _ = _load_prepared_wav(audio_path)
    return _feature_from_waveform(waveform, start_time, end_time)


def _predict_with_model(
    model: Any,
    index_to_label: dict[int, str],
    device: Any,
    checkpoint_file: Path,
    audio_path: str | Path,
    start_time: float | None = None,
    end_time: float | None = None,
    waveform: Any | None = None,
) -> dict[str, Any]:
    if waveform is None:
        waveform, _ = _load_prepared_wav(audio_path)
    feature, audio_metadata = _feature_from_waveform(waveform, start_time, end_time)
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


def _context_mode() -> str:
    return os.getenv("CNN_ATTENTION_CONTEXT_MODE", "context_0_10").strip() or "context_0_10"


def _context_seconds(env_var_name: str, default: float) -> float:
    raw_value = os.getenv(env_var_name)
    if raw_value in {None, ""}:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise CNNAttentionScorerError(f"{env_var_name} must be a number") from exc
    if value < 0:
        raise CNNAttentionScorerError(f"{env_var_name} must be greater than or equal to 0")
    return value


def _context_config() -> dict[str, Any]:
    mode = _context_mode()
    left_seconds = _context_seconds("CNN_ATTENTION_CONTEXT_LEFT_SECONDS", 0.10)
    right_seconds = _context_seconds("CNN_ATTENTION_CONTEXT_RIGHT_SECONDS", 0.10)
    if mode != "context_0_10":
        raise CNNAttentionScorerError(
            f"Unsupported CNN_ATTENTION_CONTEXT_MODE={mode!r}. The Phase 2 selected mode is context_0_10."
        )
    return {
        "context_mode": mode,
        "context_used": True,
        "context_left_seconds": left_seconds,
        "context_right_seconds": right_seconds,
    }


def _audio_duration_seconds(audio_path: str | Path) -> float:
    audio, _ = _load_prepared_wav(audio_path)
    return len(audio) / SAMPLE_RATE if SAMPLE_RATE else 0.0


def _context_crop_metadata_for_duration(
    audio_duration: float,
    segment_start_time: float,
    segment_end_time: float,
    context_config: dict[str, Any],
) -> dict[str, Any]:
    segment_start = max(float(segment_start_time), 0.0)
    segment_end = max(float(segment_end_time), segment_start)
    crop_start = max(0.0, segment_start - float(context_config["context_left_seconds"]))
    crop_end = min(audio_duration, segment_end + float(context_config["context_right_seconds"]))
    crop_end = max(crop_end, crop_start)
    return {
        **context_config,
        "segment_start_time": round(segment_start, 3),
        "segment_end_time": round(segment_end, 3),
        "crop_start_time": round(crop_start, 3),
        "crop_end_time": round(crop_end, 3),
        "audio_duration_seconds": round(audio_duration, 3),
    }


def _context_crop_metadata(
    audio_path: str | Path,
    segment_start_time: float,
    segment_end_time: float,
    context_config: dict[str, Any],
) -> dict[str, Any]:
    return _context_crop_metadata_for_duration(
        _audio_duration_seconds(audio_path),
        segment_start_time,
        segment_end_time,
        context_config,
    )


def _predict_context_with_model(
    model: Any,
    index_to_label: dict[int, str],
    device: Any,
    checkpoint_file: Path,
    audio_path: str | Path,
    start_time: float,
    end_time: float,
    context_config: dict[str, Any],
    waveform: Any | None = None,
    audio_duration: float | None = None,
) -> dict[str, Any]:
    if waveform is None:
        waveform, _ = _load_prepared_wav(audio_path)
    duration = audio_duration if audio_duration is not None else len(waveform) / SAMPLE_RATE
    context_metadata = _context_crop_metadata_for_duration(duration, start_time, end_time, context_config)
    prediction = _predict_with_model(
        model,
        index_to_label,
        device,
        checkpoint_file,
        audio_path,
        context_metadata["crop_start_time"],
        context_metadata["crop_end_time"],
        waveform,
    )
    prediction["audio"]["context"] = context_metadata
    return prediction


def predict_audio(audio_path: str | Path, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    model, index_to_label, checkpoint_file, device = _load_model(checkpoint_path)
    return _predict_with_model(model, index_to_label, device, checkpoint_file, audio_path)


def predict_segment(
    audio_path: str | Path,
    start_time: float,
    end_time: float,
    checkpoint_path: str | Path | None = None,
    phone: str | None = None,
    word: str | None = None,
) -> dict[str, Any]:
    model, index_to_label, checkpoint_file, device = _load_model(checkpoint_path)
    prediction = _predict_with_model(model, index_to_label, device, checkpoint_file, audio_path, start_time, end_time)
    return _format_segment_prediction(prediction, start_time, end_time, phone=phone, word=word)


def _format_segment_prediction(
    prediction: dict[str, Any],
    start_time: float,
    end_time: float,
    *,
    phone: str | None = None,
    word: str | None = None,
    segment_type: str | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    result = {
        "index": index,
        "type": segment_type,
        "phone": phone,
        "word": word,
        "start": round(float(start_time), 3),
        "end": round(float(end_time), 3),
        "predicted_error_type": prediction["predicted_error_type"],
        "class_probabilities": prediction["class_probabilities"],
        "diagnosis_confidence": prediction["diagnosis_confidence"],
        "confidence_note": "Classifier confidence, not pronunciation correctness.",
    }
    context_metadata = (prediction.get("audio") or {}).get("context")
    if context_metadata:
        result["context"] = context_metadata
    return result


def predict_segments(
    audio_path: str | Path,
    segments: list[dict[str, Any]],
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    model, index_to_label, checkpoint_file, device = _load_model(checkpoint_path)
    waveform, _ = _load_prepared_wav(audio_path)
    predictions = []
    for fallback_index, segment in enumerate(segments):
        start_time = float(segment.get("start") or 0.0)
        end_time = float(segment.get("end") if segment.get("end") is not None else start_time)
        if end_time <= start_time:
            end_time = start_time + MAX_SECONDS
        prediction = _predict_with_model(
            model,
            index_to_label,
            device,
            checkpoint_file,
            audio_path,
            start_time,
            end_time,
            waveform,
        )
        predictions.append(
            _format_segment_prediction(
                prediction,
                start_time,
                end_time,
                phone=segment.get("phone"),
                word=segment.get("word"),
                segment_type=segment.get("type"),
                index=segment.get("index", fallback_index),
            )
        )
    return predictions


def predict_context_segments(
    audio_path: str | Path,
    segments: list[dict[str, Any]],
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    model, index_to_label, checkpoint_file, device = _load_context_model(checkpoint_path)
    context_config = _context_config()
    waveform, _ = _load_prepared_wav(audio_path)
    audio_duration = len(waveform) / SAMPLE_RATE if SAMPLE_RATE else 0.0
    predictions = []
    for fallback_index, segment in enumerate(segments):
        start_time = float(segment.get("start") or 0.0)
        end_time = float(segment.get("end") if segment.get("end") is not None else start_time)
        if end_time <= start_time:
            end_time = start_time + MAX_SECONDS
        prediction = _predict_context_with_model(
            model,
            index_to_label,
            device,
            checkpoint_file,
            audio_path,
            start_time,
            end_time,
            context_config,
            waveform,
            audio_duration,
        )
        predictions.append(
            _format_segment_prediction(
                prediction,
                start_time,
                end_time,
                phone=segment.get("phone"),
                word=segment.get("word"),
                segment_type=segment.get("type"),
                index=segment.get("index", fallback_index),
            )
        )
    return predictions


def _aggregate_segment_predictions(
    segment_predictions: list[dict[str, Any]],
    *,
    alignment_result: dict[str, Any],
    confidence_threshold: float | None = None,
    scorer_metadata: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from app.scoring.scoring_service import score_pronunciation_segments

    threshold = 0.5 if confidence_threshold is None else float(confidence_threshold)
    ordered_predictions = sorted(
        segment_predictions,
        key=lambda segment: (
            float(segment.get("start") or 0.0),
            float(segment.get("end") or 0.0),
            int(segment.get("index") or 0),
        ),
    )
    issue_segments = [
        segment
        for segment in ordered_predictions
        if segment.get("predicted_error_type") not in {None, "unknown"}
    ]
    issue_segments.sort(key=lambda segment: float(segment.get("diagnosis_confidence") or 0.0), reverse=True)
    top_segment = issue_segments[0] if issue_segments else None
    primary_selection_rule = "highest_classifier_diagnosis_confidence"
    problem_phonemes = []
    predicted_error_type = top_segment.get("predicted_error_type") if top_segment else "unknown"
    class_probabilities = top_segment.get("class_probabilities") if top_segment else None
    diagnosis_confidence = top_segment.get("diagnosis_confidence") if top_segment else None
    problem_segments = [
        segment
        for segment in issue_segments
        if float(segment.get("diagnosis_confidence") or 0.0) >= threshold
    ]

    demo_score = estimate_demo_score(predicted_error_type, diagnosis_confidence)
    scoring_result = score_pronunciation_segments(alignment_result, ordered_predictions)
    hybrid_result = None
    try:
        from app.hybrid.hybrid_diagnosis import build_hybrid_diagnosis

        hybrid_result = build_hybrid_diagnosis(alignment_result, segment_predictions, scoring_result)
    except Exception as exc:
        hybrid_error = str(exc)
    else:
        hybrid_error = None

    if hybrid_result and hybrid_result.get("primary_error_type") not in {None, "unknown"}:
        predicted_error_type = hybrid_result["primary_error_type"]
        problem_segments = list(hybrid_result.get("top_issues") or problem_segments)
        primary_issue = problem_segments[0] if problem_segments else {}
        class_probabilities = primary_issue.get("class_probabilities") or class_probabilities
        diagnosis_confidence = primary_issue.get("diagnosis_confidence", diagnosis_confidence)
        primary_selection_rule = "hybrid_severity_then_heuristic_phone_score_then_classifier_diagnosis_confidence"
        top_segment = next(
            (
                segment
                for segment in ordered_predictions
                if segment.get("phone") == primary_issue.get("phone")
                and segment.get("word") == primary_issue.get("word")
                and float(segment.get("start") or 0.0) == float(primary_issue.get("start") or 0.0)
            ),
            top_segment,
        )

    # Public problem-phone lists follow alignment time, never hybrid severity.
    # Severity remains available only through the explicit top-segment metadata.
    for segment in sorted(
        problem_segments,
        key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)),
    ):
        phone = segment.get("phone")
        if phone and phone not in problem_phonemes:
            problem_phonemes.append(phone)

    segmental_score = scoring_result.get("utterance_segmental_score")
    scoring_method = scoring_result.get("scoring_method")
    scoring_is_heuristic = bool((scoring_result.get("metadata") or {}).get("is_heuristic"))
    score = segmental_score if segmental_score is not None else demo_score["score"]
    score_note = (
        "Heuristic/demo score, not production GOP."
        if segmental_score is not None and scoring_is_heuristic
        else demo_score["score_note"]
    )
    pronunciation_score_source = scoring_method if segmental_score is not None else "demo_error_type_heuristic"
    alignment_method = alignment_result.get("method")
    alignment_note = alignment_result.get("note")
    is_fallback = str(alignment_method or "").startswith("fallback")
    if is_fallback and not alignment_note:
        alignment_note = FALLBACK_ALIGNMENT_NOTE
    alignment_metadata = _safe_alignment_metadata(alignment_result)

    result = build_ai_result(
        score=score,
        problem_phonemes=problem_phonemes,
        predicted_error_type=predicted_error_type,
        class_probabilities=class_probabilities,
        diagnosis_confidence=diagnosis_confidence,
        feedback=(hybrid_result or {}).get("feedback") if hybrid_result else None,
        scorer=scorer_metadata or SCORER_METADATA,
        scoring=scoring_result,
        score_note=score_note,
        pronunciation_score_source=pronunciation_score_source,
        diagnosis_extra={
            "top_issues": (hybrid_result or {}).get("top_issues", []),
            "severity": (hybrid_result or {}).get("severity", "unknown"),
        } if hybrid_result else None,
        metadata={
            **DEFAULT_METADATA,
            "alignment_used": True,
            "alignment_status": alignment_result.get("status"),
            "alignment_method": alignment_method,
            "alignment_note": alignment_note,
            "gop_used": False,
            "hybrid_used": hybrid_result is not None,
            "hybrid_method": (hybrid_result or {}).get("hybrid_method"),
            "hybrid_status": (hybrid_result or {}).get("hybrid_status"),
            "hybrid_error": hybrid_error,
            "model_output_is_scoring": False,
            "segment_level_inference": True,
            "fallback_alignment": is_fallback,
            "location_reliability": (hybrid_result or {}).get(
                "location_reliability",
                "limited_fallback_alignment" if is_fallback else "unknown",
            ),
            "is_demo_score": True,
            "score_note": score_note,
            "pronunciation_score_source": pronunciation_score_source,
            "scoring_method": scoring_method,
            "scoring_is_heuristic": scoring_is_heuristic,
            "scoring_status": scoring_result.get("scoring_status"),
            "confidence_threshold": confidence_threshold,
            "segments_count": len(ordered_predictions),
            "global_diagnosis_selection": primary_selection_rule,
            "problem_phonemes_order": "alignment_time",
            "top_issue_segments_order": "classifier_diagnosis_confidence_desc",
            "limitation": (
                "Segment boundaries come from approximate fallback alignment unless an external aligner supplies "
                "the alignment_result. Heuristic GOP-like scoring is not production GOP. Classifier confidence is "
                "diagnosis confidence, not pronunciation scoring."
            ),
            **alignment_metadata,
            **(extra_metadata or {}),
        },
    )
    result["segments"] = ordered_predictions
    result["metadata"]["top_issue_segments"] = issue_segments[:5]
    if top_segment and top_segment.get("context"):
        result["metadata"].update(top_segment["context"])
    result["feedback"]["diagnosis"] = result["diagnosis"]
    result["feedback"]["scorer"] = result["scorer"]
    result["feedback"]["metadata"] = result["metadata"]

    # ── Canonical phone contract ──────────────────────────────────────────────
    phone_raw_list = [
        str(seg.get("phone") or "")
        for seg in (alignment_result.get("phones") or [])
    ]
    canon_result = _canonicalize_phones(phone_raw_list)

    primary_phone: str | None = None
    primary_err_type: str | None = None
    if hybrid_result:
        top_issues = hybrid_result.get("top_issues") or []
        if top_issues:
            primary_phone = str(top_issues[0].get("phone") or "").strip() or None
            primary_err_type = str(
                top_issues[0].get("predicted_error_type") or predicted_error_type or ""
            ).strip() or None
    if primary_phone is None and top_segment:
        primary_phone = str(top_segment.get("phone") or "").strip() or None
        primary_err_type = str(top_segment.get("predicted_error_type") or "").strip() or None

    diagnosis_invalid = False
    guard_reason: str | None = None
    for issue in problem_segments:
        invalid, reason = _check_inventory_guard(
            str(issue.get("phone") or "").strip() or None,
            str(issue.get("predicted_error_type") or predicted_error_type or "").strip() or None,
            canon_result["canonical_phones"],
        )
        if invalid:
            diagnosis_invalid, guard_reason = True, reason
            break
    if not diagnosis_invalid:
        diagnosis_invalid, guard_reason = _check_inventory_guard(
            primary_phone,
            primary_err_type,
            canon_result["canonical_phones"],
        )
    reliability = _compute_reliability(
        alignment_result,
        canon_result,
        diagnosis_invalid=diagnosis_invalid,
        guard_reason=guard_reason,
    )

    canonical_problem_phones: list[str] = []
    for raw_phone in problem_phonemes:
        canonical = _canonicalize_phones([str(raw_phone)]).get("canonical_phones") or []
        if canonical and canonical[0] not in canonical_problem_phones:
            canonical_problem_phones.append(canonical[0])

    if reliability["status"] == "invalid":
        result["score"] = None
        result["score_type"] = "unavailable"
        result["score_note"] = "Score unavailable: alignment or diagnosis validation failed."
        result["problem_phonemes"] = []
        result["primary_feedback"] = None
    else:
        result["problem_phonemes"] = canonical_problem_phones
        result["primary_feedback"] = result["feedback"].get("summary")

    phone_results = _build_phone_results(
        alignment_result,
        ordered_predictions,
        canon_result,
        reliability_status=reliability["status"],
    )

    result["reliability"] = reliability
    # MFA is an alignment source, not the authoritative pronunciation source.
    # The backend replaces these fields from stored vocabulary_items data before
    # persisting a result for the frontend.
    result["display_pronunciation"] = None
    result["pronunciation_variant"] = None
    result["expected_phones"] = []
    result["pronunciation_source"] = "unavailable"
    result["phone_results"] = phone_results

    # Segment diagnostics are public contract data too. Preserve their temporal
    # order while replacing raw MFA/ARPABET symbols with safe canonical IPA.
    public_segments: list[dict[str, Any]] = []
    for segment in ordered_predictions:
        canonical = _canonicalize_phones([str(segment.get("phone") or "")]).get("canonical_phones")
        public_segments.append({
            **segment,
            "phone": canonical[0] if canonical else None,
        })
    result["segments"] = public_segments
    result["metadata"]["top_issue_segments"] = [
        segment for segment in public_segments
        if segment.get("predicted_error_type") not in {None, "unknown"}
    ][:5]
    top_issues = result.get("diagnosis", {}).get("top_issues")
    if isinstance(top_issues, list):
        public_top_issues: list[dict[str, Any]] = []
        for issue in top_issues:
            if not isinstance(issue, dict):
                continue
            canonical = _canonicalize_phones([str(issue.get("phone") or "")]).get("canonical_phones")
            public_top_issues.append({
                **issue,
                "phone": canonical[0] if canonical else None,
            })
        result["diagnosis"]["top_issues"] = public_top_issues

    return result


def _safe_alignment_metadata(alignment_result: dict[str, Any]) -> dict[str, Any]:
    metadata = alignment_result.get("metadata") if isinstance(alignment_result.get("metadata"), dict) else {}
    alignment_status = str(alignment_result.get("status") or "")
    used_mfa = alignment_result.get("method") == "mfa" and alignment_status in {"success", "warning"}
    is_fallback = bool(
        metadata.get("fallback_alignment")
        or metadata.get("is_fallback")
        or str(alignment_result.get("method") or "").startswith("fallback")
    )
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if str(key).lower() not in SENSITIVE_ALIGNMENT_METADATA_KEYS | {"mfa_error"}
    }
    mfa_error = metadata.get("mfa_error")
    if isinstance(mfa_error, dict):
        safe_metadata["mfa_error"] = {
            "code": mfa_error.get("code"),
            "message": mfa_error.get("message"),
        }
    words = alignment_result.get("words") if isinstance(alignment_result.get("words"), list) else []
    phones = alignment_result.get("phones") if isinstance(alignment_result.get("phones"), list) else []
    safe_metadata.setdefault("word_segments_count", len(words))
    safe_metadata.setdefault("phone_segments_count", len(phones))
    safe_metadata.setdefault("is_forced_alignment", used_mfa)
    safe_metadata.setdefault("mfa_used", used_mfa)
    safe_metadata.setdefault("textgrid_parse_success", used_mfa)
    safe_metadata["fallback_alignment"] = is_fallback
    if is_fallback:
        safe_metadata["alignment_status"] = "fallback"
        safe_metadata.setdefault(
            "alignment_note",
            "Fallback alignment is approximate and has limited location reliability.",
        )
        safe_metadata.setdefault("location_reliability", "limited_fallback_alignment")
    return safe_metadata


def score_aligned_audio(
    audio_path: str | Path,
    alignment_result: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    _require_usable_alignment(alignment_result)
    segments = get_alignment_segments(alignment_result)
    segment_predictions = predict_segments(audio_path, segments, checkpoint_path)
    return _aggregate_segment_predictions(
        segment_predictions,
        alignment_result=alignment_result,
        confidence_threshold=confidence_threshold,
    )


def score_aligned_audio_context(
    audio_path: str | Path,
    alignment_result: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    _require_usable_alignment(alignment_result)
    segments = get_alignment_segments(alignment_result)
    segment_predictions = predict_context_segments(audio_path, segments, checkpoint_path)
    context_config = _context_config()
    return _aggregate_segment_predictions(
        segment_predictions,
        alignment_result=alignment_result,
        confidence_threshold=confidence_threshold,
        scorer_metadata=CONTEXT_SCORER_METADATA,
        extra_metadata={
            **context_config,
            "context_inference_note": (
                "CNN Attention ran on context-expanded crops while user-facing locations retain original "
                "alignment segment boundaries."
            ),
        },
    )


def _job_prompt_text(job: dict[str, Any]) -> str | None:
    for key in ("prompt_text", "target_text", "target_sentence", "target_word"):
        value = str(job.get(key) or "").strip()
        if value:
            return value
    return None


def _job_canonical_phones(job: dict[str, Any]) -> list[str]:
    raw_phones = job.get("canonical_phones") or job.get("phones")
    if isinstance(raw_phones, str):
        return [phone for phone in raw_phones.split() if phone]
    if isinstance(raw_phones, (list, tuple)):
        return [str(phone).strip() for phone in raw_phones if str(phone).strip()]
    return []


def _job_prepared_audio(job: dict[str, Any]) -> PreparedMfaAudio | None:
    prepared = job.get("_prepared_audio")
    return prepared if isinstance(prepared, PreparedMfaAudio) else None


def _job_audio_quality_warnings(job: dict[str, Any]) -> list[str]:
    warnings = job.get("_audio_quality_warnings")
    if not isinstance(warnings, (list, tuple)):
        return []
    return [str(warning) for warning in warnings if str(warning).strip()]


def _require_usable_alignment(alignment_result: dict[str, Any]) -> None:
    status = str(alignment_result.get("alignment_status") or alignment_result.get("status") or "")
    segments = get_alignment_segments(alignment_result)
    if status == "failed" or not segments:
        error = alignment_result.get("error") if isinstance(alignment_result.get("error"), dict) else {}
        error_code = error.get("code") or (alignment_result.get("metadata") or {}).get("error_code") or "alignment_failed"
        raise CNNAttentionScorerError(f"Alignment is unavailable for segment inference: {error_code}.")


def score_pronunciation(job: dict[str, Any], confidence_threshold: float | None = None) -> dict[str, Any]:
    temp_audio_path: Path | None = None
    try:
        audio_path, is_temp = _audio_path_from_job(job)
        temp_audio_path = audio_path if is_temp else None
        prompt_text = _job_prompt_text(job)
        if prompt_text:
            from app.alignment.alignment_service import align_audio

            prepared_from_job = _job_prepared_audio(job)
            with tempfile.TemporaryDirectory(prefix="cnn-prepared-") as prepared_directory:
                prepared_audio = prepared_from_job or prepare_audio_for_mfa(audio_path, Path(prepared_directory) / "prepared.wav")
                alignment_result = align_audio(
                    audio_path,
                    prompt_text=prompt_text,
                    canonical_phones=_job_canonical_phones(job),
                    job_id=str(job.get("job_id") or ""),
                    prepared_audio=prepared_audio,
                    upstream_quality_warnings=_job_audio_quality_warnings(job),
                )
                return score_aligned_audio(
                    prepared_audio.path,
                    alignment_result,
                    confidence_threshold=confidence_threshold,
                )

        prepared_from_job = _job_prepared_audio(job)
        with tempfile.TemporaryDirectory(prefix="cnn-prepared-") as prepared_directory:
            prepared_audio = prepared_from_job or prepare_audio_for_mfa(audio_path, Path(prepared_directory) / "prepared.wav")
            prediction = predict_audio(prepared_audio.path)
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
            "pronunciation_score_source": "demo_error_type_heuristic",
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


def score_pronunciation_context(job: dict[str, Any], confidence_threshold: float | None = None) -> dict[str, Any]:
    temp_audio_path: Path | None = None
    try:
        audio_path, is_temp = _audio_path_from_job(job)
        temp_audio_path = audio_path if is_temp else None
        prompt_text = _job_prompt_text(job)
        if not prompt_text:
            raise CNNAttentionScorerError(
                "SCORER_MODE=cnn_attention_context requires prompt_text, target_text, target_sentence, "
                "or target_word so segment boundaries can be aligned before applying context_0_10."
            )

        from app.alignment.alignment_service import align_audio

        prepared_from_job = _job_prepared_audio(job)
        with tempfile.TemporaryDirectory(prefix="cnn-prepared-") as prepared_directory:
            prepared_audio = prepared_from_job or prepare_audio_for_mfa(audio_path, Path(prepared_directory) / "prepared.wav")
            alignment_result = align_audio(
                audio_path,
                prompt_text=prompt_text,
                canonical_phones=_job_canonical_phones(job),
                job_id=str(job.get("job_id") or ""),
                prepared_audio=prepared_audio,
                upstream_quality_warnings=_job_audio_quality_warnings(job),
            )
            return score_aligned_audio_context(
                prepared_audio.path,
                alignment_result,
                confidence_threshold=confidence_threshold,
            )
    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)
