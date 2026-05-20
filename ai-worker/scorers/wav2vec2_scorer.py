from __future__ import annotations

import os
import re
import tempfile
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from audio.preprocessing import AudioDecodeError, AudioPreprocessingError, preprocess_audio


LOW_CONFIDENCE_WARNING = (
    "Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn."
)
BASELINE_NOTE = (
    "Pretrained Wav2Vec2 baseline with heuristic text matching; not a final pronunciation diagnosis model."
)
AUDIO_VALIDATION_FAILURE_SUMMARY = (
    "Bản ghi quá ngắn hoặc không đủ rõ để AI phân tích."
)
AUDIO_QUIET_FAILURE_SUMMARY = (
    "Bản ghi quá nhỏ hoặc không đủ rõ để AI phân tích."
)
AUDIO_DECODE_FAILURE_SUMMARY = (
    "Không thể đọc định dạng âm thanh của bản ghi. Vui lòng ghi âm lại và gửi lại."
)
AUDIO_DECODE_FAILURE_TIPS = [
    "Hãy thử ghi âm lại trong môi trường yên tĩnh hơn.",
    "Nếu lỗi tiếp tục xảy ra trên trình duyệt, hãy thử tải lại trang hoặc dùng định dạng ghi âm khác.",
]
TARGET_MATCH_THRESHOLD = 0.8
PARTIAL_MATCH_THRESHOLD = 0.5
DEFAULT_BASELINE_MAX_SCORE = 92
CALIBRATION_TIP = (
    "AI nhận dạng được từ mục tiêu, nhưng điểm được hiệu chỉnh vì Wav2Vec2 baseline "
    "chưa đánh giá đầy đủ phát âm từng âm."
)


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z' ]+", " ", value).lower()
    return " ".join(normalized.split())


def _content_suffix(content_type: str | None, audio_url: str) -> str:
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        suffix_by_type = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/m4a": ".m4a",
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
        }
        if normalized in suffix_by_type:
            return suffix_by_type[normalized]

    suffix = Path(audio_url.split("?", 1)[0]).suffix
    return suffix if suffix else ".audio"


def _download_audio(audio_url: str, timeout_seconds: float) -> tuple[Path, int, str | None]:
    response = requests.get(audio_url, timeout=timeout_seconds)
    response.raise_for_status()

    content_type = response.headers.get("content-type")
    suffix = _content_suffix(content_type, audio_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(response.content)
        temp_file.flush()
    finally:
        temp_file.close()

    return Path(temp_file.name), len(response.content), content_type


def _transcribe(waveform: Any, sample_rate: int, model_name: str) -> tuple[str, float]:
    import numpy as np
    import torch

    if waveform.size == 0:
        raise RuntimeError("Downloaded audio contains no samples")

    processor, model = _load_baseline_model(model_name)

    inputs = processor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    probabilities = torch.softmax(logits, dim=-1)
    max_probabilities, predicted_ids = torch.max(probabilities, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    blank_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    confidence_values = max_probabilities[0].detach().cpu().numpy()
    token_ids = predicted_ids[0].detach().cpu().numpy()
    if blank_token_id is not None:
        non_blank_values = confidence_values[token_ids != blank_token_id]
        if non_blank_values.size:
            confidence_values = non_blank_values

    confidence = float(np.clip(np.mean(confidence_values), 0.0, 1.0))
    return transcription, round(confidence, 2)


@lru_cache(maxsize=2)
def _load_baseline_model(model_name: str) -> tuple[Any, Any]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transformers.utils import logging as transformers_logging

    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)

    model.eval()
    print(f"Loaded Wav2Vec2 baseline model: {model_name}")
    return processor, model


def _build_audio_feedback(
    audio_metadata: dict[str, Any],
    downloaded_bytes: int,
    content_type: str | None,
) -> dict[str, Any]:
    return {
        "original_path": audio_metadata.get("original_path"),
        "duration_seconds": audio_metadata.get("duration_seconds", 0.0),
        "sample_rate": audio_metadata.get("sample_rate", 16000),
        "target_sample_rate": audio_metadata.get("target_sample_rate", 16000),
        "preprocessing": audio_metadata.get("preprocessing", {}),
        "warnings": audio_metadata.get("warnings", []),
        "downloaded_bytes": downloaded_bytes,
        "content_type": content_type,
        "is_too_short": audio_metadata.get("is_too_short", False),
        "is_too_long": audio_metadata.get("is_too_long", False),
        "is_too_quiet": audio_metadata.get("is_too_quiet", False),
        "peak_amplitude": audio_metadata.get("peak_amplitude", 0.0),
        "rms_energy": audio_metadata.get("rms_energy", 0.0),
        "denoise_enabled": audio_metadata.get("denoise_enabled", False),
        "denoise_strength": audio_metadata.get("denoise_strength", "light"),
        "noise_warning": audio_metadata.get("noise_warning"),
    }


def _build_completed_result(
    target_word: str,
    recognized_text: str,
    confidence: float,
    confidence_threshold: float,
    model_name: str,
    downloaded_bytes: int,
    content_type: str | None,
    audio_metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized_target = _normalize_text(target_word)
    normalized_recognized = _normalize_text(recognized_text)
    similarity = SequenceMatcher(None, normalized_target, normalized_recognized).ratio()
    if normalized_target and normalized_target in normalized_recognized.split():
        similarity = max(similarity, 0.9)

    # model_confidence is ASR transcript confidence, not pronunciation accuracy.
    # The final score combines transcript similarity to the target with that model confidence.
    text_match_component = round(similarity * 80.0)
    confidence_component = round(confidence * 20.0)
    audio_quality_component = None
    raw_score = int(max(0, min(100, text_match_component + confidence_component)))
    baseline_max_score = _baseline_max_score()
    score = min(raw_score, baseline_max_score)
    is_reliable = confidence >= confidence_threshold
    is_target_match = similarity >= TARGET_MATCH_THRESHOLD
    is_capped = score < raw_score
    rounded_similarity = round(similarity, 2)

    if similarity < PARTIAL_MATCH_THRESHOLD:
        result_label = "mismatch"
        summary = "Wav2Vec2 nhận dạng bản ghi chưa khớp với từ mục tiêu. Có thể bạn đã đọc sai từ hoặc âm thanh chưa đủ rõ."
    elif score >= 90:
        result_label = "correct"
        summary = "Bạn đọc khá đúng từ mục tiêu."
    elif score >= 80:
        result_label = "near_correct"
        summary = "Bản ghi gần chính xác với từ mục tiêu, nhưng vẫn còn một vài điểm cần cải thiện."
    elif score >= 50:
        result_label = "partial_match"
        summary = "Bản ghi chỉ khớp một phần với từ mục tiêu. Bạn nên đọc chậm hơn và rõ từng âm tiết hơn."
    else:
        result_label = "mismatch"
        summary = "Wav2Vec2 nhận dạng bản ghi chưa khớp với từ mục tiêu. Có thể bạn đã đọc sai từ hoặc âm thanh chưa đủ rõ."

    tips = []
    problem_phonemes = []
    if similarity < PARTIAL_MATCH_THRESHOLD:
        tips.append(
            "Hãy đọc lại đúng từ mục tiêu, chậm hơn và rõ từng âm tiết hơn."
        )
        problem_phonemes.append(
            {
                "phoneme": "/?/",
                "type": "recognition_mismatch",
                "severity": "high",
                "confidence": confidence,
                "tip": "Mô hình chưa nhận ra rõ từ mục tiêu trong bản ghi.",
            }
        )
    elif similarity < TARGET_MATCH_THRESHOLD:
        tips.append("Luyện đọc chậm hơn và nhấn rõ từng âm tiết trong từ mục tiêu.")
    else:
        tips.append("Tiếp tục giữ tốc độ đọc ổn định và phát âm rõ phần cuối của từ.")

    if not is_reliable:
        tips.insert(0, LOW_CONFIDENCE_WARNING)
    if is_capped and similarity >= TARGET_MATCH_THRESHOLD:
        tips.append(CALIBRATION_TIP)

    if not is_reliable:
        summary = f"{summary} {LOW_CONFIDENCE_WARNING}"
    if is_capped and similarity >= TARGET_MATCH_THRESHOLD:
        summary = f"{summary} {CALIBRATION_TIP}"

    return {
        "status": "completed",
        "score": score,
        "problem_phonemes": problem_phonemes,
        "feedback": {
            "summary": summary,
            "result_label": result_label,
            "scorer": "wav2vec2",
            "model_name": model_name,
            "target_word": target_word,
            "recognized_text": recognized_text,
            "normalized_recognized_text": normalized_recognized,
            "text_similarity": rounded_similarity,
            "tips": tips,
            "model_confidence": {
                "value": confidence,
                "threshold": confidence_threshold,
                "level": _confidence_level(confidence),
                "is_reliable": is_reliable,
            },
            "target_match": {
                "target_word": target_word,
                "recognized_text": recognized_text,
                "text_similarity": rounded_similarity,
                "is_match": is_target_match,
            },
            "score_breakdown": {
                "text_similarity": rounded_similarity,
                "text_match_component": text_match_component,
                "confidence_component": confidence_component,
                "audio_quality_component": audio_quality_component,
                "raw_score_before_cap": raw_score,
                "baseline_max_score": baseline_max_score,
                "final_score": score,
            },
            "audio": _build_audio_feedback(
                audio_metadata,
                downloaded_bytes=downloaded_bytes,
                content_type=content_type,
            ),
            "baseline_note": BASELINE_NOTE,
        },
    }


def _build_failed_result(
    target_word: str,
    confidence_threshold: float,
    model_name: str,
    error_message: str,
    summary: str | None = None,
    audio_metadata: dict[str, Any] | None = None,
    downloaded_bytes: int = 0,
    content_type: str | None = None,
    tips: list[str] | None = None,
) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        "summary": summary
        or (
            "Không thể xử lý bản ghi bằng AI. "
            "Vui lòng thử ghi âm lại hoặc kiểm tra định dạng âm thanh."
        ),
        "scorer": "wav2vec2",
        "result_label": "failed",
        "model_name": model_name,
        "target_word": target_word,
        "recognized_text": "",
        "text_similarity": None,
        "tips": tips
        or [
            "Không thể xử lý bản ghi. Vui lòng thử ghi âm lại trong môi trường yên tĩnh hơn."
        ],
        "model_confidence": {
            "value": 0.0,
            "threshold": confidence_threshold,
            "level": "low",
            "is_reliable": False,
        },
        "target_match": {
            "target_word": target_word,
            "recognized_text": "",
            "text_similarity": None,
            "is_match": False,
        },
        "score_breakdown": {
            "text_similarity": None,
            "text_match_component": 0,
            "confidence_component": 0,
            "audio_quality_component": None,
            "raw_score_before_cap": None,
            "baseline_max_score": _baseline_max_score(),
            "final_score": None,
        },
        "baseline_note": BASELINE_NOTE,
        "error": error_message,
    }

    if audio_metadata is not None:
        feedback["audio"] = _build_audio_feedback(
            audio_metadata,
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
        )

    return {
        "status": "failed",
        "score": None,
        "problem_phonemes": [],
        "feedback": feedback,
    }


def _audio_validation_summary(audio_metadata: dict[str, Any]) -> str:
    if audio_metadata.get("is_too_short"):
        return AUDIO_VALIDATION_FAILURE_SUMMARY
    if audio_metadata.get("is_too_quiet"):
        return AUDIO_QUIET_FAILURE_SUMMARY
    return AUDIO_VALIDATION_FAILURE_SUMMARY


def _baseline_max_score() -> int:
    value = os.getenv("WAV2VEC2_BASELINE_MAX_SCORE")
    if value is None:
        return DEFAULT_BASELINE_MAX_SCORE
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_BASELINE_MAX_SCORE
    return max(0, min(100, parsed))


def score_pronunciation(
    job: dict[str, Any],
    confidence_threshold: float,
    model_name: str,
    audio_download_timeout_seconds: float,
) -> dict[str, Any]:
    target_word = str(job.get("target_word") or "").strip()
    audio_url = str(job.get("audio_url") or "").strip()
    audio_path: Path | None = None
    downloaded_bytes = 0
    content_type: str | None = None

    try:
        if not audio_url:
            raise RuntimeError("Missing audio_url")

        audio_path, downloaded_bytes, content_type = _download_audio(
            audio_url,
            audio_download_timeout_seconds,
        )
        print(
            "audio_downloaded="
            f"bytes:{downloaded_bytes},content_type:{content_type or 'unknown'}"
        )

        preprocessing = preprocess_audio(audio_path, content_type=content_type)
        audio_metadata = preprocessing.metadata
        print(
            "audio_preprocessed="
            f"duration:{audio_metadata['duration_seconds']},"
            f"sample_rate:{audio_metadata['sample_rate']},"
            f"too_short:{audio_metadata['is_too_short']},"
            f"too_long:{audio_metadata['is_too_long']},"
            f"too_quiet:{audio_metadata['is_too_quiet']}"
        )

        if (
            audio_metadata["is_too_short"]
            or audio_metadata["is_too_quiet"]
            or preprocessing.waveform.size == 0
        ):
            return _build_failed_result(
                target_word=target_word,
                confidence_threshold=confidence_threshold,
                model_name=model_name,
                error_message="audio_validation_failed",
                summary=_audio_validation_summary(audio_metadata),
                audio_metadata=audio_metadata,
                downloaded_bytes=downloaded_bytes,
                content_type=content_type,
            )

        recognized_text, confidence = _transcribe(
            preprocessing.waveform,
            preprocessing.sample_rate,
            model_name,
        )
        return _build_completed_result(
            target_word=target_word,
            recognized_text=recognized_text,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
            audio_metadata=audio_metadata,
        )
    except AudioDecodeError as exc:
        return _build_failed_result(
            target_word=target_word,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            error_message=str(exc),
            summary=AUDIO_DECODE_FAILURE_SUMMARY,
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
            tips=AUDIO_DECODE_FAILURE_TIPS,
        )
    except AudioPreprocessingError as exc:
        return _build_failed_result(
            target_word=target_word,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            error_message=str(exc),
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
        )
    except Exception as exc:
        return _build_failed_result(
            target_word=target_word,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            error_message=str(exc),
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
        )
    finally:
        if audio_path:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
