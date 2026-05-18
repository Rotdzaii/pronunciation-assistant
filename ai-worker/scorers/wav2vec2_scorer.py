from __future__ import annotations

import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


LOW_CONFIDENCE_WARNING = (
    "Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn."
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


def _load_waveform(audio_path: Path) -> Any:
    import librosa

    waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    return waveform


def _transcribe(audio_path: Path, model_name: str) -> tuple[str, float]:
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    waveform = _load_waveform(audio_path)
    if waveform.size == 0:
        raise RuntimeError("Downloaded audio contains no samples")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    inputs = processor(
        waveform,
        sampling_rate=16000,
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


def _build_completed_result(
    target_word: str,
    recognized_text: str,
    confidence: float,
    confidence_threshold: float,
    model_name: str,
    downloaded_bytes: int,
    content_type: str | None,
) -> dict[str, Any]:
    normalized_target = _normalize_text(target_word)
    normalized_recognized = _normalize_text(recognized_text)
    similarity = SequenceMatcher(None, normalized_target, normalized_recognized).ratio()
    if normalized_target and normalized_target in normalized_recognized.split():
        similarity = max(similarity, 0.9)

    score = round(max(0.0, min(100.0, (similarity * 80.0) + (confidence * 20.0))), 1)
    is_reliable = confidence >= confidence_threshold

    tips = []
    problem_phonemes = []
    if similarity < 0.85:
        tips.append("Âm thanh nhận dạng chưa khớp rõ với từ mục tiêu, hãy phát âm chậm và rõ hơn.")
        problem_phonemes.append(
            {
                "phoneme": "/?/",
                "type": "word_recognition",
                "severity": "medium" if similarity >= 0.55 else "high",
                "confidence": confidence,
                "tip": "Mô hình chưa nhận ra rõ từ mục tiêu trong bản ghi.",
            }
        )
    else:
        tips.append("Từ mục tiêu được nhận dạng khá gần với bản ghi.")

    if not is_reliable:
        tips.insert(0, LOW_CONFIDENCE_WARNING)

    return {
        "status": "completed",
        "score": score,
        "problem_phonemes": problem_phonemes,
        "feedback": {
            "summary": (
                "Kết quả baseline Wav2Vec2 cho thấy bản ghi gần với từ mục tiêu."
                if is_reliable
                else LOW_CONFIDENCE_WARNING
            ),
            "target_word": target_word,
            "recognized_text": recognized_text,
            "normalized_recognized_text": normalized_recognized,
            "text_similarity": round(similarity, 3),
            "tips": tips,
            "model_confidence": {
                "value": confidence,
                "threshold": confidence_threshold,
                "level": _confidence_level(confidence),
                "is_reliable": is_reliable,
            },
            "scorer": "wav2vec2",
            "model_name": model_name,
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "downloaded_bytes": downloaded_bytes,
                "content_type": content_type,
            },
            "baseline_note": (
                "Pretrained Wav2Vec2 baseline with heuristic text matching; "
                "not a final pronunciation diagnosis model."
            ),
        },
    }


def _build_failed_result(
    target_word: str,
    confidence_threshold: float,
    model_name: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "score": None,
        "problem_phonemes": [],
        "feedback": {
            "summary": f"AI processing failed: {error_message}",
            "target_word": target_word,
            "tips": ["Không thể xử lý bản ghi. Vui lòng thử ghi âm lại hoặc kiểm tra định dạng âm thanh."],
            "model_confidence": {
                "value": 0.0,
                "threshold": confidence_threshold,
                "level": "low",
                "is_reliable": False,
            },
            "scorer": "wav2vec2",
            "model_name": model_name,
        },
    }


def score_pronunciation(
    job: dict[str, Any],
    confidence_threshold: float,
    model_name: str,
    audio_download_timeout_seconds: float,
) -> dict[str, Any]:
    target_word = str(job.get("target_word") or "").strip()
    audio_url = str(job.get("audio_url") or "").strip()
    audio_path: Path | None = None

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

        recognized_text, confidence = _transcribe(audio_path, model_name)
        return _build_completed_result(
            target_word=target_word,
            recognized_text=recognized_text,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            downloaded_bytes=downloaded_bytes,
            content_type=content_type,
        )
    except Exception as exc:
        return _build_failed_result(
            target_word=target_word,
            confidence_threshold=confidence_threshold,
            model_name=model_name,
            error_message=str(exc),
        )
    finally:
        if audio_path:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
