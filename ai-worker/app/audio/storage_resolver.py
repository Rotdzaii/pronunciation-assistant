from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import requests


AudioReferenceType = Literal["local_path", "signed_url", "storage_object_path"]


class AudioReferenceError(RuntimeError):
    """Safe error exposed when a queued audio reference cannot be resolved."""


@dataclass
class ResolvedAudioReference:
    path: Path
    reference_type: AudioReferenceType
    cleanup_required: bool = False

    def cleanup(self) -> None:
        if self.cleanup_required:
            self.path.unlink(missing_ok=True)


def _suffix_for(reference: str) -> str:
    suffix = Path(urlparse(reference).path).suffix
    return suffix if suffix and len(suffix) <= 12 else ".audio"


def _write_temp_audio(content: bytes, *, reference: str, temp_dir: Path | None) -> tuple[Path, bool]:
    if not content:
        raise AudioReferenceError("Queued audio download was empty.")

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        target = temp_dir / f"source{_suffix_for(reference)}"
        target.write_bytes(content)
        return target, False

    with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix_for(reference), prefix="phoenix-queued-audio-") as output:
        output.write(content)
        return Path(output.name), True


def resolve_audio_reference(
    job: dict[str, Any],
    *,
    storage_client: Any | None,
    practice_audio_bucket: str = "practice-audios",
    temp_dir: Path | None = None,
    request_timeout_seconds: int = 30,
) -> ResolvedAudioReference:
    """Resolve queue audio without treating a Storage object key as a local file.

    A local ``audio_path`` is only supported for debug/test use. Production queue
    messages may keep using the legacy ``audio_url`` key, whose value is either a
    legacy HTTP(S) URL or a stable Supabase Storage object path.
    """
    local_value = str(job.get("audio_path") or "").strip()
    if local_value:
        try:
            local_path = Path(local_value).expanduser()
            if local_path.is_file():
                return ResolvedAudioReference(local_path, "local_path")
        except OSError:
            pass

    reference = str(job.get("audio_url") or "").strip()
    if not reference:
        raise AudioReferenceError("Job has no audio reference.")

    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https"}:
        try:
            response = requests.get(reference, timeout=request_timeout_seconds)
            response.raise_for_status()
            content = response.content
        except Exception as exc:
            raise AudioReferenceError("Queued audio URL could not be downloaded.") from exc
        path, cleanup_required = _write_temp_audio(content, reference=reference, temp_dir=temp_dir)
        return ResolvedAudioReference(path, "signed_url", cleanup_required)

    if parsed.scheme:
        raise AudioReferenceError("Queued audio reference has an unsupported URL scheme.")

    if storage_client is None:
        raise AudioReferenceError("Worker has no Storage client for queued audio.")

    try:
        content = storage_client.storage.from_(practice_audio_bucket).download(reference)
    except Exception as exc:
        raise AudioReferenceError("Queued audio could not be downloaded from Storage.") from exc

    try:
        path, cleanup_required = _write_temp_audio(content, reference=reference, temp_dir=temp_dir)
    except AudioReferenceError:
        raise
    except Exception as exc:
        raise AudioReferenceError("Queued audio could not be prepared from Storage.") from exc
    return ResolvedAudioReference(path, "storage_object_path", cleanup_required)
