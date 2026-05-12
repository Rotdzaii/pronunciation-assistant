from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_supabase_service_client, require_roles
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/practice", tags=["Practice"])

ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
}


class AudioUploadResponse(BaseModel):
    message: str
    storage_path: str
    audio_url: str
    mime_type: str
    size: int


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "audio").name.strip()
    return name or "audio"


@router.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_practice_audio(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AudioUploadResponse:
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio type",
        )

    content = await file.read()
    size = len(content)
    storage_path = f"{current_user.id}/{uuid4()}-{_safe_filename(file.filename)}"
    supabase_client = get_supabase_service_client(settings)

    try:
        supabase_client.storage.from_(settings.practice_audio_bucket).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": file.content_type},
        )
        signed = supabase_client.storage.from_(
            settings.practice_audio_bucket
        ).create_signed_url(storage_path, 3600)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to upload audio",
        ) from exc

    return AudioUploadResponse(
        message="uploaded",
        storage_path=storage_path,
        audio_url=signed.get("signedURL") or signed.get("signedUrl") or "",
        mime_type=file.content_type,
        size=size,
    )
