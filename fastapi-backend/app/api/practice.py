from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.auth import (
    CurrentUser,
    get_current_user,
    get_supabase_service_client,
    require_roles,
)
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


class PracticeJobCreate(BaseModel):
    target_word: str = Field(..., min_length=1)
    audio_url: str = Field(..., min_length=1)


class PracticeJobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PracticeJobResponse(BaseModel):
    id: str
    student_id: str
    target_word: str
    audio_url: str
    status: str
    score: float | None = None
    problem_phonemes: list[Any]
    feedback: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "audio").name.strip()
    return name or "audio"


@router.post(
    "/create-job",
    response_model=PracticeJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_practice_job(
    payload: PracticeJobCreate,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> PracticeJobCreateResponse:
    target_word = payload.target_word.strip()
    audio_url = payload.audio_url.strip()
    if not target_word or not audio_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="target_word and audio_url are required",
        )

    job_id = uuid4()
    supabase_client = get_supabase_service_client(settings)
    row = {
        "id": str(job_id),
        "student_id": current_user.id,
        "target_word": target_word,
        "audio_url": audio_url,
        "status": "processing",
        "score": None,
        "problem_phonemes": [],
        "feedback": {},
    }

    try:
        supabase_client.table("practice_history").insert(row).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create practice job",
        ) from exc

    try:
        supabase_client.rpc(
            "enqueue_practice_job",
            {
                "p_job_id": str(job_id),
                "p_student_id": current_user.id,
                "p_target_word": target_word,
                "p_audio_url": audio_url,
            },
        ).execute()
    except Exception as exc:
        try:
            supabase_client.table("practice_history").update({"status": "failed"}).eq(
                "id", str(job_id)
            ).execute()
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Practice job was created but could not be queued",
        ) from exc

    return PracticeJobCreateResponse(
        job_id=str(job_id),
        status="processing",
        message="Practice job created and queued",
    )


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


@router.get("/{job_id}", response_model=PracticeJobResponse)
def get_practice_job(
    job_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PracticeJobResponse:
    supabase_client = get_supabase_service_client(settings)

    try:
        query = (
            supabase_client.table("practice_history")
            .select(
                "id,student_id,target_word,audio_url,status,score,"
                "problem_phonemes,feedback,created_at,updated_at"
            )
            .eq("id", str(job_id))
            .limit(1)
        )
        if current_user.app_role == "student":
            query = query.eq("student_id", current_user.id)
        elif current_user.app_role != "teacher":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )

        response = query.execute()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load practice job",
        ) from exc

    data = response.data or []
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice job not found",
        )

    return PracticeJobResponse(**data[0])
