from datetime import datetime, timezone
from hmac import compare_digest
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, model_validator

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
    "audio/m4a",
    "audio/webm",
    "audio/ogg",
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


class PracticeJobResult(BaseModel):
    job_id: UUID
    status: Literal["completed", "failed"]
    score: float | None = Field(default=None, ge=0, le=100)
    problem_phonemes: list[Any]
    feedback: dict[str, Any]

    @model_validator(mode="after")
    def validate_completed_score(self) -> "PracticeJobResult":
        if self.status == "completed" and self.score is None:
            raise ValueError("score is required when status is completed")

        return self


class PracticeJobResultResponse(BaseModel):
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


class PracticeHistoryResponse(BaseModel):
    items: list[PracticeJobResponse]
    limit: int
    offset: int


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "audio").name.strip()
    return name or "audio"


def _normalize_audio_content_type(content_type: str | None) -> str:
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Missing audio content type",
        )

    normalized_content_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio type",
        )

    return normalized_content_type


def _require_ai_webhook_secret(
    provided_secret: str | None,
    configured_secret: str | None,
) -> None:
    if not configured_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI webhook secret is not configured",
        )

    if not provided_secret or not compare_digest(provided_secret, configured_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid AI webhook secret",
        )


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


@router.post("/webhook/ai-result", response_model=PracticeJobResultResponse)
def update_practice_job_result(
    payload: PracticeJobResult,
    x_ai_webhook_secret: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> PracticeJobResultResponse:
    _require_ai_webhook_secret(x_ai_webhook_secret, settings.ai_webhook_secret)

    job_id = str(payload.job_id)
    supabase_client = get_supabase_service_client(settings)

    try:
        existing = (
            supabase_client.table("practice_history")
            .select("id")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load practice job",
        ) from exc

    if not (existing.data or []):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice job not found",
        )

    update_payload = {
        "status": payload.status,
        "score": payload.score,
        "problem_phonemes": payload.problem_phonemes,
        "feedback": payload.feedback,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        supabase_client.table("practice_history").update(update_payload).eq(
            "id", job_id
        ).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update practice job result",
        ) from exc

    return PracticeJobResultResponse(
        job_id=job_id,
        status=payload.status,
        message="Practice job result updated",
    )


@router.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_practice_audio(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AudioUploadResponse:
    normalized_content_type = _normalize_audio_content_type(file.content_type)

    content = await file.read()
    size = len(content)
    storage_path = f"{current_user.id}/{uuid4()}-{_safe_filename(file.filename)}"
    supabase_client = get_supabase_service_client(settings)

    try:
        supabase_client.storage.from_(settings.practice_audio_bucket).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": normalized_content_type},
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
        mime_type=normalized_content_type,
        size=size,
    )


@router.get("/history", response_model=PracticeHistoryResponse)
def list_practice_history(
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    student_id: UUID | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PracticeHistoryResponse:
    supabase_client = get_supabase_service_client(settings)

    try:
        query = (
            supabase_client.table("practice_history")
            .select(
                "id,student_id,target_word,audio_url,status,score,"
                "problem_phonemes,feedback,created_at,updated_at"
            )
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )

        if status_filter:
            query = query.eq("status", status_filter)

        if current_user.app_role == "student":
            if student_id is not None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="student_id filter is only available to teachers",
                )
            query = query.eq("student_id", current_user.id)
        elif current_user.app_role in ("teacher", "admin"):
            if student_id is not None:
                query = query.eq("student_id", str(student_id))
        else:
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
            detail="Unable to load practice history",
        ) from exc

    return PracticeHistoryResponse(
        items=[PracticeJobResponse(**item) for item in response.data or []],
        limit=limit,
        offset=offset,
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
        elif current_user.app_role not in ("teacher", "admin"):
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
