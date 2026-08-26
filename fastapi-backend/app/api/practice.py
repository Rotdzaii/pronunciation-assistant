from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
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
from app.api.vocabulary import resolve_stored_pronunciation
from app.services.assignment_grading import (
    assignment_item_map,
    is_past_deadline,
    is_recipient,
    recalculate_assignment_grade,
    sweep_assignment_state,
)


router = APIRouter(prefix="/practice", tags=["Practice"])


def _maybe_auto_finalize_submission(supabase_client, assignment: dict, submission: dict) -> dict:
    """Lazily finalizes an in-progress submission if past deadline."""
    deadline = assignment.get("deadline")
    if not deadline:
        return submission
    if submission.get("is_locked") or not submission.get("started_at") or submission.get("submitted_at"):
        return submission
    try:
        now = datetime.now(timezone.utc)
        deadline_dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if deadline_dt < now:
            updated = (
                supabase_client.table("assessment_submissions")
                .update({
                    "submitted_at": deadline_dt.isoformat(),
                    "is_locked": True,
                    "updated_at": now.isoformat(),
                })
                .eq("id", submission["id"])
                .execute()
                .data
            )
            return updated[0] if updated else submission
    except (ValueError, KeyError):
        pass
    return submission

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
    assignment_id: str | None = None
    item_id: str | None = None
    audio_storage_path: str | None = None


class PracticeJobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PracticeJobResult(BaseModel):
    job_id: UUID
    status: Literal["completed", "failed"]
    score: float | None = Field(default=None, ge=0, le=100)
    score_type: str | None = None
    problem_phonemes: list[Any] = Field(default_factory=list)
    feedback: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_contract(self) -> "PracticeJobResult":
        # A completed result may have no numerical score when the deployed
        # model only performs error-type classification.
        if (
            self.status == "completed"
            and self.score is None
            and self.score_type != "unavailable"
        ):
            raise ValueError(
                "score is required when status is completed "
                "unless score_type is unavailable"
            )

        if self.status == "failed" and self.score is not None:
            raise ValueError(
                "score must be null when status is failed"
            )

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
    problem_phonemes: list[Any] = Field(default_factory=list)
    feedback: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class PracticeHistoryResponse(BaseModel):
    items: list[PracticeJobResponse]
    limit: int
    offset: int


class PracticeHistoryAudioResponse(BaseModel):
    audio_url: str
    expires_in: int


PRACTICE_AUDIO_SIGNED_URL_TTL_SECONDS = 300


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


_PRIVATE_RESULT_KEYS = {
    "audio_url",
    "audio_path",
    "local_path",
    "checkpoint_path",
    "textgrid_path",
    "mfa_output_dir",
    "mfa_temp_dir",
    "debug_artifact_dir",
    "stderr",
    "stdout",
    "command",
    "command_args",
    "mfa_process_diagnostic",
}
_PRIVATE_RESULT_MARKERS = (
    "x-amz-signature=",
    "signature=",
    "token=",
    "sig=",
    "c:\\",
    "/tmp/",
    ".textgrid",
)


def _sanitize_public_result(value: Any) -> Any:
    """Drop worker-local diagnostics before a result enters practice_history."""
    if isinstance(value, dict):
        return {
            key: _sanitize_public_result(item)
            for key, item in value.items()
            if str(key).lower() not in _PRIVATE_RESULT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_public_result(item) for item in value]
    if isinstance(value, str) and any(marker in value.lower() for marker in _PRIVATE_RESULT_MARKERS):
        return "[redacted-sensitive-value]"
    return value


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _storage_object_exists(storage_bucket: Any, object_path: str) -> bool:
    """Check the object name without downloading it or exposing its path."""
    parent, separator, filename = object_path.rpartition("/")
    if not separator or not parent or not filename:
        return False
    objects = storage_bucket.list(
        path=parent,
        options={"limit": 100, "search": filename},
    ) or []
    return any(isinstance(item, dict) and item.get("name") == filename for item in objects)


def _teacher_can_access_practice_audio(
    supabase_client: Any,
    *,
    teacher_id: str,
    student_id: str,
) -> bool:
    """Mirror the active teacher-class membership policy for audio playback."""
    teacher_classes = (
        supabase_client.table("teacher_classes")
        .select("class_id")
        .eq("teacher_id", teacher_id)
        .eq("status", "active")
        .execute()
        .data
        or []
    )
    class_ids = [str(row["class_id"]) for row in teacher_classes if row.get("class_id")]
    if not class_ids:
        return False
    memberships = (
        supabase_client.table("student_classes")
        .select("student_id")
        .in_("class_id", class_ids)
        .eq("student_id", student_id)
        .eq("status", "active")
        .limit(1)
        .execute()
        .data
        or []
    )
    return bool(memberships)


def _authorize_practice_audio_access(
    supabase_client: Any,
    current_user: CurrentUser,
    practice_row: dict[str, Any],
) -> None:
    if str(practice_row.get("student_id") or "") == current_user.id:
        return
    if current_user.app_role == "teacher" and _teacher_can_access_practice_audio(
        supabase_client,
        teacher_id=current_user.id,
        student_id=str(practice_row.get("student_id") or ""),
    ):
        return
    # Admin access is deliberately not elevated here: this endpoint has no
    # separate audio-review policy beyond owner and active teacher membership.
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You cannot access this recording")


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
    pronunciation_contract = resolve_stored_pronunciation(supabase_client, target_word)

    if bool(payload.assignment_id) != bool(payload.item_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="assignment_id and item_id must be provided together",
        )

    # Assignment practice is server-authorized before the job exists.  Free
    # practice remains unchanged when neither assignment field is supplied.
    assessment_sub = None
    assignment = None
    if payload.assignment_id and payload.item_id:
        asgn_rows = supabase_client.table("assignments").select("*").eq("id", payload.assignment_id).limit(1).execute().data or []
        if not asgn_rows or not is_recipient(supabase_client, payload.assignment_id, current_user.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assignment is not available to this student")
        assignment = asgn_rows[0]
        if is_past_deadline(assignment):
            sweep_assignment_state(supabase_client, assignment, current_user.id)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assignment deadline has passed")
        item = assignment_item_map(supabase_client, assignment).get(str(payload.item_id))
        vocabulary_item = (item or {}).get("vocabulary_items") or {}
        if not item or vocabulary_item.get("word", "").strip().lower() != target_word.lower():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Word is not part of this assignment")
        if assignment.get("is_assessment"):
            sub_rows = (
                supabase_client.table("assessment_submissions").select("*")
                .eq("assignment_id", payload.assignment_id).eq("student_id", current_user.id).limit(1).execute().data or []
            )
            if not sub_rows:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment must be started before recording")
            assessment_sub = sweep_assignment_state(supabase_client, assignment, current_user.id) or sub_rows[0]
            if assessment_sub.get("is_locked") or assessment_sub.get("submitted_at"):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assessment is locked")
            if any(str(recording.get("item_id")) == str(payload.item_id) for recording in assessment_sub.get("recordings") or []):
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This assessment word was already recorded")
            recordings = assessment_sub.get("recordings") or []
            previous_times = [datetime.fromisoformat(str(recording["recorded_at"]).replace("Z", "+00:00")) for recording in recordings if recording.get("recorded_at")]
            word_started_at = max(previous_times) if previous_times else datetime.fromisoformat(str(assessment_sub["started_at"]).replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > word_started_at + timedelta(seconds=int(assignment.get("timer_per_word_seconds") or 60)):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Time for this assessment word has elapsed")

    row = {
        "id": str(job_id),
        "student_id": current_user.id,
        "target_word": target_word,
        "audio_url": payload.audio_storage_path or audio_url,
        "status": "processing",
        "score": None,
        "problem_phonemes": [],
        "feedback": pronunciation_contract,
    }
    if assignment:
        row.update({
            "assignment_id": str(assignment["id"]),
            "item_id": str(payload.item_id),
            "assessment_submission_id": assessment_sub.get("id") if assessment_sub else None,
        })

    try:
        supabase_client.table("practice_history").insert(row).execute()
    except Exception as exc:
        if assessment_sub and ("duplicate" in str(exc).lower() or "unique" in str(exc).lower()):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This assessment word was already recorded") from exc
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
                # Queue the stable Storage key, never the one-hour signed URL.
                # The worker uses its service-role client to read this key when
                # it actually processes the message.
                "p_audio_url": str(row["audio_url"]),
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

    # Assessment recordings are immutable per word and store the stable path
    # when the client supplied it. The queue receives the same stable key.
    if payload.assignment_id and payload.item_id and assessment_sub:
        try:
            existing_recordings = list(assessment_sub.get("recordings") or [])
            existing_recordings.append({
                "item_id": payload.item_id,
                "word": payload.target_word,
                "audio_url": payload.audio_storage_path or payload.audio_url,
                "practice_job_id": str(job_id),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            supabase_client.table("assessment_submissions").update({
                "recordings": existing_recordings,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", assessment_sub["id"]).execute()
        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Practice job created but assessment recording could not be saved")

    return PracticeJobCreateResponse(
        job_id=str(job_id),
        status="processing",
        message="Practice job created and queued",
    )


@router.post(
    "/webhook/ai-result",
    response_model=PracticeJobResultResponse,
    status_code=status.HTTP_200_OK,
)
def update_practice_job_result(
    payload: PracticeJobResult,
    x_ai_webhook_secret: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> PracticeJobResultResponse:
    _require_ai_webhook_secret(
        x_ai_webhook_secret,
        settings.ai_webhook_secret,
    )

    job_id = str(payload.job_id)
    supabase_client = get_supabase_service_client(settings)

    print(
        "ai_webhook_received "
        f"job_id={job_id} "
        f"status={payload.status!r} "
        f"score={payload.score!r} "
        f"score_type={payload.score_type!r}",
        flush=True,
    )

    try:
        existing_response = (
            supabase_client.table("practice_history")
            .select(
                "id,target_word,feedback,status,score,"
                "assignment_id,student_id"
            )
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load practice job",
        ) from exc

    existing_rows = existing_response.data or []
    if not existing_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice job not found",
        )

    existing_row = existing_rows[0]
    existing_status = str(existing_row.get("status") or "")
    existing_score = existing_row.get("score")

    print(
        "ai_webhook_existing "
        f"job_id={job_id} "
        f"status={existing_status!r} "
        f"score={existing_score!r}",
        flush=True,
    )

    if existing_status == "completed" and existing_score is not None:
        print(
            "ai_webhook_skipped "
            f"job_id={job_id} "
            "reason=completed_result_already_has_score",
            flush=True,
        )
        return PracticeJobResultResponse(
            job_id=job_id,
            status=existing_status,
            message="Practice job result already processed",
        )

    if existing_status == "failed":
        print(
            "ai_webhook_skipped "
            f"job_id={job_id} "
            "reason=failed_result_already_terminal",
            flush=True,
        )
        return PracticeJobResultResponse(
            job_id=job_id,
            status=existing_status,
            message="Practice job result already processed",
        )

    repairing_null_score = (
        existing_status == "completed"
        and existing_score is None
        and payload.status == "completed"
        and payload.score is not None
    )

    if existing_status == "completed" and not repairing_null_score:
        print(
            "ai_webhook_skipped "
            f"job_id={job_id} "
            "reason=completed_null_score_not_repairable_by_payload",
            flush=True,
        )
        return PracticeJobResultResponse(
            job_id=job_id,
            status=existing_status,
            message="Practice job result already processed",
        )

    pronunciation_contract = resolve_stored_pronunciation(
        supabase_client,
        str(existing_row.get("target_word") or ""),
    )

    feedback = _sanitize_public_result(payload.feedback)
    if not isinstance(feedback, dict):
        feedback = {}

    if payload.score_type:
        feedback["score_type"] = payload.score_type

    feedback.update(pronunciation_contract)

    update_payload = {
        "status": payload.status,
        "score": payload.score,
        "problem_phonemes": _sanitize_public_result(
            payload.problem_phonemes
        ),
        "feedback": feedback,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    print(
        "ai_webhook_update "
        f"job_id={job_id} "
        f"status={update_payload['status']!r} "
        f"score={update_payload['score']!r} "
        f"repairing_null_score={repairing_null_score}",
        flush=True,
    )

    try:
        (
            supabase_client.table("practice_history")
            .update(update_payload)
            .eq("id", job_id)
            .execute()
        )

        verify_response = (
            supabase_client.table("practice_history")
            .select("id,status,score,updated_at")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(
            "ai_webhook_persist_error "
            f"job_id={job_id} "
            f"error={type(exc).__name__}",
            flush=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update practice job result",
        ) from exc

    saved_rows = verify_response.data or []
    if not saved_rows:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Practice result update could not be verified",
        )

    saved_row = saved_rows[0]
    saved_status = str(saved_row.get("status") or "")
    saved_score = saved_row.get("score")

    print(
        "ai_webhook_persisted "
        f"job_id={job_id} "
        f"status={saved_status!r} "
        f"score={saved_score!r}",
        flush=True,
    )

    if payload.status == "completed" and saved_score is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Completed practice result was persisted without a score",
        )

    if existing_row.get("assignment_id") and existing_row.get("student_id"):
        try:
            assignment_rows = (
                supabase_client.table("assignments")
                .select("*")
                .eq("id", existing_row["assignment_id"])
                .limit(1)
                .execute()
                .data
                or []
            )
            if assignment_rows:
                recalculate_assignment_grade(
                    supabase_client,
                    assignment_rows[0],
                    str(existing_row["student_id"]),
                )
        except Exception as exc:
            print(
                "assignment_grade_recalculation_failed "
                f"job_id={job_id} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Practice result was saved but assignment grade "
                    "could not be recalculated"
                ),
            ) from exc

    return PracticeJobResultResponse(
        job_id=job_id,
        status=payload.status,
        message=(
            "Practice job null score repaired"
            if repairing_null_score
            else "Practice job result updated"
        ),
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


@router.get("/history/{practice_id}/audio", response_model=PracticeHistoryAudioResponse)
def get_practice_history_audio(
    practice_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> PracticeHistoryAudioResponse:
    """Return a short-lived playback URL after applying recording access policy."""
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = (
            supabase_client.table("practice_history")
            .select("id,student_id,audio_url")
            .eq("id", str(practice_id))
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load recording",
        ) from exc

    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    practice_row = rows[0]
    _authorize_practice_audio_access(supabase_client, current_user, practice_row)
    audio_reference = str(practice_row.get("audio_url") or "").strip()
    if not audio_reference:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    # Old records can contain a previous HTTP(S) playback URL. Do not rewrite
    # it into the database; retain this compatibility path while new records
    # use stable, private Storage object keys.
    if _is_http_url(audio_reference):
        return PracticeHistoryAudioResponse(
            audio_url=audio_reference,
            expires_in=PRACTICE_AUDIO_SIGNED_URL_TTL_SECONDS,
        )

    try:
        storage_bucket = supabase_client.storage.from_(settings.practice_audio_bucket)
        if not _storage_object_exists(storage_bucket, audio_reference):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
        signed = storage_bucket.create_signed_url(
            audio_reference,
            PRACTICE_AUDIO_SIGNED_URL_TTL_SECONDS,
        )
        signed_url = signed.get("signedURL") or signed.get("signedUrl")
        if not isinstance(signed_url, str) or not signed_url:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
    except HTTPException:
        raise
    except Exception as exc:
        # Storage implementations return different error shapes. Never echo a
        # bucket name, object path, signed URL, or service credential to users.
        message = str(exc).lower()
        if "not found" in message or "404" in message:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Recording is temporarily unavailable",
        ) from exc

    return PracticeHistoryAudioResponse(
        audio_url=signed_url,
        expires_in=PRACTICE_AUDIO_SIGNED_URL_TTL_SECONDS,
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