from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.core.auth import (
    CurrentUser,
    get_current_user,
    get_supabase_service_client,
    require_roles,
)
from app.core.config import Settings, get_settings

router = APIRouter(prefix="/assignments", tags=["Assignments"])


def _maybe_auto_finalize(supabase_client, assignment: dict, submission: dict) -> dict:
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


class AssignmentCreatePayload(BaseModel):
    title: str
    description: str | None = None
    content_type: str
    content_id: str
    class_id: str | None = None
    student_id: str | None = None
    due_date: str | None = None
    is_assessment: bool = False
    deadline: str | None = None
    timer_per_word_seconds: int = 60


class AssignmentUpdatePayload(BaseModel):
    title: str | None = None
    description: str | None = None
    due_date: str | None = None


class AssignmentProgressUpdatePayload(BaseModel):
    completed_items: int


class AssignmentProgressResponse(BaseModel):
    id: str
    assignment_id: str
    student_id: str
    status: str
    completed_items: int
    total_items: int
    completed_at: str | None = None
    created_at: str
    updated_at: str


class AssignmentResponse(BaseModel):
    id: str
    title: str
    description: str | None = None
    content_type: str
    content_id: str
    class_id: str | None = None
    student_id: str | None = None
    assigned_by: str
    due_date: str | None = None
    created_at: str
    updated_at: str
    is_assessment: bool = False
    deadline: str | None = None
    timer_per_word_seconds: int = 60


class AssignmentDetailResponse(AssignmentResponse):
    progress: list[AssignmentProgressResponse] = []


class AssignmentsListResponse(BaseModel):
    items: list[AssignmentResponse]
    total: int


class StudentAssignmentResponse(BaseModel):
    id: str
    title: str
    description: str | None = None
    content_type: str
    content_id: str
    class_id: str | None = None
    assigned_by: str
    due_date: str | None = None
    created_at: str
    progress_status: str
    completed_items: int
    total_items: int
    completed_at: str | None = None


class AssessmentSubmissionRecording(BaseModel):
    item_id: str
    word: str
    audio_url: str
    practice_job_id: str | None = None
    recorded_at: str


class AssessmentSubmissionResponse(BaseModel):
    id: str
    assignment_id: str
    student_id: str
    recordings: list[AssessmentSubmissionRecording]
    started_at: str
    submitted_at: str | None = None
    is_locked: bool
    created_at: str
    updated_at: str


class AssignmentStatusResponse(BaseModel):
    can_start: bool
    is_locked: bool
    started_at: str | None = None
    submitted_at: str | None = None
    deadline: str | None = None
    timer_per_word_seconds: int


class AssignmentWordsResponse(BaseModel):
    assignment_id: str
    title: str
    timer_per_word_seconds: int
    deadline: str | None
    items: list[dict]


class AssessmentStartResponse(BaseModel):
    submission_id: str
    started_at: str


class AssessmentSubmitPayload(BaseModel):
    pass  # recordings already in DB from incremental saves; no body needed


def _validate_content(supabase_client, content_type: str, content_id: str) -> int:
    if content_type == "vocabulary_set":
        rows = supabase_client.table("vocabulary_sets").select("id").eq("id", content_id).eq("is_active", True).execute().data
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vocabulary set not found")
        count_resp = supabase_client.table("vocabulary_set_items").select("id", count="exact").eq("set_id", content_id).execute()
        return count_resp.count or 0
    if content_type == "sentence_set":
        rows = supabase_client.table("sentence_sets").select("id").eq("id", content_id).execute().data
        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sentence set not found")
        count_resp = supabase_client.table("sentence_set_items").select("id", count="exact").eq("set_id", content_id).execute()
        return count_resp.count or 0
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="content_type must be 'vocabulary_set' or 'sentence_set'")


def _get_active_class_students(supabase_client, class_id: str) -> list[str]:
    rows = supabase_client.table("student_classes").select("student_id").eq("class_id", class_id).eq("status", "active").execute().data
    return [r["student_id"] for r in rows]


@router.post("/", response_model=AssignmentDetailResponse, status_code=status.HTTP_201_CREATED)
def create_assignment(
    payload: AssignmentCreatePayload,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentDetailResponse:
    if not payload.class_id and not payload.student_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide class_id or student_id")
    if payload.class_id and payload.student_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide class_id or student_id, not both")

    supabase_client = get_supabase_service_client(settings)
    total_items = _validate_content(supabase_client, payload.content_type, payload.content_id)

    try:
        assignment_row = (
            supabase_client.table("assignments")
            .insert({
                "title": payload.title,
                "description": payload.description,
                "content_type": payload.content_type,
                "content_id": payload.content_id,
                "class_id": payload.class_id,
                "student_id": payload.student_id,
                "assigned_by": current_user.id,
                "due_date": payload.due_date,
                "is_assessment": payload.is_assessment,
                "deadline": payload.deadline,
                "timer_per_word_seconds": payload.timer_per_word_seconds,
            })
            .execute()
            .data[0]
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create assignment") from exc

    assignment_id = assignment_row["id"]

    if payload.class_id:
        student_ids = _get_active_class_students(supabase_client, payload.class_id)
    else:
        student_ids = [payload.student_id]

    progress_rows: list[dict] = []
    if student_ids:
        try:
            progress_data = [
                {"assignment_id": assignment_id, "student_id": sid, "total_items": total_items}
                for sid in student_ids
            ]
            progress_rows = supabase_client.table("assignment_progress").insert(progress_data).execute().data
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Assignment created but failed to create progress records") from exc

    return AssignmentDetailResponse(
        **assignment_row,
        progress=[AssignmentProgressResponse(**p) for p in progress_rows],
    )


@router.get("/", response_model=AssignmentsListResponse)
def list_assignments(
    class_id: str | None = Query(default=None),
    student_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentsListResponse:
    supabase_client = get_supabase_service_client(settings)
    q = (
        supabase_client.table("assignments")
        .select("*", count="exact")
        .eq("assigned_by", current_user.id)
    )
    if class_id:
        q = q.eq("class_id", class_id)
    if student_id:
        q = q.eq("student_id", student_id)
    result = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return AssignmentsListResponse(items=result.data, total=result.count or 0)


@router.get("/student", response_model=list[StudentAssignmentResponse])
def list_student_assignments(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> list[StudentAssignmentResponse]:
    supabase_client = get_supabase_service_client(settings)
    rows = (
        supabase_client.table("assignment_progress")
        .select("*, assignments(*)")
        .eq("student_id", current_user.id)
        .execute()
        .data
    )
    now = datetime.now(timezone.utc)
    result: list[StudentAssignmentResponse] = []
    for row in rows:
        assignment = row.get("assignments") or {}
        prog_status = row["status"]
        due_date = assignment.get("due_date")
        if due_date and prog_status != "completed":
            try:
                dt = datetime.fromisoformat(due_date.replace("Z", "+00:00"))
                if dt < now:
                    prog_status = "overdue"
            except ValueError:
                pass
        result.append(StudentAssignmentResponse(
            id=assignment["id"],
            title=assignment["title"],
            description=assignment.get("description"),
            content_type=assignment["content_type"],
            content_id=str(assignment["content_id"]),
            class_id=str(assignment["class_id"]) if assignment.get("class_id") else None,
            assigned_by=str(assignment["assigned_by"]),
            due_date=due_date,
            created_at=assignment["created_at"],
            progress_status=prog_status,
            completed_items=row["completed_items"],
            total_items=row["total_items"],
            completed_at=row.get("completed_at"),
        ))
    return result


@router.get("/{assignment_id}", response_model=AssignmentDetailResponse)
def get_assignment(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentDetailResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = (
        supabase_client.table("assignments")
        .select("*")
        .eq("id", str(assignment_id))
        .eq("assigned_by", current_user.id)
        .execute()
        .data
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    progress_rows = (
        supabase_client.table("assignment_progress")
        .select("*")
        .eq("assignment_id", str(assignment_id))
        .execute()
        .data
    )
    return AssignmentDetailResponse(
        **rows[0],
        progress=[AssignmentProgressResponse(**p) for p in progress_rows],
    )


@router.patch("/{assignment_id}", response_model=AssignmentResponse)
def update_assignment(
    assignment_id: UUID,
    payload: AssignmentUpdatePayload,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentResponse:
    supabase_client = get_supabase_service_client(settings)
    existing = supabase_client.table("assignments").select("id").eq("id", str(assignment_id)).eq("assigned_by", current_user.id).execute().data
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")
    try:
        updated = supabase_client.table("assignments").update(updates).eq("id", str(assignment_id)).execute().data
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to update assignment") from exc
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    return AssignmentResponse(**updated[0])


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assignment(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> None:
    supabase_client = get_supabase_service_client(settings)
    existing = supabase_client.table("assignments").select("id").eq("id", str(assignment_id)).eq("assigned_by", current_user.id).execute().data
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    try:
        supabase_client.table("assignments").delete().eq("id", str(assignment_id)).execute()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to delete assignment") from exc


@router.patch("/student/{assignment_id}/progress", response_model=AssignmentProgressResponse)
def update_student_progress(
    assignment_id: UUID,
    payload: AssignmentProgressUpdatePayload,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentProgressResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = (
        supabase_client.table("assignment_progress")
        .select("*")
        .eq("assignment_id", str(assignment_id))
        .eq("student_id", current_user.id)
        .execute()
        .data
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Progress record not found")
    progress = rows[0]
    completed = max(0, min(payload.completed_items, progress["total_items"]))

    completed_at = progress.get("completed_at")
    if completed == 0:
        new_status = "not_started"
    elif progress["total_items"] > 0 and completed >= progress["total_items"]:
        new_status = "completed"
        if not completed_at:
            completed_at = datetime.now(timezone.utc).isoformat()
    else:
        new_status = "in_progress"

    try:
        updated = (
            supabase_client.table("assignment_progress")
            .update({"completed_items": completed, "status": new_status, "completed_at": completed_at})
            .eq("assignment_id", str(assignment_id))
            .eq("student_id", current_user.id)
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to update progress") from exc
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Progress record not found")
    return AssignmentProgressResponse(**updated[0])


@router.get("/{assignment_id}/words", response_model=AssignmentWordsResponse)
def get_assessment_words(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> AssignmentWordsResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    # Access check: student must have a progress row, teacher must own it
    if current_user.app_role == "student":
        prog = supabase_client.table("assignment_progress").select("id").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
        if not prog:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in this assignment")
    elif current_user.app_role == "teacher":
        if assignment.get("assigned_by") != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your assignment")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    if assignment.get("content_type") != "vocabulary_set":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only vocabulary_set assignments have a word list")
    set_id = str(assignment["content_id"])
    set_item_rows = (
        supabase_client.table("vocabulary_set_items")
        .select("item_id, sort_order, vocabulary_items(id,word,phonetic,meaning_vi,topic,level)")
        .eq("set_id", set_id)
        .order("sort_order")
        .execute()
        .data
    ) or []
    items = []
    for r in set_item_rows:
        vi = r.get("vocabulary_items") or {}
        items.append({
            "id": str(vi.get("id", r["item_id"])),
            "word": vi.get("word", ""),
            "phonetic": vi.get("phonetic"),
            "meaning_vi": vi.get("meaning_vi"),
            "topic": vi.get("topic"),
            "level": vi.get("level"),
        })
    return AssignmentWordsResponse(
        assignment_id=str(assignment_id),
        title=assignment["title"],
        timer_per_word_seconds=assignment.get("timer_per_word_seconds", 60),
        deadline=assignment.get("deadline"),
        items=items,
    )


@router.post("/{assignment_id}/start", response_model=AssessmentStartResponse, status_code=status.HTTP_201_CREATED)
def start_assessment(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AssessmentStartResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    if not assignment.get("is_assessment"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an assessment assignment")
    prog = supabase_client.table("assignment_progress").select("id").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if not prog:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in this assignment")
    # Check deadline
    deadline = assignment.get("deadline")
    if deadline:
        try:
            now = datetime.now(timezone.utc)
            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if dl < now:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assessment deadline has passed")
        except ValueError:
            pass
    # Check existing submission
    existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if existing:
        sub = existing[0]
        if sub.get("is_locked"):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Assessment already submitted")
        return AssessmentStartResponse(submission_id=sub["id"], started_at=sub["started_at"])
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        new_sub = supabase_client.table("assessment_submissions").insert({
            "assignment_id": str(assignment_id),
            "student_id": current_user.id,
            "started_at": now_iso,
        }).execute().data[0]
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to start assessment") from exc
    return AssessmentStartResponse(submission_id=new_sub["id"], started_at=new_sub["started_at"])


@router.get("/{assignment_id}/status", response_model=AssignmentStatusResponse)
def get_assignment_status(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentStatusResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    deadline = assignment.get("deadline")
    timer = assignment.get("timer_per_word_seconds", 60)
    existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if existing:
        sub = _maybe_auto_finalize(supabase_client, assignment, existing[0])
        can_start = not sub.get("is_locked", False)
        return AssignmentStatusResponse(
            can_start=can_start,
            is_locked=sub.get("is_locked", False),
            started_at=sub.get("started_at"),
            submitted_at=sub.get("submitted_at"),
            deadline=deadline,
            timer_per_word_seconds=timer,
        )
    # No submission yet — check if deadline allows starting
    can_start = True
    if deadline:
        try:
            dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if dl < datetime.now(timezone.utc):
                can_start = False
        except ValueError:
            pass
    return AssignmentStatusResponse(
        can_start=can_start,
        is_locked=False,
        started_at=None,
        submitted_at=None,
        deadline=deadline,
        timer_per_word_seconds=timer,
    )


@router.post("/{assignment_id}/submit", response_model=AssessmentSubmissionResponse)
def submit_assessment(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> AssessmentSubmissionResponse:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not started")
    sub = _maybe_auto_finalize(supabase_client, assignment, existing[0])
    if sub.get("is_locked"):
        detail = "Assessment auto-finalized at deadline" if sub.get("submitted_at") == assignment.get("deadline") else "Already submitted"
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    try:
        now = datetime.now(timezone.utc)
        updated = supabase_client.table("assessment_submissions").update({
            "submitted_at": now.isoformat(),
            "is_locked": True,
            "updated_at": now.isoformat(),
        }).eq("id", sub["id"]).execute().data
        sub = updated[0] if updated else sub
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to submit assessment") from exc
    recordings = [AssessmentSubmissionRecording(**r) for r in (sub.get("recordings") or [])]
    return AssessmentSubmissionResponse(
        id=sub["id"],
        assignment_id=str(assignment_id),
        student_id=current_user.id,
        recordings=recordings,
        started_at=sub["started_at"],
        submitted_at=sub.get("submitted_at"),
        is_locked=sub.get("is_locked", True),
        created_at=sub["created_at"],
        updated_at=sub["updated_at"],
    )


@router.get("/{assignment_id}/submissions", response_model=list[AssessmentSubmissionResponse])
def get_assessment_submissions(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> list[AssessmentSubmissionResponse]:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).eq("assigned_by", current_user.id).execute().data
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    subs = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).execute().data or []
    result = []
    for sub in subs:
        sub = _maybe_auto_finalize(supabase_client, assignment, sub)
        recordings = [AssessmentSubmissionRecording(**r) for r in (sub.get("recordings") or [])]
        result.append(AssessmentSubmissionResponse(
            id=sub["id"],
            assignment_id=str(assignment_id),
            student_id=sub["student_id"],
            recordings=recordings,
            started_at=sub["started_at"],
            submitted_at=sub.get("submitted_at"),
            is_locked=sub.get("is_locked", False),
            created_at=sub["created_at"],
            updated_at=sub["updated_at"],
        ))
    return result
