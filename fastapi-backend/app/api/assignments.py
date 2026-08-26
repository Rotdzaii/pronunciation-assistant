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
from app.services.assignment_grading import (
    assignment_item_rows,
    is_past_deadline,
    is_recipient,
    recalculate_assignment_grade,
    sweep_assignment_state,
)

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


class AssignmentGradeOverridePayload(BaseModel):
    score: float
    override_reason: str


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
    assignment_id: str
    title: str
    description: str | None = None
    content_type: str
    content_id: str
    class_id: str | None = None
    assigned_by: str
    due_date: str | None = None
    deadline: str | None = None
    is_assessment: bool
    timer_per_word_seconds: int
    class_name: str | None = None
    teacher_name: str | None = None
    created_at: str
    progress_status: str
    completed_items: int
    total_items: int
    completed_at: str | None = None
    work_status: str
    grading_status: str
    auto_score: float | None = None
    final_score: float | None = None
    is_overdue: bool
    can_start: bool
    can_continue: bool
    started_at: str | None = None
    submitted_at: str | None = None
    is_locked: bool = False


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
    work_status: str = "not_started"
    grading_status: str = "pending"
    can_continue: bool = False


class AssignmentGradebookItem(BaseModel):
    student_id: str
    student_name: str | None = None
    work_status: str
    completed_items: int
    total_items: int
    submitted_at: str | None = None
    is_locked: bool = False
    grading_status: str
    auto_score: float | None = None
    final_score: float | None = None
    teacher_override_score: float | None = None
    recordings: list[AssessmentSubmissionRecording] = []
    details: dict = {}


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


def _teacher_can_assign(supabase_client, teacher_id: str, class_id: str | None, student_id: str | None) -> bool:
    if class_id:
        rows = (
            supabase_client.table("teacher_classes").select("class_id")
            .eq("class_id", class_id).eq("teacher_id", teacher_id).eq("status", "active").limit(1).execute().data or []
        )
        return bool(rows)
    if not student_id:
        return False
    student_classes = (
        supabase_client.table("student_classes").select("class_id")
        .eq("student_id", student_id).eq("status", "active").execute().data or []
    )
    for membership in student_classes:
        rows = (
            supabase_client.table("teacher_classes").select("class_id")
            .eq("class_id", membership["class_id"]).eq("teacher_id", teacher_id).eq("status", "active").limit(1).execute().data or []
        )
        if rows:
            return True
    return False


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
    if not _teacher_can_assign(supabase_client, current_user.id, payload.class_id, payload.student_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only assign work to students in your active classes")
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
            recipient_data = [
                {"assignment_id": assignment_id, "student_id": sid}
                for sid in student_ids if sid
            ]
            if recipient_data:
                supabase_client.table("assignment_recipients").upsert(
                    recipient_data, on_conflict="assignment_id,student_id"
                ).execute()
            progress_data = [
                {"assignment_id": assignment_id, "student_id": sid, "total_items": total_items}
                for sid in student_ids if sid
            ]
            if progress_data:
                progress_rows = supabase_client.table("assignment_progress").upsert(
                    progress_data, on_conflict="assignment_id,student_id"
                ).execute().data or []
                supabase_client.table("assignment_grades").upsert(
                    [{"assignment_id": assignment_id, "student_id": sid, "grading_status": "pending"} for sid in student_ids if sid],
                    on_conflict="assignment_id,student_id",
                ).execute()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Assignment created but failed to create recipients") from exc

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
        supabase_client.table("assignment_recipients")
        .select("assignment_id,assigned_at,assignments(*)")
        .eq("student_id", current_user.id)
        .execute()
        .data
        or []
    )
    result: list[StudentAssignmentResponse] = []
    for row in rows:
        assignment = row.get("assignments") or {}
        if not assignment:
            continue
        assignment_id = str(assignment["id"])
        submission = sweep_assignment_state(supabase_client, assignment, current_user.id)
        progress_rows = supabase_client.table("assignment_progress").select("*").eq("assignment_id", assignment_id).eq("student_id", current_user.id).limit(1).execute().data or []
        grade_rows = supabase_client.table("assignment_grades").select("*").eq("assignment_id", assignment_id).eq("student_id", current_user.id).limit(1).execute().data or []
        class_rows = []
        if assignment.get("class_id"):
            class_rows = supabase_client.table("classes").select("name").eq("id", assignment["class_id"]).limit(1).execute().data or []
        teacher_rows = supabase_client.table("profiles").select("display_name").eq("id", assignment["assigned_by"]).limit(1).execute().data or []
        progress = progress_rows[0] if progress_rows else {}
        grade = grade_rows[0] if grade_rows else {}
        locked = bool(submission and submission.get("is_locked"))
        overdue = is_past_deadline(assignment)
        work_status = progress.get("status", "not_started")
        can_start = not overdue and (not assignment.get("is_assessment") or not locked)
        result.append(StudentAssignmentResponse(
            id=assignment_id,
            assignment_id=assignment_id,
            title=assignment["title"],
            description=assignment.get("description"),
            content_type=assignment["content_type"],
            content_id=str(assignment["content_id"]),
            class_id=str(assignment["class_id"]) if assignment.get("class_id") else None,
            assigned_by=str(assignment["assigned_by"]),
            class_name=class_rows[0].get("name") if class_rows else None,
            teacher_name=teacher_rows[0].get("display_name") if teacher_rows else None,
            due_date=assignment.get("due_date"),
            deadline=assignment.get("deadline"),
            is_assessment=bool(assignment.get("is_assessment")),
            timer_per_word_seconds=assignment.get("timer_per_word_seconds", 60),
            created_at=assignment["created_at"],
            progress_status=work_status,
            work_status=work_status,
            completed_items=progress.get("completed_items", 0),
            total_items=progress.get("total_items", 0),
            completed_at=progress.get("completed_at"),
            grading_status=grade.get("grading_status", "pending"),
            auto_score=grade.get("auto_score"),
            final_score=grade.get("final_score"),
            is_overdue=overdue,
            can_start=can_start,
            can_continue=can_start and work_status in {"not_started", "in_progress"},
            started_at=submission.get("started_at") if submission else None,
            submitted_at=submission.get("submitted_at") if submission else None,
            is_locked=locked,
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
    # Kept as a compatibility route, but clients cannot manufacture progress.
    # Derived progress is written only by the verified AI-result webhook.
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Assignment progress is calculated from verified practice results",
    )


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
    # Recipients, not the legacy progress table, define student access.
    if current_user.app_role == "student":
        if not is_recipient(supabase_client, str(assignment_id), current_user.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in this assignment")
    elif current_user.app_role == "teacher":
        if assignment.get("assigned_by") != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your assignment")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    if assignment.get("content_type") != "vocabulary_set":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only vocabulary_set assignments have a word list")
    set_item_rows = assignment_item_rows(supabase_client, assignment)
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
    if not is_recipient(supabase_client, str(assignment_id), current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in this assignment")
    if is_past_deadline(assignment):
        sweep_assignment_state(supabase_client, assignment, current_user.id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assessment deadline has passed")
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
        # Unique(assignment_id, student_id) is the concurrent-start guard.
        existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).limit(1).execute().data or []
        if existing and not existing[0].get("is_locked"):
            return AssessmentStartResponse(submission_id=existing[0]["id"], started_at=existing[0]["started_at"])
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to start assessment") from exc
    recalculate_assignment_grade(supabase_client, assignment, current_user.id, submission=new_sub)
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
    if not is_recipient(supabase_client, str(assignment_id), current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enrolled in this assignment")
    deadline = assignment.get("deadline")
    timer = assignment.get("timer_per_word_seconds", 60)
    existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if existing:
        sub = sweep_assignment_state(supabase_client, assignment, current_user.id) or existing[0]
        state = recalculate_assignment_grade(supabase_client, assignment, current_user.id, submission=sub)
        can_start = not is_past_deadline(assignment) and not sub.get("is_locked", False)
        return AssignmentStatusResponse(
            can_start=can_start,
            is_locked=sub.get("is_locked", False),
            started_at=sub.get("started_at"),
            submitted_at=sub.get("submitted_at"),
            deadline=deadline,
            timer_per_word_seconds=timer,
            work_status=state["progress"].get("status", "not_started"),
            grading_status=state["grade"].get("grading_status", "pending"),
            can_continue=can_start and not sub.get("submitted_at"),
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
    state = recalculate_assignment_grade(supabase_client, assignment, current_user.id)
    return AssignmentStatusResponse(
        can_start=can_start,
        is_locked=False,
        started_at=None,
        submitted_at=None,
        deadline=deadline,
        timer_per_word_seconds=timer,
        work_status=state["progress"].get("status", "not_started"),
        grading_status=state["grade"].get("grading_status", "pending"),
        can_continue=False,
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
    if not assignment.get("is_assessment") or not is_recipient(supabase_client, str(assignment_id), current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Assessment is not available to this student")
    existing = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", current_user.id).execute().data
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not started")
    sub = sweep_assignment_state(supabase_client, assignment, current_user.id) or existing[0]
    if sub.get("is_locked"):
        recordings = [AssessmentSubmissionRecording(**r) for r in (sub.get("recordings") or [])]
        return AssessmentSubmissionResponse(
            id=sub["id"], assignment_id=str(assignment_id), student_id=current_user.id,
            recordings=recordings, started_at=sub["started_at"], submitted_at=sub.get("submitted_at"),
            is_locked=True, created_at=sub["created_at"], updated_at=sub["updated_at"],
        )
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
    recalculate_assignment_grade(supabase_client, assignment, current_user.id, submission=sub)
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


@router.get("/{assignment_id}/submissions", response_model=list[AssignmentGradebookItem])
def get_assessment_submissions(
    assignment_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher", "admin"])),
    settings: Settings = Depends(get_settings),
) -> list[AssignmentGradebookItem]:
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data
    if not rows or (current_user.app_role != "admin" and rows[0].get("assigned_by") != current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    recipients = supabase_client.table("assignment_recipients").select("student_id").eq("assignment_id", str(assignment_id)).execute().data or []
    result: list[AssignmentGradebookItem] = []
    for recipient in recipients:
        student_id = str(recipient["student_id"])
        submission = sweep_assignment_state(supabase_client, assignment, student_id)
        state = recalculate_assignment_grade(supabase_client, assignment, student_id, submission=submission)
        profile_rows = supabase_client.table("profiles").select("display_name").eq("id", student_id).limit(1).execute().data or []
        grade = state["grade"]
        progress = state["progress"]
        result.append(AssignmentGradebookItem(
            student_id=student_id,
            student_name=profile_rows[0].get("display_name") if profile_rows else None,
            work_status=progress.get("status", "not_started"),
            completed_items=progress.get("completed_items", 0),
            total_items=progress.get("total_items", 0),
            submitted_at=submission.get("submitted_at") if submission else None,
            is_locked=bool(submission and submission.get("is_locked")),
            grading_status=grade.get("grading_status", "pending"),
            auto_score=grade.get("auto_score"),
            final_score=grade.get("final_score"),
            teacher_override_score=grade.get("teacher_override_score"),
            recordings=[AssessmentSubmissionRecording(**recording) for recording in (submission.get("recordings") or [])] if submission else [],
            details=grade.get("details") or {},
        ))
    return result


@router.patch("/{assignment_id}/grades/{student_id}", response_model=AssignmentGradebookItem)
def override_assignment_grade(
    assignment_id: UUID,
    student_id: UUID,
    payload: AssignmentGradeOverridePayload,
    current_user: CurrentUser = Depends(require_roles(["teacher", "admin"])),
    settings: Settings = Depends(get_settings),
) -> AssignmentGradebookItem:
    if not payload.override_reason.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="override_reason is required")
    if not 0 <= payload.score <= 100:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="score must be between 0 and 100")
    supabase_client = get_supabase_service_client(settings)
    rows = supabase_client.table("assignments").select("*").eq("id", str(assignment_id)).execute().data or []
    if not rows or (current_user.app_role != "admin" and rows[0].get("assigned_by") != current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    assignment = rows[0]
    if not is_recipient(supabase_client, str(assignment_id), str(student_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student is not an assignment recipient")
    state = recalculate_assignment_grade(supabase_client, assignment, str(student_id))
    grade = state["grade"]
    updated = supabase_client.table("assignment_grades").update({
        "teacher_override_score": payload.score,
        "final_score": payload.score,
        "grading_status": "graded",
        "graded_by": current_user.id,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "override_reason": payload.override_reason.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", grade["id"]).execute().data or []
    grade = updated[0] if updated else grade
    progress = state["progress"]
    submission_rows = supabase_client.table("assessment_submissions").select("*").eq("assignment_id", str(assignment_id)).eq("student_id", str(student_id)).limit(1).execute().data or []
    submission = submission_rows[0] if submission_rows else None
    profile_rows = supabase_client.table("profiles").select("display_name").eq("id", str(student_id)).limit(1).execute().data or []
    return AssignmentGradebookItem(student_id=str(student_id), student_name=profile_rows[0].get("display_name") if profile_rows else None, work_status=progress.get("status", "not_started"), completed_items=progress.get("completed_items", 0), total_items=progress.get("total_items", 0), submitted_at=submission.get("submitted_at") if submission else None, is_locked=bool(submission and submission.get("is_locked")), grading_status="graded", auto_score=grade.get("auto_score"), final_score=grade.get("final_score"), teacher_override_score=grade.get("teacher_override_score"), recordings=[AssessmentSubmissionRecording(**recording) for recording in (submission.get("recordings") or [])] if submission else [], details=grade.get("details") or {})
