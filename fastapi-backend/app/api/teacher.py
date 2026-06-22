from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_supabase_service_client, require_roles
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/teacher", tags=["Teacher"])


class TeacherCommonError(BaseModel):
    label: str
    count: int


class TeacherStudentSummary(BaseModel):
    id: str
    email: str | None = None
    display_name: str
    practice_count: int
    average_score: float
    latest_practice_at: str | None = None
    status: str
    common_errors: list[TeacherCommonError]


class TeacherAnalyticsResponse(BaseModel):
    total_students: int
    total_practice_sessions: int
    average_score: float
    avg_score: float
    active_jobs: int
    need_support_count: int
    good_progress_count: int
    improving_count: int
    common_errors: list[TeacherCommonError]
    progress_distribution: dict[str, int]
    students: list[TeacherStudentSummary]


class ReviewAuditLog(BaseModel):
    id: str
    actor_id: str
    actor_role: str
    action_type: str
    before_value: dict[str, Any] | None = None
    after_value: dict[str, Any] | None = None
    reason: str | None = None
    created_at: str | None = None


class TeacherReviewRequest(BaseModel):
    id: str
    practice_history_id: str
    student_id: str
    student_name: str
    student_email: str | None = None
    class_id: str | None = None
    class_name: str | None = None
    class_code: str | None = None
    target_word: str | None = None
    audio_url: str | None = None
    ai_score: float | None = None
    practice_status: str | None = None
    submitted_at: str | None = None
    source: str
    reason: str
    severity: str
    status: str
    student_note: str | None = None
    teacher_note: str | None = None
    teacher_resolution: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    resolved_at: str | None = None


class TeacherReviewRequestListResponse(BaseModel):
    items: list[TeacherReviewRequest]
    total: int
    pending_count: int
    student_report_count: int
    system_flag_count: int
    resolved_this_week_count: int
    limit: int
    offset: int


class TeacherReviewRequestDetail(TeacherReviewRequest):
    practice_history: dict[str, Any] | None = None
    ai_feedback: dict[str, Any] | None = None
    problem_phonemes: list[Any] | None = None
    audit_logs: list[ReviewAuditLog]


class TeacherReviewNotePayload(BaseModel):
    teacher_note: str = Field(..., min_length=1, max_length=1000)


class TeacherReviewResolvePayload(BaseModel):
    status: str = Field(pattern="^(resolved|rejected)$")
    teacher_resolution: str = Field(..., min_length=1, max_length=1000)
    teacher_note: str | None = Field(default=None, max_length=1000)


class PracticeResultItem(BaseModel):
    practice_history_id: str
    target_word: str | None = None
    audio_url: str | None = None
    score: float | None = None
    text_similarity: float | None = None
    model_confidence: float | None = None
    status: str
    mistake_categories: list[str]
    feedback: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    needs_review: bool = False


class TeacherStudentScoreSummary(BaseModel):
    total_attempts: int
    completed_attempts: int
    average_score: float | None = None
    last_practice_at: str | None = None
    common_mistakes: list[TeacherCommonError]


class TeacherStudentScoresResponse(BaseModel):
    student: TeacherStudentSummary
    class_id: str | None = None
    summary: TeacherStudentScoreSummary
    items: list[PracticeResultItem]
    limit: int
    offset: int


class TeacherClassScoreStudent(BaseModel):
    student_id: str
    student_name: str
    student_email: str | None = None
    total_attempts: int
    completed_attempts: int
    average_score: float | None = None
    last_practice_at: str | None = None
    needs_review_count: int


class TeacherClassScoresResponse(BaseModel):
    class_id: str
    class_name: str
    class_code: str | None = None
    completed_count: int
    pending_count: int
    failed_count: int
    common_mistakes: list[TeacherCommonError]
    students: list[TeacherClassScoreStudent]


def _derive_display_name(profile: dict[str, Any]) -> str:
    for key in ("display_name", "full_name", "name"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    email = profile.get("email")
    if isinstance(email, str) and email.strip():
        local_part = email.split("@", 1)[0]
        words = [word for word in local_part.replace("_", ".").split(".") if word]
        if words:
            return " ".join(word.capitalize() for word in words)

    return "Học viên"


def _extract_error_label(error: Any) -> str | None:
    if isinstance(error, str):
        return error.strip() or None

    if isinstance(error, dict):
        for key in ("phoneme", "type", "label", "message", "description", "word"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _score_average(scores: list[float]) -> float:
    if not scores:
        return 0
    return round(sum(scores) / len(scores), 1)


def _status_for_average(average_score: float) -> str:
    if average_score < 70:
        return "need_support"
    if average_score >= 80:
        return "good_progress"
    return "stable"


def _nullable_score_average(scores: list[float]) -> float | None:
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)


def _teacher_class_ids(supabase_client: Any, teacher_id: str) -> list[str]:
    rows = (
        supabase_client.table("teacher_classes")
        .select("class_id")
        .eq("teacher_id", teacher_id)
        .eq("status", "active")
        .execute()
        .data
        or []
    )
    return [str(row["class_id"]) for row in rows if row.get("class_id")]


def _teacher_student_memberships(
    supabase_client: Any,
    teacher_id: str,
    class_id: str | None = None,
) -> list[dict[str, Any]]:
    class_ids = _teacher_class_ids(supabase_client, teacher_id)
    if class_id is not None:
        if class_id not in class_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Teacher is not active in this class",
            )
        class_ids = [class_id]
    if not class_ids:
        return []
    return (
        supabase_client.table("student_classes")
        .select("class_id,student_id,status")
        .in_("class_id", class_ids)
        .eq("status", "active")
        .execute()
        .data
        or []
    )


def _teacher_student_ids(
    supabase_client: Any,
    teacher_id: str,
    class_id: str | None = None,
) -> set[str]:
    return {
        str(row["student_id"])
        for row in _teacher_student_memberships(supabase_client, teacher_id, class_id)
        if row.get("student_id")
    }


def _require_teacher_can_view_student(
    supabase_client: Any,
    teacher_id: str,
    student_id: str,
    class_id: str | None = None,
) -> None:
    if student_id not in _teacher_student_ids(supabase_client, teacher_id, class_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student is not in a class taught by this teacher",
        )


def _profiles_by_id(supabase_client: Any, user_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = sorted({user_id for user_id in user_ids if user_id})
    if not unique_ids:
        return {}
    rows = (
        supabase_client.table("profiles")
        .select("id,email,display_name,app_role")
        .in_("id", unique_ids)
        .execute()
        .data
        or []
    )
    return {str(row["id"]): row for row in rows if row.get("id")}


def _classes_by_id(supabase_client: Any, class_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = sorted({class_id for class_id in class_ids if class_id})
    if not unique_ids:
        return {}
    rows = (
        supabase_client.table("classes")
        .select("id,code,name,description,status")
        .in_("id", unique_ids)
        .execute()
        .data
        or []
    )
    return {str(row["id"]): row for row in rows if row.get("id")}


def _practice_rows_by_id(supabase_client: Any, practice_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = sorted({practice_id for practice_id in practice_ids if practice_id})
    if not unique_ids:
        return {}
    rows = (
        supabase_client.table("practice_history")
        .select("id,student_id,target_word,audio_url,status,score,problem_phonemes,feedback,created_at,updated_at")
        .in_("id", unique_ids)
        .execute()
        .data
        or []
    )
    return {str(row["id"]): row for row in rows if row.get("id")}


def _extract_feedback_number(feedback: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(feedback, dict):
        return None
    for key in keys:
        value = feedback.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _mistake_categories(problem_phonemes: Any) -> list[str]:
    labels: list[str] = []
    if not isinstance(problem_phonemes, list):
        return labels
    for item in problem_phonemes:
        label = _extract_error_label(item)
        if label and label not in labels:
            labels.append(label)
    return labels


def _practice_item(row: dict[str, Any], needs_review_ids: set[str] | None = None) -> PracticeResultItem:
    feedback = row.get("feedback") if isinstance(row.get("feedback"), dict) else {}
    practice_id = str(row.get("id"))
    return PracticeResultItem(
        practice_history_id=practice_id,
        target_word=row.get("target_word"),
        audio_url=row.get("audio_url"),
        score=float(row["score"]) if isinstance(row.get("score"), (int, float)) else None,
        text_similarity=_extract_feedback_number(feedback, ("text_similarity", "similarity")),
        model_confidence=_extract_feedback_number(feedback, ("model_confidence", "confidence")),
        status=str(row.get("status") or "processing"),
        mistake_categories=_mistake_categories(row.get("problem_phonemes")),
        feedback=feedback,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        needs_review=practice_id in (needs_review_ids or set()),
    )


def _review_response(
    row: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    classes: dict[str, dict[str, Any]],
    practice_rows: dict[str, dict[str, Any]],
) -> TeacherReviewRequest:
    student_id = str(row.get("student_id"))
    class_id = str(row.get("class_id")) if row.get("class_id") else None
    practice_id = str(row.get("practice_history_id"))
    profile = profiles.get(student_id, {})
    class_item = classes.get(class_id or "", {})
    practice = practice_rows.get(practice_id, {})
    return TeacherReviewRequest(
        id=str(row.get("id")),
        practice_history_id=practice_id,
        student_id=student_id,
        student_name=_derive_display_name(profile),
        student_email=profile.get("email"),
        class_id=class_id,
        class_name=class_item.get("name"),
        class_code=class_item.get("code"),
        target_word=practice.get("target_word"),
        audio_url=practice.get("audio_url"),
        ai_score=float(practice["score"]) if isinstance(practice.get("score"), (int, float)) else None,
        practice_status=practice.get("status"),
        submitted_at=practice.get("created_at"),
        source=str(row.get("source") or "student_report"),
        reason=str(row.get("reason") or "other"),
        severity=str(row.get("severity") or "medium"),
        status=str(row.get("status") or "pending"),
        student_note=row.get("student_note"),
        teacher_note=row.get("teacher_note"),
        teacher_resolution=row.get("teacher_resolution"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        resolved_at=row.get("resolved_at"),
    )


def _load_student_profiles(supabase_client: Any) -> list[dict[str, Any]]:
    try:
        response = (
            supabase_client.table("profiles")
            .select("id,email,app_role,display_name,full_name,name")
            .eq("app_role", "student")
            .execute()
        )
    except Exception:
        response = (
            supabase_client.table("profiles")
            .select("id,email,app_role")
            .eq("app_role", "student")
            .execute()
        )

    return response.data or []


def _load_practice_history(supabase_client: Any) -> list[dict[str, Any]]:
    response = (
        supabase_client.table("practice_history")
        .select("student_id,status,score,problem_phonemes,created_at,updated_at")
        .execute()
    )
    return response.data or []


@router.get("/review-requests", response_model=TeacherReviewRequestListResponse)
def list_teacher_review_requests(
    class_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    source: str | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherReviewRequestListResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
      memberships = _teacher_student_memberships(supabase_client, current_user.id, class_id)
      allowed_student_ids = {str(row["student_id"]) for row in memberships if row.get("student_id")}
      allowed_class_ids = {str(row["class_id"]) for row in memberships if row.get("class_id")}
      if not allowed_student_ids:
          return TeacherReviewRequestListResponse(
              items=[],
              total=0,
              pending_count=0,
              student_report_count=0,
              system_flag_count=0,
              resolved_this_week_count=0,
              limit=limit,
              offset=offset,
          )

      query = (
          supabase_client.table("review_requests")
          .select("*")
          .in_("student_id", sorted(allowed_student_ids))
          .order("created_at", desc=True)
      )
      if class_id:
          query = query.eq("class_id", class_id)
      if status_filter:
          query = query.eq("status", status_filter)
      if source:
          query = query.eq("source", source)
      rows = query.execute().data or []
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "review_requests" in message and ("not found" in message or "schema cache" in message or "does not exist" in message):
            return TeacherReviewRequestListResponse(
                items=[],
                total=0,
                pending_count=0,
                student_report_count=0,
                system_flag_count=0,
                resolved_this_week_count=0,
                limit=limit,
                offset=offset,
            )
        raise HTTPException(status_code=500, detail="Unable to load review requests") from exc

    student_ids = [str(row.get("student_id")) for row in rows if row.get("student_id")]
    class_ids = [str(row.get("class_id")) for row in rows if row.get("class_id")]
    practice_ids = [str(row.get("practice_history_id")) for row in rows if row.get("practice_history_id")]
    profiles = _profiles_by_id(supabase_client, student_ids)
    classes = _classes_by_id(supabase_client, class_ids)
    practices = _practice_rows_by_id(supabase_client, practice_ids)
    responses = [_review_response(row, profiles, classes, practices) for row in rows]

    if q:
        needle = q.strip().lower()
        responses = [
            item for item in responses
            if needle in item.student_name.lower()
            or needle in (item.student_email or "").lower()
            or needle in (item.target_word or "").lower()
            or needle in (item.class_name or "").lower()
            or needle in (item.class_code or "").lower()
        ]

    total = len(responses)
    paged = responses[offset: offset + limit]
    now = datetime.now(tz=UTC)
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = week_start.replace(day=now.day - now.weekday())
    return TeacherReviewRequestListResponse(
        items=paged,
        total=total,
        pending_count=sum(1 for item in responses if item.status in {"pending", "in_review", "reanalyzing"}),
        student_report_count=sum(1 for item in responses if item.source == "student_report"),
        system_flag_count=sum(1 for item in responses if item.source == "system_flag"),
        resolved_this_week_count=sum(
            1
            for item in responses
            if item.status in {"resolved", "rejected"}
            and item.resolved_at
            and item.resolved_at >= week_start.isoformat()
        ),
        limit=limit,
        offset=offset,
    )


def _load_teacher_review_request(
    supabase_client: Any,
    teacher_id: str,
    request_id: str,
) -> dict[str, Any]:
    rows = (
        supabase_client.table("review_requests")
        .select("*")
        .eq("id", request_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Review request not found")
    row = rows[0]
    _require_teacher_can_view_student(
        supabase_client,
        teacher_id,
        str(row.get("student_id")),
        str(row.get("class_id")) if row.get("class_id") else None,
    )
    return row


@router.get("/review-requests/{request_id}", response_model=TeacherReviewRequestDetail)
def get_teacher_review_request(
    request_id: str,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherReviewRequestDetail:
    supabase_client = get_supabase_service_client(settings)
    try:
        row = _load_teacher_review_request(supabase_client, current_user.id, request_id)
        profiles = _profiles_by_id(supabase_client, [str(row.get("student_id"))])
        classes = _classes_by_id(supabase_client, [str(row.get("class_id"))] if row.get("class_id") else [])
        practices = _practice_rows_by_id(supabase_client, [str(row.get("practice_history_id"))])
        audit_rows = (
            supabase_client.table("review_audit_logs")
            .select("*")
            .eq("review_request_id", request_id)
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to load review request") from exc

    base = _review_response(row, profiles, classes, practices)
    practice = practices.get(base.practice_history_id)
    return TeacherReviewRequestDetail(
        **base.model_dump(),
        practice_history=practice,
        ai_feedback=practice.get("feedback") if isinstance(practice, dict) else None,
        problem_phonemes=practice.get("problem_phonemes") if isinstance(practice, dict) else None,
        audit_logs=[
            ReviewAuditLog(
                id=str(audit.get("id")),
                actor_id=str(audit.get("actor_id")),
                actor_role=str(audit.get("actor_role") or ""),
                action_type=str(audit.get("action_type") or ""),
                before_value=audit.get("before_value"),
                after_value=audit.get("after_value"),
                reason=audit.get("reason"),
                created_at=audit.get("created_at"),
            )
            for audit in audit_rows
        ],
    )


def _insert_review_audit(
    supabase_client: Any,
    request_id: str,
    actor_id: str,
    action_type: str,
    before_value: dict[str, Any] | None,
    after_value: dict[str, Any] | None,
    reason: str | None = None,
) -> None:
    supabase_client.table("review_audit_logs").insert(
        {
            "review_request_id": request_id,
            "actor_id": actor_id,
            "actor_role": "teacher",
            "action_type": action_type,
            "before_value": before_value,
            "after_value": after_value,
            "reason": reason,
        }
    ).execute()


@router.post("/review-requests/{request_id}/note", response_model=TeacherReviewRequestDetail)
def add_teacher_review_note(
    request_id: str,
    payload: TeacherReviewNotePayload,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherReviewRequestDetail:
    supabase_client = get_supabase_service_client(settings)
    row = _load_teacher_review_request(supabase_client, current_user.id, request_id)
    note = payload.teacher_note.strip()
    updated = (
        supabase_client.table("review_requests")
        .update({"teacher_id": current_user.id, "teacher_note": note, "updated_at": datetime.now(tz=UTC).isoformat()})
        .eq("id", request_id)
        .execute()
        .data
        or []
    )
    _insert_review_audit(
        supabase_client,
        request_id,
        current_user.id,
        "teacher_added_note",
        {"teacher_note": row.get("teacher_note")},
        {"teacher_note": note},
    )
    return get_teacher_review_request(request_id, current_user, settings)


@router.post("/review-requests/{request_id}/resolve", response_model=TeacherReviewRequestDetail)
def resolve_teacher_review_request(
    request_id: str,
    payload: TeacherReviewResolvePayload,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherReviewRequestDetail:
    supabase_client = get_supabase_service_client(settings)
    row = _load_teacher_review_request(supabase_client, current_user.id, request_id)
    updates = {
        "teacher_id": current_user.id,
        "status": payload.status,
        "teacher_resolution": payload.teacher_resolution.strip(),
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "resolved_at": datetime.now(tz=UTC).isoformat(),
    }
    if payload.teacher_note is not None:
        updates["teacher_note"] = payload.teacher_note.strip()
    supabase_client.table("review_requests").update(updates).eq("id", request_id).execute()
    _insert_review_audit(
        supabase_client,
        request_id,
        current_user.id,
        "teacher_resolved_request" if payload.status == "resolved" else "teacher_rejected_request",
        {"status": row.get("status"), "teacher_resolution": row.get("teacher_resolution")},
        {"status": payload.status, "teacher_resolution": updates["teacher_resolution"]},
    )
    return get_teacher_review_request(request_id, current_user, settings)


@router.post("/review-requests/{request_id}/request-reanalysis", response_model=TeacherReviewRequestDetail)
def request_teacher_review_reanalysis(
    request_id: str,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherReviewRequestDetail:
    supabase_client = get_supabase_service_client(settings)
    row = _load_teacher_review_request(supabase_client, current_user.id, request_id)
    supabase_client.table("review_requests").update(
        {
            "teacher_id": current_user.id,
            "status": "reanalyzing",
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
    ).eq("id", request_id).execute()
    _insert_review_audit(
        supabase_client,
        request_id,
        current_user.id,
        "teacher_requested_reanalysis",
        {"status": row.get("status")},
        {"status": "reanalyzing"},
        "Queue integration is not enabled in this batch.",
    )
    return get_teacher_review_request(request_id, current_user, settings)


@router.get("/students/{student_id}/scores", response_model=TeacherStudentScoresResponse)
def get_teacher_student_scores(
    student_id: str,
    class_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherStudentScoresResponse:
    supabase_client = get_supabase_service_client(settings)
    _require_teacher_can_view_student(supabase_client, current_user.id, student_id, class_id)
    profile = _profiles_by_id(supabase_client, [student_id]).get(student_id, {})
    query = (
        supabase_client.table("practice_history")
        .select("id,student_id,target_word,audio_url,status,score,problem_phonemes,feedback,created_at,updated_at")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
    )
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        query = query.lte("created_at", date_to)
    rows = query.execute().data or []
    try:
        review_rows = (
            supabase_client.table("review_requests")
            .select("practice_history_id")
            .eq("student_id", student_id)
            .in_("status", ["pending", "in_review", "reanalyzing"])
            .execute()
            .data
            or []
        )
    except Exception:
        review_rows = []
    needs_review_ids = {str(row.get("practice_history_id")) for row in review_rows if row.get("practice_history_id")}
    items = [_practice_item(row, needs_review_ids) for row in rows]
    scores = [item.score for item in items if item.status == "completed" and item.score is not None]
    completed = [item for item in items if item.status == "completed"]
    latest = next((item.updated_at or item.created_at for item in items if item.updated_at or item.created_at), None)
    mistakes = Counter(label for item in items for label in item.mistake_categories)
    return TeacherStudentScoresResponse(
        student=TeacherStudentSummary(
            id=student_id,
            email=profile.get("email"),
            display_name=_derive_display_name(profile),
            practice_count=len(items),
            average_score=_nullable_score_average(scores) or 0,
            latest_practice_at=latest,
            status=_status_for_average(_nullable_score_average(scores) or 0),
            common_errors=[TeacherCommonError(label=label, count=count) for label, count in mistakes.most_common(3)],
        ),
        class_id=class_id,
        summary=TeacherStudentScoreSummary(
            total_attempts=len(items),
            completed_attempts=len(completed),
            average_score=_nullable_score_average(scores),
            last_practice_at=latest,
            common_mistakes=[TeacherCommonError(label=label, count=count) for label, count in mistakes.most_common(5)],
        ),
        items=items[offset: offset + limit],
        limit=limit,
        offset=offset,
    )


@router.get("/classes/{class_id}/scores", response_model=TeacherClassScoresResponse)
def get_teacher_class_scores(
    class_id: str,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherClassScoresResponse:
    supabase_client = get_supabase_service_client(settings)
    memberships = _teacher_student_memberships(supabase_client, current_user.id, class_id)
    student_ids = [str(row["student_id"]) for row in memberships if row.get("student_id")]
    classes = _classes_by_id(supabase_client, [class_id])
    class_item = classes.get(class_id, {})
    profiles = _profiles_by_id(supabase_client, student_ids)
    rows = []
    if student_ids:
        rows = (
            supabase_client.table("practice_history")
            .select("id,student_id,target_word,audio_url,status,score,problem_phonemes,feedback,created_at,updated_at")
            .in_("student_id", student_ids)
            .execute()
            .data
            or []
        )
    try:
        review_rows = (
            supabase_client.table("review_requests")
            .select("student_id")
            .eq("class_id", class_id)
            .in_("status", ["pending", "in_review", "reanalyzing"])
            .execute()
            .data
            or []
        )
    except Exception:
        review_rows = []
    review_counts = Counter(str(row.get("student_id")) for row in review_rows if row.get("student_id"))
    by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    mistakes: Counter[str] = Counter()
    for row in rows:
        sid = str(row.get("student_id"))
        by_student[sid].append(row)
        status_counts[str(row.get("status") or "processing")] += 1
        for label in _mistake_categories(row.get("problem_phonemes")):
            mistakes[label] += 1
    students: list[TeacherClassScoreStudent] = []
    for student_id in student_ids:
        items = by_student.get(student_id, [])
        scores = [
            float(row["score"])
            for row in items
            if row.get("status") == "completed" and isinstance(row.get("score"), (int, float))
        ]
        latest = max([str(row.get("updated_at") or row.get("created_at")) for row in items if row.get("updated_at") or row.get("created_at")] or [None])
        profile = profiles.get(student_id, {})
        students.append(
            TeacherClassScoreStudent(
                student_id=student_id,
                student_name=_derive_display_name(profile),
                student_email=profile.get("email"),
                total_attempts=len(items),
                completed_attempts=sum(1 for row in items if row.get("status") == "completed"),
                average_score=_nullable_score_average(scores),
                last_practice_at=latest,
                needs_review_count=review_counts.get(student_id, 0),
            )
        )
    students.sort(key=lambda item: item.student_name.lower())
    return TeacherClassScoresResponse(
        class_id=class_id,
        class_name=class_item.get("name") or "",
        class_code=class_item.get("code"),
        completed_count=status_counts.get("completed", 0),
        pending_count=status_counts.get("processing", 0) + status_counts.get("queued", 0),
        failed_count=status_counts.get("failed", 0),
        common_mistakes=[TeacherCommonError(label=label, count=count) for label, count in mistakes.most_common(6)],
        students=students,
    )


@router.get("/analytics", response_model=TeacherAnalyticsResponse)
def get_teacher_analytics(
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> TeacherAnalyticsResponse:
    supabase_client = get_supabase_service_client(settings)

    try:
        profiles = _load_student_profiles(supabase_client)
        history = _load_practice_history(supabase_client)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load teacher analytics",
        ) from exc

    profiles_by_id = {
        str(profile.get("id")): profile
        for profile in profiles
        if profile.get("id") is not None
    }
    history_by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_jobs = 0
    all_scores: list[float] = []
    all_errors: Counter[str] = Counter()

    for item in history:
        student_id = item.get("student_id")
        if student_id is None:
            continue

        student_key = str(student_id)
        if student_key not in profiles_by_id:
            continue

        history_by_student[student_key].append(item)

        if item.get("status") in {"processing", "queued"}:
            active_jobs += 1

        score = item.get("score")
        if item.get("status") == "completed" and isinstance(score, (int, float)):
            all_scores.append(float(score))

        errors = item.get("problem_phonemes")
        if isinstance(errors, list):
            for error in errors:
                label = _extract_error_label(error)
                if label:
                    all_errors[label] += 1

    students: list[TeacherStudentSummary] = []
    progress_distribution = {
        "need_support": 0,
        "improving": 0,
        "good_progress": 0,
    }

    for student_id, profile in profiles_by_id.items():
        items = history_by_student.get(student_id, [])
        scores = [
            float(item["score"])
            for item in items
            if item.get("status") == "completed" and isinstance(item.get("score"), (int, float))
        ]
        average_score = _score_average(scores)
        student_status = _status_for_average(average_score)
        if student_status == "stable":
            progress_distribution["improving"] += 1
        else:
            progress_distribution[student_status] += 1

        student_errors: Counter[str] = Counter()
        for item in items:
            errors = item.get("problem_phonemes")
            if isinstance(errors, list):
                for error in errors:
                    label = _extract_error_label(error)
                    if label:
                        student_errors[label] += 1

        latest_practice_at = None
        for item in items:
            candidate = item.get("updated_at") or item.get("created_at")
            if isinstance(candidate, str) and (
                latest_practice_at is None or candidate > latest_practice_at
            ):
                latest_practice_at = candidate

        students.append(
            TeacherStudentSummary(
                id=student_id,
                email=profile.get("email"),
                display_name=_derive_display_name(profile),
                practice_count=len(items),
                average_score=average_score,
                latest_practice_at=latest_practice_at,
                status=student_status,
                common_errors=[
                    TeacherCommonError(label=label, count=count)
                    for label, count in student_errors.most_common(3)
                ],
            )
        )

    students.sort(key=lambda student: student.display_name.lower())
    average_score = _score_average(all_scores)

    return TeacherAnalyticsResponse(
        total_students=len(students),
        total_practice_sessions=sum(student.practice_count for student in students),
        average_score=average_score,
        avg_score=average_score,
        active_jobs=active_jobs,
        need_support_count=progress_distribution["need_support"],
        good_progress_count=progress_distribution["good_progress"],
        improving_count=progress_distribution["improving"],
        common_errors=[
            TeacherCommonError(label=label, count=count)
            for label, count in all_errors.most_common(6)
        ],
        progress_distribution=progress_distribution,
        students=students,
    )
