from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_supabase_service_client, require_roles
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/student", tags=["Student Reviews"])


ReviewReason = Literal["ai_scored_wrong", "audio_issue", "result_mismatch", "other"]


class ReportPracticeResultPayload(BaseModel):
    reason: ReviewReason
    student_note: str | None = Field(default=None, max_length=1000)


class StudentReviewRequest(BaseModel):
    id: str
    practice_history_id: str
    reason: str
    student_note: str | None = None
    status: str
    teacher_note: str | None = None
    teacher_resolution: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class StudentReviewRequestDetail(StudentReviewRequest):
    source: str
    severity: str
    audit_logs: list[dict[str, Any]]


def _student_review_response(row: dict[str, Any]) -> StudentReviewRequest:
    return StudentReviewRequest(
        id=str(row.get("id")),
        practice_history_id=str(row.get("practice_history_id")),
        reason=str(row.get("reason") or "other"),
        student_note=row.get("student_note"),
        status=str(row.get("status") or "pending"),
        teacher_note=row.get("teacher_note"),
        teacher_resolution=row.get("teacher_resolution"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _load_owned_practice_history(
    supabase_client: Any,
    practice_history_id: str,
    student_id: str,
) -> dict[str, Any]:
    rows = (
        supabase_client.table("practice_history")
        .select("id,student_id,status,created_at")
        .eq("id", practice_history_id)
        .eq("student_id", student_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Practice history item not found",
        )
    return rows[0]


def _resolve_student_class_id(supabase_client: Any, student_id: str) -> str | None:
    rows = (
        supabase_client.table("student_classes")
        .select("class_id,joined_at")
        .eq("student_id", student_id)
        .eq("status", "active")
        .order("joined_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        return None
    class_id = rows[0].get("class_id")
    return str(class_id) if class_id else None


@router.post("/practice-history/{practice_history_id}/report", response_model=StudentReviewRequest)
def report_student_practice_result(
    practice_history_id: str,
    payload: ReportPracticeResultPayload,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentReviewRequest:
    supabase_client = get_supabase_service_client(settings)
    try:
        _load_owned_practice_history(supabase_client, practice_history_id, current_user.id)
        existing = (
            supabase_client.table("review_requests")
            .select("*")
            .eq("practice_history_id", practice_history_id)
            .eq("student_id", current_user.id)
            .eq("source", "student_report")
            .in_("status", ["pending", "in_review", "reanalyzing"])
            .limit(1)
            .execute()
            .data
            or []
        )
        if existing:
            return _student_review_response(existing[0])

        row = {
            "practice_history_id": practice_history_id,
            "student_id": current_user.id,
            "class_id": _resolve_student_class_id(supabase_client, current_user.id),
            "source": "student_report",
            "reason": payload.reason,
            "student_note": (payload.student_note or "").strip() or None,
            "severity": "medium",
            "status": "pending",
        }
        created = supabase_client.table("review_requests").insert(row).execute().data or []
        if not created:
            raise HTTPException(status_code=500, detail="Unable to create review request")
        created_row = created[0]
        supabase_client.table("review_audit_logs").insert(
            {
                "review_request_id": created_row["id"],
                "actor_id": current_user.id,
                "actor_role": "student",
                "action_type": "student_reported_result",
                "after_value": {
                    "reason": payload.reason,
                    "student_note": row["student_note"],
                    "status": "pending",
                },
            }
        ).execute()
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "review_requests" in message or "review_audit_logs" in message:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Review request tables are not available. Apply migration 007_create_review_requests.sql.",
            ) from exc
        raise HTTPException(status_code=500, detail="Unable to report practice result") from exc

    return _student_review_response(created_row)


@router.get("/review-requests", response_model=list[StudentReviewRequest])
def list_student_review_requests(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> list[StudentReviewRequest]:
    supabase_client = get_supabase_service_client(settings)
    rows = (
        supabase_client.table("review_requests")
        .select("*")
        .eq("student_id", current_user.id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    return [_student_review_response(row) for row in rows]


@router.get("/review-requests/{request_id}", response_model=StudentReviewRequestDetail)
def get_student_review_request(
    request_id: str,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentReviewRequestDetail:
    supabase_client = get_supabase_service_client(settings)
    rows = (
        supabase_client.table("review_requests")
        .select("*")
        .eq("id", request_id)
        .eq("student_id", current_user.id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Review request not found")
    row = rows[0]
    audit_logs = (
        supabase_client.table("review_audit_logs")
        .select("*")
        .eq("review_request_id", request_id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    base = _student_review_response(row)
    return StudentReviewRequestDetail(
        **base.model_dump(),
        source=str(row.get("source") or "student_report"),
        severity=str(row.get("severity") or "medium"),
        audit_logs=audit_logs,
    )
