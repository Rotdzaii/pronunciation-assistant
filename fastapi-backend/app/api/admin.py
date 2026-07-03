import csv
import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_supabase_service_client, require_roles
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AdminDeleteUserResponse(BaseModel):
    user_id: str
    deleted: bool
    message: str


class AdminAuditLog(BaseModel):
    id: str
    review_request_id: str | None = None
    actor_id: str
    actor_role: str
    action_type: str
    before_value: dict[str, Any] | None = None
    after_value: dict[str, Any] | None = None
    reason: str | None = None
    created_at: str | None = None


class AdminActivityLogsResponse(BaseModel):
    items: list[AdminAuditLog]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.delete("/users/{user_id}", response_model=AdminDeleteUserResponse)
def delete_user(
    user_id: str,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> AdminDeleteUserResponse:
    if user_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete your own account")

    supabase_client = get_supabase_service_client(settings)

    # Verify user exists before attempting deletion
    profile_rows = (
        supabase_client.table("profiles")
        .select("id,email,app_role")
        .eq("id", user_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not profile_rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    try:
        supabase_client.auth.admin.delete_user(user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user from authentication service",
        ) from exc

    return AdminDeleteUserResponse(user_id=user_id, deleted=True, message="User deleted successfully")


@router.get("/reports/export")
def export_system_report(
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        profiles = (
            supabase_client.table("profiles")
            .select("id,email,app_role,created_at")
            .execute()
            .data
            or []
        )
        history = (
            supabase_client.table("practice_history")
            .select("student_id,status,score,created_at")
            .execute()
            .data
            or []
        )
        try:
            classes = (
                supabase_client.table("classes")
                .select("id,name,status,created_at")
                .execute()
                .data
                or []
            )
        except Exception:
            classes = []
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to generate report") from exc

    from collections import Counter
    role_counts: Counter[str] = Counter(p.get("app_role", "unknown") for p in profiles)
    session_count = len(history)
    completed = [row for row in history if row.get("status") == "completed"]
    scores = [float(row["score"]) for row in completed if isinstance(row.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["metric", "value"])
    writer.writerow(["total_users", len(profiles)])
    writer.writerow(["students", role_counts.get("student", 0)])
    writer.writerow(["teachers", role_counts.get("teacher", 0)])
    writer.writerow(["admins", role_counts.get("admin", 0)])
    writer.writerow(["total_practice_sessions", session_count])
    writer.writerow(["completed_sessions", len(completed)])
    writer.writerow(["overall_average_score", avg_score])
    writer.writerow(["total_classes", len(classes)])
    writer.writerow(["active_classes", sum(1 for c in classes if c.get("status") == "active")])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=system_report.csv"},
    )


@router.get("/activity-logs", response_model=AdminActivityLogsResponse)
def get_activity_logs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor_id: str | None = Query(default=None),
    action_type: str | None = Query(default=None),
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> AdminActivityLogsResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        query = (
            supabase_client.table("review_audit_logs")
            .select("*")
            .order("created_at", desc=True)
        )
        if actor_id:
            query = query.eq("actor_id", actor_id)
        if action_type:
            query = query.eq("action_type", action_type)

        all_rows = query.execute().data or []
    except Exception as exc:
        message = str(exc).lower()
        if "review_audit_logs" in message and ("not found" in message or "does not exist" in message):
            return AdminActivityLogsResponse(items=[], total=0, limit=limit, offset=offset)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load activity logs") from exc

    total = len(all_rows)
    paged = all_rows[offset: offset + limit]

    return AdminActivityLogsResponse(
        items=[
            AdminAuditLog(
                id=str(row.get("id")),
                review_request_id=str(row["review_request_id"]) if row.get("review_request_id") else None,
                actor_id=str(row.get("actor_id")),
                actor_role=str(row.get("actor_role") or ""),
                action_type=str(row.get("action_type") or ""),
                before_value=row.get("before_value"),
                after_value=row.get("after_value"),
                reason=row.get("reason"),
                created_at=row.get("created_at"),
            )
            for row in paged
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
