"""Authoritative assignment progress and grade calculation.

The service is intentionally called from the AI webhook and read-side
assignment endpoints.  That makes webhook retries safe: a recalculation is a
replacement of derived state, never an increment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


WORK_STATUSES = {"not_started", "in_progress", "completed", "submitted", "overdue", "locked"}
GRADING_STATUSES = {"pending", "provisional", "processing", "graded", "needs_review"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def is_past_deadline(assignment: dict[str, Any], now: datetime | None = None) -> bool:
    deadline = parse_timestamp(assignment.get("deadline") or assignment.get("due_date"))
    return bool(deadline and deadline <= (now or utcnow()))


def assignment_item_rows(supabase_client: Any, assignment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the stable vocabulary-item IDs assigned by a vocabulary set."""
    if assignment.get("content_type") != "vocabulary_set":
        return []
    return (
        supabase_client.table("vocabulary_set_items")
        .select("item_id,sort_order,vocabulary_items(id,word,phonetic,meaning_vi,topic,level)")
        .eq("set_id", str(assignment["content_id"]))
        .order("sort_order")
        .execute()
        .data
        or []
    )


def assignment_item_map(supabase_client: Any, assignment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in assignment_item_rows(supabase_client, assignment)}


def is_recipient(supabase_client: Any, assignment_id: str, student_id: str) -> bool:
    rows = (
        supabase_client.table("assignment_recipients")
        .select("id")
        .eq("assignment_id", assignment_id)
        .eq("student_id", student_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    return bool(rows)


def _get_one(supabase_client: Any, table: str, **filters: str) -> dict[str, Any] | None:
    query = supabase_client.table(table).select("*")
    for column, value in filters.items():
        query = query.eq(column, value)
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else None


def _upsert_grade(supabase_client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    existing = _get_one(
        supabase_client,
        "assignment_grades",
        assignment_id=str(payload["assignment_id"]),
        student_id=str(payload["student_id"]),
    )
    if existing:
        rows = supabase_client.table("assignment_grades").update(payload).eq("id", existing["id"]).execute().data or []
    else:
        rows = supabase_client.table("assignment_grades").insert(payload).execute().data or []
    return rows[0] if rows else {**(existing or {}), **payload}


def _save_progress(
    supabase_client: Any,
    assignment_id: str,
    student_id: str,
    total_items: int,
    completed_items: int,
    status: str,
) -> dict[str, Any]:
    payload = {
        "completed_items": completed_items,
        "total_items": total_items,
        "status": status,
        "completed_at": utcnow().isoformat() if status in {"completed", "submitted", "locked"} else None,
        "updated_at": utcnow().isoformat(),
    }
    existing = _get_one(supabase_client, "assignment_progress", assignment_id=assignment_id, student_id=student_id)
    if existing:
        rows = supabase_client.table("assignment_progress").update(payload).eq("id", existing["id"]).execute().data or []
    else:
        rows = supabase_client.table("assignment_progress").insert({"assignment_id": assignment_id, "student_id": student_id, **payload}).execute().data or []
    return rows[0] if rows else {**(existing or {}), **payload}


def sweep_assignment_state(
    supabase_client: Any,
    assignment: dict[str, Any],
    student_id: str,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Lock an open assessment after its server-side deadline and recalculate."""
    now = now or utcnow()
    submission = _get_one(
        supabase_client,
        "assessment_submissions",
        assignment_id=str(assignment["id"]),
        student_id=student_id,
    )
    if assignment.get("is_assessment") and submission and not submission.get("is_locked") and is_past_deadline(assignment, now):
        rows = (
            supabase_client.table("assessment_submissions")
            .update({"submitted_at": now.isoformat(), "is_locked": True, "updated_at": now.isoformat()})
            .eq("id", submission["id"])
            .execute()
            .data
            or []
        )
        submission = rows[0] if rows else submission
    recalculate_assignment_grade(supabase_client, assignment, student_id, now=now, submission=submission)
    return submission


def recalculate_assignment_grade(
    supabase_client: Any,
    assignment: dict[str, Any],
    student_id: str,
    *,
    now: datetime | None = None,
    submission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive progress and grade from verified AI job rows.

    A job is eligible iff it was created no later than the server deadline.
    Results arriving after the deadline therefore still count, while a job
    created after it never can.  Re-running this calculation is idempotent.
    """
    now = now or utcnow()
    assignment_id = str(assignment["id"])
    is_assessment = bool(assignment.get("is_assessment"))
    required = assignment_item_map(supabase_client, assignment)
    required_ids = set(required)
    deadline = parse_timestamp(assignment.get("deadline") or assignment.get("due_date"))
    if submission is None and assignment.get("is_assessment"):
        submission = _get_one(supabase_client, "assessment_submissions", assignment_id=assignment_id, student_id=student_id)

    jobs_query = (
        supabase_client.table("practice_history")
        .select("id,item_id,status,score,created_at,assessment_submission_id")
        .eq("assignment_id", assignment_id)
        .eq("student_id", student_id)
    )
    if is_assessment and submission:
        jobs_query = jobs_query.eq("assessment_submission_id", str(submission["id"]))
    jobs = jobs_query.execute().data or []
    eligible = [job for job in jobs if str(job.get("item_id") or "") in required_ids and (not deadline or (parse_timestamp(job.get("created_at")) or now) <= deadline)]
    completed_by_item: dict[str, float] = {}
    failed_items: set[str] = set()
    processing = False
    for job in eligible:
        item_id = str(job.get("item_id"))
        if job.get("status") == "completed" and job.get("score") is not None:
            completed_by_item[item_id] = max(completed_by_item.get(item_id, 0.0), float(job["score"]))
        elif job.get("status") == "completed":
            # An AI response without a score is not a student omission and
            # must not silently become a zero.
            failed_items.add(item_id)
        elif job.get("status") == "failed":
            failed_items.add(item_id)
        else:
            processing = True

    completed_items = len(completed_by_item)
    total_items = len(required_ids)
    past_due = is_past_deadline(assignment, now)
    failed_without_success = failed_items - set(completed_by_item)
    locked = bool(submission and submission.get("is_locked"))
    submitted = bool(submission and submission.get("submitted_at"))

    if is_assessment:
        if not submission:
            work_status = "overdue" if past_due else "not_started"
        elif locked:
            work_status = "locked"
        elif submitted:
            work_status = "submitted"
        else:
            work_status = "in_progress"
    elif past_due:
        work_status = "overdue"
    elif completed_items >= total_items and total_items > 0:
        work_status = "completed"
    elif completed_items or processing:
        work_status = "in_progress"
    else:
        work_status = "not_started"

    can_calculate_assessment = not is_assessment or locked or submitted
    auto_score: float | None = None
    if total_items and can_calculate_assessment and (completed_items or past_due or locked or submitted):
        auto_score = round(sum(completed_by_item.get(item_id, 0.0) for item_id in required_ids) / total_items, 2)

    # A completed AI job without a score is indeterminate, rather than an
    # attempted item worth zero. Leave the score unset for teacher review.
    if failed_without_success:
        auto_score = None

    if failed_without_success:
        grading_status = "needs_review"
    elif is_assessment and (submitted or locked) and processing:
        grading_status = "processing"
    elif is_assessment and not (submitted or locked):
        grading_status = "pending"
    elif past_due or (is_assessment and (submitted or locked)):
        grading_status = "graded" if auto_score is not None else "pending"
    elif completed_items or processing:
        grading_status = "processing" if processing else "provisional"
    else:
        grading_status = "pending"

    current_grade = _get_one(supabase_client, "assignment_grades", assignment_id=assignment_id, student_id=student_id) or {}
    override = current_grade.get("teacher_override_score")
    final_score = float(override) if override is not None else auto_score
    if override is not None:
        grading_status = "graded"

    progress = _save_progress(supabase_client, assignment_id, student_id, total_items, completed_items, work_status)
    grade = _upsert_grade(
        supabase_client,
        {
            "assignment_id": assignment_id,
            "student_id": student_id,
            "assessment_submission_id": submission.get("id") if submission else current_grade.get("assessment_submission_id"),
            "auto_score": auto_score,
            "final_score": final_score,
            "grading_status": grading_status,
            "graded_at": now.isoformat() if grading_status == "graded" else current_grade.get("graded_at"),
            "details": {
                "formula": "sum(best_score_per_required_word) / total_required_words",
                "best_scores": completed_by_item,
                "failed_item_ids": sorted(failed_without_success),
                "calculated_at": now.isoformat(),
            },
            "updated_at": now.isoformat(),
        },
    )
    return {"progress": progress, "grade": grade, "submission": submission, "best_scores": completed_by_item}
