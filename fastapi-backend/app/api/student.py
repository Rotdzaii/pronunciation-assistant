from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_supabase_service_client, require_roles
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/student", tags=["Student"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class StudentMasteredResponse(BaseModel):
    item_id: str
    student_id: str
    mastered_at: str | None = None


class StudentStreakResponse(BaseModel):
    current_streak: int
    longest_streak: int
    last_practice_date: str | None = None


class PhonemeErrorEntry(BaseModel):
    phoneme: str
    error_count: int


class StudentPhonemeMapResponse(BaseModel):
    phonemes: list[PhonemeErrorEntry]


class StudentDashboardResponse(BaseModel):
    total_sessions: int
    completed_sessions: int
    average_score: float | None = None
    current_streak: int
    longest_streak: int
    last_practice_date: str | None = None
    top_problem_phonemes: list[PhonemeErrorEntry]


class StudentGoalResponse(BaseModel):
    id: str
    student_id: str
    daily_target: int
    created_at: str | None = None
    updated_at: str | None = None


class StudentGoalCreate(BaseModel):
    daily_target: int = Field(..., ge=1, le=100)


class StudentGoalUpdate(BaseModel):
    daily_target: int = Field(..., ge=1, le=100)


class StudentBadge(BaseModel):
    id: str
    name: str
    description: str
    earned: bool


class StudentBadgesResponse(BaseModel):
    badges: list[StudentBadge]


class LeaderboardEntry(BaseModel):
    rank: int
    student_id: str
    display_name: str
    average_score: float | None = None
    total_sessions: int
    is_self: bool = False


class StudentLeaderboardResponse(BaseModel):
    class_id: str
    entries: list[LeaderboardEntry]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BADGE_DEFINITIONS = [
    {"id": "streak_3",    "name": "Consistent Learner",  "description": "Practice 3 days in a row",      "type": "streak",   "threshold": 3},
    {"id": "streak_7",    "name": "Weekly Champion",     "description": "Practice 7 days in a row",      "type": "streak",   "threshold": 7},
    {"id": "streak_30",   "name": "Monthly Master",      "description": "Practice 30 days in a row",     "type": "streak",   "threshold": 30},
    {"id": "accuracy_70", "name": "Good Start",          "description": "Achieve 70% average accuracy",  "type": "accuracy", "threshold": 70},
    {"id": "accuracy_80", "name": "Skilled Speaker",     "description": "Achieve 80% average accuracy",  "type": "accuracy", "threshold": 80},
    {"id": "accuracy_90", "name": "Excellence Award",    "description": "Achieve 90% average accuracy",  "type": "accuracy", "threshold": 90},
    {"id": "sessions_10", "name": "Getting Started",     "description": "Complete 10 practice sessions", "type": "sessions", "threshold": 10},
    {"id": "sessions_50", "name": "Dedicated Learner",   "description": "Complete 50 practice sessions", "type": "sessions", "threshold": 50},
    {"id": "sessions_100","name": "Century Club",        "description": "Complete 100 practice sessions","type": "sessions", "threshold": 100},
]


def _calculate_streak(practice_rows: list[dict[str, Any]]) -> tuple[int, int, str | None]:
    """Returns (current_streak, longest_streak, last_practice_date_str)."""
    date_strings = sorted({
        row["created_at"][:10]
        for row in practice_rows
        if row.get("created_at") and len(row["created_at"]) >= 10
    })
    if not date_strings:
        return 0, 0, None

    last_date_str = date_strings[-1]
    today = date.today()
    last_date = date.fromisoformat(last_date_str)

    current_streak = 0
    if last_date >= today - timedelta(days=1):
        expected = last_date
        for d_str in reversed(date_strings):
            d = date.fromisoformat(d_str)
            if d == expected:
                current_streak += 1
                expected -= timedelta(days=1)
            elif d < expected:
                break

    # Longest streak over full history
    longest = 1
    current = 1
    for i in range(1, len(date_strings)):
        prev = date.fromisoformat(date_strings[i - 1])
        curr = date.fromisoformat(date_strings[i])
        if curr == prev + timedelta(days=1):
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    longest = max(longest, current)

    return current_streak, longest, last_date_str


def _extract_phoneme_label(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        for key in ("phoneme", "type", "label", "message"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _aggregate_phoneme_errors(practice_rows: list[dict[str, Any]]) -> list[PhonemeErrorEntry]:
    counts: Counter[str] = Counter()
    for row in practice_rows:
        phonemes = row.get("problem_phonemes")
        if not isinstance(phonemes, list):
            continue
        for p in phonemes:
            label = _extract_phoneme_label(p)
            if label:
                counts[label] += 1
    return [PhonemeErrorEntry(phoneme=label, error_count=count) for label, count in counts.most_common()]


def _derive_display_name(profile: dict[str, Any]) -> str:
    for key in ("display_name", "full_name", "name"):
        value = profile.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    email = profile.get("email", "")
    if isinstance(email, str) and "@" in email:
        local = email.split("@", 1)[0]
        words = [w for w in local.replace("_", ".").split(".") if w]
        if words:
            return " ".join(w.capitalize() for w in words)
    return "Học viên"


def _load_student_practice(supabase_client: Any, student_id: str) -> list[dict[str, Any]]:
    return (
        supabase_client.table("practice_history")
        .select("id,student_id,status,score,problem_phonemes,created_at")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.patch("/vocabulary-items/{item_id}/mastered", response_model=StudentMasteredResponse)
def mark_vocabulary_item_mastered(
    item_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentMasteredResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = (
            supabase_client.table("student_vocabulary_mastery")
            .upsert(
                {"student_id": current_user.id, "item_id": str(item_id), "mastered_at": datetime.now(tz=UTC).isoformat()},
                on_conflict="student_id,item_id",
            )
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to mark item as mastered") from exc
    row = rows[0] if rows else {"student_id": current_user.id, "item_id": str(item_id)}
    return StudentMasteredResponse(
        item_id=str(item_id),
        student_id=current_user.id,
        mastered_at=row.get("mastered_at"),
    )


@router.get("/streak", response_model=StudentStreakResponse)
def get_student_streak(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentStreakResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = _load_student_practice(supabase_client, current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load practice history") from exc
    current_streak, longest_streak, last_date = _calculate_streak(rows)
    return StudentStreakResponse(
        current_streak=current_streak,
        longest_streak=longest_streak,
        last_practice_date=last_date,
    )


@router.get("/dashboard", response_model=StudentDashboardResponse)
def get_student_dashboard(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentDashboardResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = _load_student_practice(supabase_client, current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load dashboard data") from exc

    completed = [row for row in rows if row.get("status") == "completed"]
    scores = [float(row["score"]) for row in completed if isinstance(row.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    current_streak, longest_streak, last_date = _calculate_streak(rows)
    top_phonemes = _aggregate_phoneme_errors(rows)[:5]

    return StudentDashboardResponse(
        total_sessions=len(rows),
        completed_sessions=len(completed),
        average_score=avg_score,
        current_streak=current_streak,
        longest_streak=longest_streak,
        last_practice_date=last_date,
        top_problem_phonemes=top_phonemes,
    )


@router.get("/phoneme-map", response_model=StudentPhonemeMapResponse)
def get_student_phoneme_map(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentPhonemeMapResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = _load_student_practice(supabase_client, current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load phoneme data") from exc
    return StudentPhonemeMapResponse(phonemes=_aggregate_phoneme_errors(rows))


@router.post("/goals", response_model=StudentGoalResponse, status_code=status.HTTP_201_CREATED)
def create_student_goal(
    payload: StudentGoalCreate,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentGoalResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = (
            supabase_client.table("student_goals")
            .insert({"student_id": current_user.id, "daily_target": payload.daily_target})
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create goal") from exc
    if not rows:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Goal creation returned no data")
    return StudentGoalResponse(**rows[0])


@router.patch("/goals", response_model=StudentGoalResponse)
def update_student_goal(
    payload: StudentGoalUpdate,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentGoalResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = (
            supabase_client.table("student_goals")
            .update({"daily_target": payload.daily_target, "updated_at": datetime.now(tz=UTC).isoformat()})
            .eq("student_id", current_user.id)
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to update goal") from exc
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No goal found; use POST /student/goals to create one")
    return StudentGoalResponse(**rows[0])


@router.get("/badges", response_model=StudentBadgesResponse)
def get_student_badges(
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentBadgesResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        rows = _load_student_practice(supabase_client, current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load badge data") from exc

    completed = [row for row in rows if row.get("status") == "completed"]
    scores = [float(row["score"]) for row in completed if isinstance(row.get("score"), (int, float))]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    current_streak, _, _ = _calculate_streak(rows)
    completed_count = len(completed)

    badges: list[StudentBadge] = []
    for defn in _BADGE_DEFINITIONS:
        badge_type = defn["type"]
        threshold = defn["threshold"]
        if badge_type == "streak":
            earned = current_streak >= threshold
        elif badge_type == "accuracy":
            earned = avg_score >= threshold
        else:
            earned = completed_count >= threshold
        badges.append(StudentBadge(id=defn["id"], name=defn["name"], description=defn["description"], earned=earned))

    return StudentBadgesResponse(badges=badges)


@router.get("/leaderboard/{class_id}", response_model=StudentLeaderboardResponse)
def get_class_leaderboard(
    class_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["student"])),
    settings: Settings = Depends(get_settings),
) -> StudentLeaderboardResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        memberships = (
            supabase_client.table("student_classes")
            .select("student_id")
            .eq("class_id", str(class_id))
            .eq("status", "active")
            .execute()
            .data
            or []
        )
        student_ids = [str(row["student_id"]) for row in memberships if row.get("student_id")]
        if not student_ids:
            return StudentLeaderboardResponse(class_id=str(class_id), entries=[])

        # Verify requesting student is in this class
        if current_user.id not in student_ids and current_user.app_role != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not a member of this class")

        profiles_rows = (
            supabase_client.table("profiles")
            .select("id,display_name,full_name,email")
            .in_("id", student_ids)
            .execute()
            .data
            or []
        )
        profiles_by_id = {str(p["id"]): p for p in profiles_rows}

        history_rows = (
            supabase_client.table("practice_history")
            .select("student_id,status,score")
            .in_("student_id", student_ids)
            .eq("status", "completed")
            .execute()
            .data
            or []
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load leaderboard") from exc

    scores_by_student: dict[str, list[float]] = {}
    sessions_by_student: Counter[str] = Counter()
    for row in history_rows:
        sid = str(row["student_id"])
        sessions_by_student[sid] += 1
        if isinstance(row.get("score"), (int, float)):
            scores_by_student.setdefault(sid, []).append(float(row["score"]))

    ranked = sorted(
        student_ids,
        key=lambda sid: (
            -(sum(scores_by_student.get(sid, [])) / len(scores_by_student[sid])) if scores_by_student.get(sid) else 0.0
        ),
    )

    entries: list[LeaderboardEntry] = []
    for rank, sid in enumerate(ranked, start=1):
        scores = scores_by_student.get(sid, [])
        avg = round(sum(scores) / len(scores), 1) if scores else None
        entries.append(LeaderboardEntry(
            rank=rank,
            student_id=sid,
            display_name=_derive_display_name(profiles_by_id.get(sid, {})),
            average_score=avg,
            total_sessions=sessions_by_student[sid],
            is_self=(sid == current_user.id),
        ))

    return StudentLeaderboardResponse(class_id=str(class_id), entries=entries)
