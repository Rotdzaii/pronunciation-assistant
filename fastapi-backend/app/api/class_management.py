import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
import httpx
from pydantic import BaseModel

from app.core.auth import (
    CurrentUser,
    get_current_user,
    get_supabase_service_client,
    require_roles,
)
from app.core.config import Settings, get_settings


student_router = APIRouter(prefix="/student/classes", tags=["Student Classes"])
teacher_router = APIRouter(prefix="/teacher/classes", tags=["Teacher Classes"])
admin_router = APIRouter(prefix="/admin", tags=["Admin"])


logger = logging.getLogger(__name__)


PROFILE_SELECT = "id,email,app_role,display_name,avatar_url"
ADMIN_PROFILE_SELECT = "id,email,app_role,display_name,created_at,updated_at"
CLASS_SELECT = "id,code,name,description,status,teacher_id"
ADMIN_LIST_RANGE_END = 9999
PROTECTED_DEMO_EMAILS = {
    "admin@gmail.com",
    "hocsinh07@phoenix.edu.vn",
    "giangvien03@phoenix.edu.vn",
}


class TeacherBrief(BaseModel):
    id: str
    email: str | None = None
    display_name: str
    teacher_role: str | None = None
    status: str | None = None


class StudentBrief(BaseModel):
    id: str
    email: str | None = None
    display_name: str
    avatar_url: str | None = None
    joined_at: str | None = None
    status: str | None = None


class StudentClassListItem(BaseModel):
    id: str
    code: str | None = None
    name: str
    description: str | None = None
    status: str | None = None
    student_count: int
    teacher_count: int
    teachers: list[TeacherBrief]


class StudentClassDetail(StudentClassListItem):
    students: list[StudentBrief] = []


class TeacherClassListItem(BaseModel):
    id: str
    code: str | None = None
    name: str
    description: str | None = None
    teacher_role: str | None = None
    student_count: int
    teacher_count: int


class TeacherClassDetail(TeacherClassListItem):
    status: str | None = None
    teachers: list[TeacherBrief]
    students: list[StudentBrief]


class AdminUserProfile(BaseModel):
    id: str
    email: str | None = None
    app_role: str | None = None
    display_name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AdminClassListItem(BaseModel):
    id: str
    code: str | None = None
    name: str
    description: str | None = None
    status: str | None = None
    student_count: int
    teacher_count: int


class AdminClassDetail(AdminClassListItem):
    students: list[StudentBrief]
    teachers: list[TeacherBrief]


class DemoReadinessCheck(BaseModel):
    key: str
    passed: bool
    expected: Any
    actual: Any
    message: str


class DemoReadinessResponse(BaseModel):
    passed: bool
    checks: list[DemoReadinessCheck]


class AdminClassUpdate(BaseModel):
    code: str | None = None
    name: str | None = None
    description: str | None = None
    status: str | None = None


class AdminDeleteUserResponse(BaseModel):
    deleted: bool
    user_id: str
    email: str | None = None
    auth_user_deleted: bool


def _server_error(detail: str, exc: Exception) -> HTTPException:
    logger.exception(detail)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )


def _require_exact_role(role: str):
    def role_dependency(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.app_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )
        return current_user

    return role_dependency


def _display_name(profile: dict[str, Any]) -> str:
    value = profile.get("display_name")
    if isinstance(value, str) and value.strip():
        return value.strip()

    email = profile.get("email")
    if isinstance(email, str) and email.strip():
        return email.split("@", 1)[0]

    return "Unknown user"


def _profile_by_id(profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(profile["id"]): profile
        for profile in profiles
        if profile.get("id") is not None
    }


def _classes_by_id(classes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(class_item["id"]): class_item
        for class_item in classes
        if class_item.get("id") is not None
    }


def _ensure_class_exists(supabase_client: Any, class_id: str) -> dict[str, Any]:
    response = (
        supabase_client.table("classes")
        .select(CLASS_SELECT)
        .eq("id", class_id)
        .limit(1)
        .execute()
    )
    data = response.data or []
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Class not found",
        )
    return data[0]


def _require_student_membership(supabase_client: Any, class_id: str, student_id: str) -> None:
    response = (
        supabase_client.table("student_classes")
        .select("class_id")
        .eq("class_id", class_id)
        .eq("student_id", student_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if not (response.data or []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student is not active in this class",
        )


def _require_teacher_membership(
    supabase_client: Any, class_id: str, teacher_id: str
) -> dict[str, Any]:
    response = (
        supabase_client.table("teacher_classes")
        .select("class_id,teacher_id,teacher_role,status")
        .eq("class_id", class_id)
        .eq("teacher_id", teacher_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    data = response.data or []
    if data:
        return data[0]

    legacy_response = (
        supabase_client.table("classes")
        .select("id,teacher_id,status")
        .eq("id", class_id)
        .eq("teacher_id", teacher_id)
        .eq("status", "active")
        .limit(1)
        .execute()
    )
    if legacy_response.data or []:
        return {
            "class_id": class_id,
            "teacher_id": teacher_id,
            "teacher_role": "owner",
            "status": "active",
        }

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Teacher is not active in this class",
    )


def _fetch_active_student_memberships(
    supabase_client: Any, class_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    query = (
        supabase_client.table("student_classes")
        .select("class_id,student_id,joined_at,status")
        .eq("status", "active")
    )
    if class_ids is not None:
        if not class_ids:
            return []
        query = query.in_("class_id", class_ids)
    return query.execute().data or []


def _fetch_active_teacher_memberships(
    supabase_client: Any, class_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    query = (
        supabase_client.table("teacher_classes")
        .select("class_id,teacher_id,teacher_role,status")
        .eq("status", "active")
    )
    if class_ids is not None:
        if not class_ids:
            return []
        query = query.in_("class_id", class_ids)
    return query.execute().data or []


def _fetch_profiles(supabase_client: Any, user_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = sorted({user_id for user_id in user_ids if user_id})
    if not unique_ids:
        return {}
    response = (
        supabase_client.table("profiles")
        .select(PROFILE_SELECT)
        .in_("id", unique_ids)
        .execute()
    )
    return _profile_by_id(response.data or [])


def _fetch_classes(supabase_client: Any, class_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_ids = sorted({class_id for class_id in class_ids if class_id})
    if not unique_ids:
        return {}
    response = (
        supabase_client.table("classes")
        .select(CLASS_SELECT)
        .in_("id", unique_ids)
        .execute()
    )
    return _classes_by_id(response.data or [])


def _fetch_legacy_teacher_classes(supabase_client: Any, teacher_id: str) -> list[dict[str, Any]]:
    response = (
        supabase_client.table("classes")
        .select(CLASS_SELECT)
        .eq("teacher_id", teacher_id)
        .eq("status", "active")
        .execute()
    )
    return response.data or []


def _execute_admin_profiles_query(supabase_client: Any, role: str | None) -> list[dict[str, Any]]:
    query = supabase_client.table("profiles").select(ADMIN_PROFILE_SELECT).range(0, ADMIN_LIST_RANGE_END)
    if role:
        query = query.eq("app_role", role)

    try:
        return query.execute().data or []
    except Exception:
        logger.warning(
            "Unable to select optional profile timestamp columns; falling back to base profile fields",
            exc_info=True,
        )

    fallback_query = supabase_client.table("profiles").select(PROFILE_SELECT).range(0, ADMIN_LIST_RANGE_END)
    if role:
        fallback_query = fallback_query.eq("app_role", role)
    return fallback_query.execute().data or []


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _delete_auth_user(settings: Settings, user_id: str) -> None:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase service role is not configured on the backend",
        )

    auth_url = settings.supabase_url.rstrip("/") + f"/auth/v1/admin/users/{user_id}"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }
    with httpx.Client(timeout=10.0) as client:
        response = client.delete(auth_url, headers=headers)

    if response.status_code not in (status.HTTP_200_OK, status.HTTP_204_NO_CONTENT, status.HTTP_404_NOT_FOUND):
        logger.error(
            "Unable to delete Supabase auth user user_id=%s status=%s body=%s",
            user_id,
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to delete Supabase auth user",
        )


def _delete_optional_user_links(supabase_client: Any, user_id: str) -> None:
    optional_deletes = [
        ("teacher_class_invitations", "invited_by"),
        ("teacher_class_invitations", "teacher_id"),
        ("teacher_class_invitations", "teacher_profile_id"),
    ]
    for table_name, column_name in optional_deletes:
        try:
            supabase_client.table(table_name).delete().eq(column_name, user_id).execute()
        except Exception:
            logger.info(
                "Skipping optional delete table=%s column=%s user_id=%s",
                table_name,
                column_name,
                user_id,
                exc_info=True,
            )


def _count_by_class(memberships: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in memberships:
        class_id = item.get("class_id")
        if class_id is None:
            continue
        key = str(class_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _teachers_for_class(
    teacher_memberships: list[dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
    class_id: str,
) -> list[TeacherBrief]:
    teachers: list[TeacherBrief] = []
    for membership in teacher_memberships:
        if str(membership.get("class_id")) != class_id:
            continue
        teacher_id = str(membership.get("teacher_id"))
        profile = profiles_by_id.get(teacher_id, {})
        teachers.append(
            TeacherBrief(
                id=teacher_id,
                email=profile.get("email"),
                display_name=_display_name(profile),
                teacher_role=membership.get("teacher_role"),
                status=membership.get("status"),
            )
        )
    teachers.sort(key=lambda teacher: teacher.display_name.lower())
    return teachers


def _students_for_class(
    student_memberships: list[dict[str, Any]],
    profiles_by_id: dict[str, dict[str, Any]],
    class_id: str,
) -> list[StudentBrief]:
    students: list[StudentBrief] = []
    for membership in student_memberships:
        if str(membership.get("class_id")) != class_id:
            continue
        student_id = str(membership.get("student_id"))
        profile = profiles_by_id.get(student_id, {})
        students.append(
            StudentBrief(
                id=student_id,
                email=profile.get("email"),
                display_name=_display_name(profile),
                avatar_url=profile.get("avatar_url"),
                joined_at=membership.get("joined_at"),
                status=membership.get("status"),
            )
        )
    students.sort(key=lambda student: student.display_name.lower())
    return students


def _load_class_detail(supabase_client: Any, class_item: dict[str, Any]) -> AdminClassDetail:
    class_id = str(class_item["id"])
    student_memberships = _fetch_active_student_memberships(supabase_client, [class_id])
    teacher_memberships = _fetch_active_teacher_memberships(supabase_client, [class_id])
    student_profiles = _fetch_profiles(
        supabase_client,
        [str(item.get("student_id")) for item in student_memberships if item.get("student_id")],
    )
    teacher_profiles = _fetch_profiles(
        supabase_client,
        [str(item.get("teacher_id")) for item in teacher_memberships if item.get("teacher_id")],
    )
    students = _students_for_class(student_memberships, student_profiles, class_id)
    teachers = _teachers_for_class(teacher_memberships, teacher_profiles, class_id)
    return AdminClassDetail(
        id=class_id,
        code=class_item.get("code"),
        name=class_item.get("name") or "",
        description=class_item.get("description"),
        status=class_item.get("status"),
        student_count=len(students),
        teacher_count=len(teachers),
        students=students,
        teachers=teachers,
    )


@student_router.get("", response_model=list[StudentClassListItem])
def list_student_classes(
    current_user: CurrentUser = Depends(_require_exact_role("student")),
    settings: Settings = Depends(get_settings),
) -> list[StudentClassListItem]:
    supabase_client = get_supabase_service_client(settings)
    try:
        own_memberships = (
            supabase_client.table("student_classes")
            .select("class_id,student_id,status")
            .eq("student_id", current_user.id)
            .eq("status", "active")
            .execute()
            .data
            or []
        )
        class_ids = [str(item["class_id"]) for item in own_memberships if item.get("class_id")]
        classes = _fetch_classes(supabase_client, class_ids)
        student_memberships = _fetch_active_student_memberships(supabase_client, class_ids)
        teacher_memberships = _fetch_active_teacher_memberships(supabase_client, class_ids)
        teacher_profiles = _fetch_profiles(
            supabase_client,
            [str(item.get("teacher_id")) for item in teacher_memberships if item.get("teacher_id")],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load student classes", exc) from exc

    student_counts = _count_by_class(student_memberships)
    teacher_counts = _count_by_class(teacher_memberships)
    items: list[StudentClassListItem] = []
    for class_id in class_ids:
        class_item = classes.get(class_id)
        if not class_item:
            continue
        items.append(
            StudentClassListItem(
                id=class_id,
                code=class_item.get("code"),
                name=class_item.get("name") or "",
                description=class_item.get("description"),
                status=class_item.get("status"),
                student_count=student_counts.get(class_id, 0),
                teacher_count=teacher_counts.get(class_id, 0),
                teachers=_teachers_for_class(teacher_memberships, teacher_profiles, class_id),
            )
        )
    items.sort(key=lambda item: (item.code or item.name).lower())
    return items


@student_router.get("/{class_id}", response_model=StudentClassDetail)
def get_student_class(
    class_id: str,
    current_user: CurrentUser = Depends(_require_exact_role("student")),
    settings: Settings = Depends(get_settings),
) -> StudentClassDetail:
    supabase_client = get_supabase_service_client(settings)
    try:
        class_item = _ensure_class_exists(supabase_client, class_id)
        _require_student_membership(supabase_client, class_id, current_user.id)
        detail = _load_class_detail(supabase_client, class_item)
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load student class", exc) from exc

    return StudentClassDetail(
        id=detail.id,
        code=detail.code,
        name=detail.name,
        description=detail.description,
        status=detail.status,
        student_count=detail.student_count,
        teacher_count=detail.teacher_count,
        teachers=detail.teachers,
        students=detail.students,
    )


@teacher_router.get("", response_model=list[TeacherClassListItem])
def list_teacher_classes(
    current_user: CurrentUser = Depends(_require_exact_role("teacher")),
    settings: Settings = Depends(get_settings),
) -> list[TeacherClassListItem]:
    supabase_client = get_supabase_service_client(settings)
    try:
        own_memberships = (
            supabase_client.table("teacher_classes")
            .select("class_id,teacher_id,teacher_role,status")
            .eq("teacher_id", current_user.id)
            .eq("status", "active")
            .execute()
            .data
            or []
        )
        membership_class_ids = [str(item["class_id"]) for item in own_memberships if item.get("class_id")]
        legacy_classes = _fetch_legacy_teacher_classes(supabase_client, current_user.id)
        legacy_class_ids = [str(item["id"]) for item in legacy_classes if item.get("id")]
        class_ids = sorted({*membership_class_ids, *legacy_class_ids})
        classes = _fetch_classes(supabase_client, class_ids)
        for legacy_class in legacy_classes:
            if legacy_class.get("id"):
                classes[str(legacy_class["id"])] = legacy_class
        student_memberships = _fetch_active_student_memberships(supabase_client, class_ids)
        teacher_memberships = _fetch_active_teacher_memberships(supabase_client, class_ids)
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load teacher classes", exc) from exc

    student_counts = _count_by_class(student_memberships)
    teacher_counts = _count_by_class(teacher_memberships)
    teacher_role_by_class = {
        str(item["class_id"]): item.get("teacher_role")
        for item in own_memberships
        if item.get("class_id")
    }
    items: list[TeacherClassListItem] = []
    for class_id in class_ids:
        class_item = classes.get(class_id)
        if not class_item:
            continue
        items.append(
            TeacherClassListItem(
                id=class_id,
                code=class_item.get("code"),
                name=class_item.get("name") or "",
                description=class_item.get("description"),
                teacher_role=teacher_role_by_class.get(class_id) or "owner",
                student_count=student_counts.get(class_id, 0),
                teacher_count=teacher_counts.get(class_id, 0),
            )
        )
    items.sort(key=lambda item: (item.code or item.name).lower())
    return items


@teacher_router.get("/{class_id}", response_model=TeacherClassDetail)
def get_teacher_class(
    class_id: str,
    current_user: CurrentUser = Depends(_require_exact_role("teacher")),
    settings: Settings = Depends(get_settings),
) -> TeacherClassDetail:
    supabase_client = get_supabase_service_client(settings)
    try:
        class_item = _ensure_class_exists(supabase_client, class_id)
        membership = _require_teacher_membership(supabase_client, class_id, current_user.id)
        detail = _load_class_detail(supabase_client, class_item)
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load teacher class", exc) from exc

    return TeacherClassDetail(
        id=detail.id,
        code=detail.code,
        name=detail.name,
        description=detail.description,
        teacher_role=membership.get("teacher_role"),
        student_count=detail.student_count,
        teacher_count=detail.teacher_count,
        status=detail.status,
        students=detail.students,
        teachers=detail.teachers,
    )


@teacher_router.get("/{class_id}/students", response_model=list[StudentBrief])
def list_teacher_class_students(
    class_id: str,
    current_user: CurrentUser = Depends(_require_exact_role("teacher")),
    settings: Settings = Depends(get_settings),
) -> list[StudentBrief]:
    supabase_client = get_supabase_service_client(settings)
    try:
        _ensure_class_exists(supabase_client, class_id)
        _require_teacher_membership(supabase_client, class_id, current_user.id)
        student_memberships = _fetch_active_student_memberships(supabase_client, [class_id])
        student_profiles = _fetch_profiles(
            supabase_client,
            [str(item.get("student_id")) for item in student_memberships if item.get("student_id")],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load class students", exc) from exc

    return _students_for_class(student_memberships, student_profiles, class_id)


@teacher_router.get("/{class_id}/teachers", response_model=list[TeacherBrief])
def list_teacher_class_teachers(
    class_id: str,
    current_user: CurrentUser = Depends(_require_exact_role("teacher")),
    settings: Settings = Depends(get_settings),
) -> list[TeacherBrief]:
    supabase_client = get_supabase_service_client(settings)
    try:
        _ensure_class_exists(supabase_client, class_id)
        _require_teacher_membership(supabase_client, class_id, current_user.id)
        teacher_memberships = _fetch_active_teacher_memberships(supabase_client, [class_id])
        teacher_profiles = _fetch_profiles(
            supabase_client,
            [str(item.get("teacher_id")) for item in teacher_memberships if item.get("teacher_id")],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load class teachers", exc) from exc

    return _teachers_for_class(teacher_memberships, teacher_profiles, class_id)


@admin_router.get("/users", response_model=list[AdminUserProfile])
def list_admin_users(
    role: str | None = Query(default=None),
    q: str | None = Query(default=None),
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> list[AdminUserProfile]:
    supabase_client = get_supabase_service_client(settings)
    try:
        profiles = _execute_admin_profiles_query(supabase_client, role)
    except Exception as exc:
        raise _server_error("Unable to load users", exc) from exc

    if q:
        needle = q.strip().lower()
        profiles = [
            profile
            for profile in profiles
            if any(
                needle in str(profile.get(key) or "").lower()
                for key in ("email", "display_name")
            )
        ]

    profiles.sort(key=lambda profile: (_display_name(profile).lower(), profile.get("email") or ""))
    return [AdminUserProfile(**profile) for profile in profiles]


@admin_router.get("/classes", response_model=list[AdminClassListItem])
def list_admin_classes(
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> list[AdminClassListItem]:
    supabase_client = get_supabase_service_client(settings)
    try:
        classes = (
            supabase_client.table("classes")
            .select(CLASS_SELECT)
            .range(0, ADMIN_LIST_RANGE_END)
            .execute()
            .data
            or []
        )
        class_ids = [str(item["id"]) for item in classes if item.get("id")]
        student_counts = _count_by_class(
            _fetch_active_student_memberships(supabase_client, class_ids)
        )
        teacher_counts = _count_by_class(
            _fetch_active_teacher_memberships(supabase_client, class_ids)
        )
    except Exception as exc:
        raise _server_error("Unable to load classes", exc) from exc

    items = [
        AdminClassListItem(
            id=str(class_item["id"]),
            code=class_item.get("code"),
            name=class_item.get("name") or "",
            description=class_item.get("description"),
            status=class_item.get("status"),
            student_count=student_counts.get(str(class_item["id"]), 0),
            teacher_count=teacher_counts.get(str(class_item["id"]), 0),
        )
        for class_item in classes
        if class_item.get("id")
    ]
    items.sort(key=lambda item: (item.code or item.name).lower())
    return items


@admin_router.get("/classes/{class_id}", response_model=AdminClassDetail)
def get_admin_class(
    class_id: str,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> AdminClassDetail:
    supabase_client = get_supabase_service_client(settings)
    try:
        class_item = _ensure_class_exists(supabase_client, class_id)
        return _load_class_detail(supabase_client, class_item)
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to load class", exc) from exc


@admin_router.patch("/classes/{class_id}", response_model=AdminClassDetail)
def update_admin_class(
    class_id: str,
    payload: AdminClassUpdate,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> AdminClassDetail:
    supabase_client = get_supabase_service_client(settings)
    try:
        _ensure_class_exists(supabase_client, class_id)
        updates: dict[str, Any] = {}

        if payload.code is not None:
            code = _normalize_optional_text(payload.code)
            if not code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Class code cannot be empty",
                )
            duplicate = (
                supabase_client.table("classes")
                .select("id")
                .eq("code", code)
                .neq("id", class_id)
                .limit(1)
                .execute()
                .data
                or []
            )
            if duplicate:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Class code already exists",
                )
            updates["code"] = code

        if payload.name is not None:
            name = _normalize_optional_text(payload.name)
            if not name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Class name cannot be empty",
                )
            updates["name"] = name

        if payload.description is not None:
            updates["description"] = payload.description.strip() or None

        if payload.status is not None:
            class_status = _normalize_optional_text(payload.status)
            if not class_status:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Class status cannot be empty",
                )
            updates["status"] = class_status

        if not updates:
            class_item = _ensure_class_exists(supabase_client, class_id)
            return _load_class_detail(supabase_client, class_item)

        updated = (
            supabase_client.table("classes")
            .update(updates)
            .eq("id", class_id)
            .execute()
            .data
            or []
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Class not found",
            )
        return _load_class_detail(supabase_client, updated[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to update class", exc) from exc


@admin_router.delete("/users/{user_id}", response_model=AdminDeleteUserResponse)
def delete_admin_user(
    user_id: str,
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> AdminDeleteUserResponse:
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admins cannot delete their own account",
        )

    supabase_client = get_supabase_service_client(settings)
    try:
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found",
            )
        profile = profile_rows[0]
        email = profile.get("email")
        if isinstance(email, str) and email.lower() in PROTECTED_DEMO_EMAILS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Protected demo accounts cannot be deleted from this tool",
            )

        if profile.get("app_role") == "admin":
            admin_count = (
                supabase_client.table("profiles")
                .select("id")
                .eq("app_role", "admin")
                .execute()
                .data
                or []
            )
            if len(admin_count) <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the last admin account",
                )

        supabase_client.table("student_classes").delete().eq("student_id", user_id).execute()
        supabase_client.table("teacher_classes").delete().eq("teacher_id", user_id).execute()
        _delete_optional_user_links(supabase_client, user_id)

        _delete_auth_user(settings, user_id)
        supabase_client.table("profiles").delete().eq("id", user_id).execute()
    except HTTPException:
        raise
    except Exception as exc:
        raise _server_error("Unable to delete user", exc) from exc

    return AdminDeleteUserResponse(
        deleted=True,
        user_id=user_id,
        email=email if isinstance(email, str) else None,
        auth_user_deleted=True,
    )


@admin_router.get("/demo/readiness", response_model=DemoReadinessResponse)
def get_demo_readiness(
    current_user: CurrentUser = Depends(require_roles(["admin"])),
    settings: Settings = Depends(get_settings),
) -> DemoReadinessResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        profiles = supabase_client.table("profiles").select("id,app_role").execute().data or []
        classes = supabase_client.table("classes").select(CLASS_SELECT).execute().data or []
        student_memberships = _fetch_active_student_memberships(supabase_client)
        teacher_memberships = _fetch_active_teacher_memberships(supabase_client)
    except Exception as exc:
        raise _server_error("Unable to check demo readiness", exc) from exc

    role_counts = {
        "student": 0,
        "teacher": 0,
        "admin": 0,
    }
    for profile in profiles:
        role = profile.get("app_role")
        if role in role_counts:
            role_counts[role] += 1

    classes_by_code = {
        str(class_item.get("code")): class_item
        for class_item in classes
        if class_item.get("code") is not None
    }
    student_counts = _count_by_class(student_memberships)
    teacher_counts = _count_by_class(teacher_memberships)

    checks: list[DemoReadinessCheck] = [
        DemoReadinessCheck(
            key="student_count",
            passed=role_counts["student"] == 30,
            expected=30,
            actual=role_counts["student"],
            message="There must be exactly 30 student profiles.",
        ),
        DemoReadinessCheck(
            key="teacher_count",
            passed=role_counts["teacher"] == 3,
            expected=3,
            actual=role_counts["teacher"],
            message="There must be exactly 3 teacher profiles.",
        ),
        DemoReadinessCheck(
            key="admin_count",
            passed=role_counts["admin"] == 1,
            expected=1,
            actual=role_counts["admin"],
            message="There must be exactly 1 admin profile.",
        ),
        DemoReadinessCheck(
            key="class_count",
            passed=len(classes) >= 3,
            expected=">= 3",
            actual=len(classes),
            message="There must be at least 3 classes.",
        ),
    ]

    for code in ("DEMO-PHOENIX-A", "DEMO-PHOENIX-B", "DEMO-PHOENIX-C"):
        class_item = classes_by_code.get(code)
        class_id = str(class_item["id"]) if class_item and class_item.get("id") else None
        active_student_count = student_counts.get(class_id or "", 0)
        active_teacher_count = teacher_counts.get(class_id or "", 0)
        checks.append(
            DemoReadinessCheck(
                key=f"{code.lower()}_active_students",
                passed=class_id is not None and active_student_count == 10,
                expected=10,
                actual=active_student_count if class_id is not None else None,
                message=f"{code} must exist and have 10 active students.",
            )
        )
        checks.append(
            DemoReadinessCheck(
                key=f"{code.lower()}_active_teachers",
                passed=class_id is not None and active_teacher_count >= 1,
                expected=">= 1",
                actual=active_teacher_count if class_id is not None else None,
                message=f"{code} must exist and have at least 1 active teacher.",
            )
        )

    return DemoReadinessResponse(
        passed=all(check.passed for check in checks),
        checks=checks,
    )
