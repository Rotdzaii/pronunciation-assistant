import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local helper
    load_dotenv = None

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_NAMESPACE = UUID("7f31f667-79e7-4a3d-93f4-1a4a69d7a501")

CLASS_DEFINITIONS = [
    {
        "code": "DEMO-PHOENIX-A",
        "name": "Phoenix A - Phát âm cơ bản",
        "description": "Lớp luyện nền tảng phát âm cho học sinh mới bắt đầu.",
    },
    {
        "code": "DEMO-PHOENIX-B",
        "name": "Phoenix B - Âm cuối và trọng âm",
        "description": "Lớp tập trung sửa âm cuối, trọng âm từ và câu ngắn.",
    },
    {
        "code": "DEMO-PHOENIX-C",
        "name": "Phoenix C - Luyện nói nâng cao",
        "description": "Lớp luyện nói nâng cao với phản hồi AI Phoenix.",
    },
]

STUDENT_NAMES = [
    "Nguyễn Minh Anh",
    "Trần Hoàng Nam",
    "Lê Gia Hân",
    "Phạm Tuấn Kiệt",
    "Võ Thanh Trúc",
    "Đỗ Minh Quân",
    "Bùi Ngọc Diệp",
    "Hoàng Bảo Trâm",
    "Mai Phương Uyên",
    "Huỳnh Minh Châu",
    "Nguyễn Thảo Nhi",
    "Trương Mỹ Linh",
    "Phan Đức Huy",
    "Lâm Bảo Ngọc",
    "Võ Đức Anh",
    "Đặng Minh Triết",
    "Trần Nhật Minh",
    "Lê Minh Tâm",
    "Phạm Gia Huy",
    "Đỗ Ngọc Ánh",
    "Bùi Tuấn Kiệt",
    "Nguyễn Hoài Nam",
    "Huỳnh Khánh Vy",
    "Nguyễn Tấn Đạt",
    "Trần Gia Hân",
    "Lê Quốc Bảo",
    "Phạm Anh Khoa",
    "Nguyễn Khánh Linh",
    "Đặng Bảo An",
    "Vũ Minh Khang",
]

TEACHERS = [
    {"email": "teacher01@phoenix-demo.local", "display_name": "Nguyễn Hải Đăng"},
    {"email": "teacher02@phoenix-demo.local", "display_name": "Trần Thu Hà"},
    {"email": "teacher03@phoenix-demo.local", "display_name": "Lê Hoàng Phúc"},
]

ADMIN = {"email": "admin@phoenix-demo.local", "display_name": "Quản trị Phoenix"}

PRACTICE_TARGETS = [
    ("architecture", [{"phoneme": "k", "type": "Âm cuối", "word": "architecture"}]),
    ("student", [{"phoneme": "t", "type": "Âm cuối", "word": "student"}]),
    ("street", [{"phoneme": "str", "type": "Cụm phụ âm", "word": "street"}]),
    ("teacher", [{"phoneme": "iː", "type": "Độ dài nguyên âm", "word": "teacher"}]),
]


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")


def required_env(name: str, *fallback_names: str) -> str:
    for env_name in (name, *fallback_names):
        value = os.getenv(env_name)
        if value:
            return value
    joined = " or ".join((name, *fallback_names))
    raise SystemExit(f"Missing required env var: {joined}")


def stable_uuid(kind: str, value: str) -> str:
    return str(uuid5(DEMO_NAMESPACE, f"{kind}:{value}"))


def auth_users_from_response(response: Any) -> list[Any]:
    if isinstance(response, list):
        return response
    users = getattr(response, "users", None)
    if users is not None:
        return list(users)
    data = getattr(response, "data", None)
    if isinstance(data, list):
        return data
    if isinstance(response, dict):
        users_value = response.get("users") or response.get("data") or []
        return users_value if isinstance(users_value, list) else []
    return []


def user_email(user: Any) -> str | None:
    if isinstance(user, dict):
        return user.get("email")
    return getattr(user, "email", None)


def user_id(user: Any) -> str | None:
    if isinstance(user, dict):
        return user.get("id")
    return getattr(user, "id", None)


def find_auth_user_by_email(supabase: Any, email: str) -> Any | None:
    page = 1
    while True:
        try:
            response = supabase.auth.admin.list_users(page=page, per_page=1000)
        except TypeError:
            response = supabase.auth.admin.list_users()
        users = auth_users_from_response(response)
        for user in users:
            if (user_email(user) or "").lower() == email.lower():
                return user
        if len(users) < 1000 or page > 10:
            return None
        page += 1


def ensure_auth_user(supabase: Any, email: str, password: str | None, display_name: str, role: str) -> str:
    existing = find_auth_user_by_email(supabase, email)
    if existing is not None:
        existing_id = user_id(existing)
        if not existing_id:
            raise RuntimeError(f"Auth user exists but id is missing: {email}")
        update_payload: dict[str, Any] = {
            "user_metadata": {"display_name": display_name, "app_role": role},
            "email_confirm": True,
        }
        if password:
            update_payload["password"] = password
        try:
            supabase.auth.admin.update_user_by_id(existing_id, update_payload)
        except Exception as exc:
            print(f"[WARN] Could not update auth metadata for {email}: {exc}")
        print(f"[OK] Auth user exists: {email}")
        return str(existing_id)

    if not password:
        raise RuntimeError(
            f"Auth user is missing and DEMO_DEFAULT_PASSWORD/DEMO_PASSWORD was not provided: {email}"
        )

    response = supabase.auth.admin.create_user(
        {
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"display_name": display_name, "app_role": role},
        }
    )
    created = getattr(response, "user", None) or (response.get("user") if isinstance(response, dict) else None)
    created_id = user_id(created)
    if not created_id:
        created = find_auth_user_by_email(supabase, email)
        created_id = user_id(created)
    if not created_id:
        raise RuntimeError(f"Created auth user but could not resolve id: {email}")
    print(f"[OK] Created auth user: {email}")
    return str(created_id)


def upsert_profile(supabase: Any, user_id_value: str, email: str, display_name: str, role: str) -> None:
    payload = {
        "id": user_id_value,
        "email": email,
        "display_name": display_name,
        "app_role": role,
    }
    supabase.table("profiles").upsert(payload, on_conflict="id").execute()
    print(f"[OK] Profile upserted: {email} ({role})")


def ensure_class(supabase: Any, class_item: dict[str, str], owner_teacher_id: str | None = None) -> str:
    existing = (
        supabase.table("classes")
        .select("id")
        .eq("code", class_item["code"])
        .limit(1)
        .execute()
        .data
        or []
    )
    payload = {
        "code": class_item["code"],
        "name": class_item["name"],
        "description": class_item["description"],
        "status": "active",
    }
    if owner_teacher_id:
        payload["teacher_id"] = owner_teacher_id
    if existing:
        class_id = str(existing[0]["id"])
        supabase.table("classes").update(payload).eq("id", class_id).execute()
    else:
        class_id = stable_uuid("class", class_item["code"])
        supabase.table("classes").insert({"id": class_id, **payload}).execute()
    print(f"[OK] Class ready: {class_item['code']}")
    return class_id


def ensure_membership(
    supabase: Any,
    table: str,
    class_id: str,
    user_column: str,
    user_id_value: str,
    extra: dict[str, Any] | None = None,
) -> None:
    existing = (
        supabase.table(table)
        .select("class_id")
        .eq("class_id", class_id)
        .eq(user_column, user_id_value)
        .limit(1)
        .execute()
        .data
        or []
    )
    payload = {
        "class_id": class_id,
        user_column: user_id_value,
        "status": "active",
        **(extra or {}),
    }
    if table == "student_classes":
        payload["joined_at"] = datetime.now(timezone.utc).isoformat()
    if existing:
        supabase.table(table).update(payload).eq("class_id", class_id).eq(user_column, user_id_value).execute()
    else:
        supabase.table(table).insert(payload).execute()


def seed_practice_history(supabase: Any, student_id: str, student_index: int) -> None:
    now = datetime.now(timezone.utc)
    for attempt_index, (target_word, phonemes) in enumerate(PRACTICE_TARGETS, start=1):
        practice_id = stable_uuid("practice", f"{student_id}:{attempt_index}")
        status = "completed"
        score = 58 + ((student_index * 7 + attempt_index * 9) % 39)
        if attempt_index == 4 and student_index % 6 == 0:
            status = "processing"
            score = None
        if attempt_index == 3 and student_index % 11 == 0:
            status = "failed"
            score = None
        created_at = now - timedelta(days=(student_index % 10), hours=attempt_index * 3)
        payload = {
            "id": practice_id,
            "student_id": student_id,
            "target_word": target_word,
            "audio_url": f"https://example.invalid/phoenix-demo/{student_id}/{attempt_index}.webm",
            "status": status,
            "score": score,
            "problem_phonemes": phonemes if status == "completed" else [],
            "feedback": {
                "summary": "Dữ liệu demo phục vụ kiểm thử dashboard Phoenix.",
                "text_similarity": round(0.72 + (student_index % 8) * 0.02, 2),
                "model_confidence": round(0.68 + (attempt_index % 5) * 0.04, 2),
            },
            "created_at": created_at.isoformat(),
            "updated_at": (created_at + timedelta(minutes=4)).isoformat(),
        }
        existing = (
            supabase.table("practice_history")
            .select("id")
            .eq("id", practice_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if existing:
            supabase.table("practice_history").update(payload).eq("id", practice_id).execute()
        else:
            supabase.table("practice_history").insert(payload).execute()


def main() -> None:
    load_env()
    supabase_url = required_env("SUPABASE_URL")
    service_role_key = required_env("SUPABASE_SERVICE_ROLE_KEY")
    password = os.getenv("DEMO_DEFAULT_PASSWORD") or os.getenv("DEMO_PASSWORD")
    supabase = create_client(supabase_url, service_role_key)

    students = [
        {
            "email": f"student{index:02d}@phoenix-demo.local",
            "display_name": name,
        }
        for index, name in enumerate(STUDENT_NAMES, start=1)
    ]

    student_ids: dict[str, str] = {}
    teacher_ids: dict[str, str] = {}

    for student in students:
        sid = ensure_auth_user(supabase, student["email"], password, student["display_name"], "student")
        upsert_profile(supabase, sid, student["email"], student["display_name"], "student")
        student_ids[student["email"]] = sid

    for teacher in TEACHERS:
        tid = ensure_auth_user(supabase, teacher["email"], password, teacher["display_name"], "teacher")
        upsert_profile(supabase, tid, teacher["email"], teacher["display_name"], "teacher")
        teacher_ids[teacher["email"]] = tid

    admin_id = ensure_auth_user(supabase, ADMIN["email"], password, ADMIN["display_name"], "admin")
    upsert_profile(supabase, admin_id, ADMIN["email"], ADMIN["display_name"], "admin")

    owner_teacher_by_code = {
        "DEMO-PHOENIX-A": teacher_ids["teacher01@phoenix-demo.local"],
        "DEMO-PHOENIX-B": teacher_ids["teacher01@phoenix-demo.local"],
        "DEMO-PHOENIX-C": teacher_ids["teacher03@phoenix-demo.local"],
    }
    class_ids = [
        ensure_class(supabase, class_item, owner_teacher_by_code.get(class_item["code"]))
        for class_item in CLASS_DEFINITIONS
    ]

    for class_index, class_id in enumerate(class_ids):
        class_students = students[class_index * 10 : (class_index + 1) * 10]
        for student in class_students:
            ensure_membership(supabase, "student_classes", class_id, "student_id", student_ids[student["email"]])

    teacher_assignments = {
        "teacher01@phoenix-demo.local": [(class_ids[0], "owner"), (class_ids[1], "owner")],
        "teacher02@phoenix-demo.local": [(class_ids[1], "substitute")],
        "teacher03@phoenix-demo.local": [(class_ids[2], "owner")],
    }
    for teacher_email, assignments in teacher_assignments.items():
        for class_id, teacher_role in assignments:
            ensure_membership(
                supabase,
                "teacher_classes",
                class_id,
                "teacher_id",
                teacher_ids[teacher_email],
                {"teacher_role": teacher_role},
            )

    for index, student in enumerate(students, start=1):
        seed_practice_history(supabase, student_ids[student["email"]], index)

    print()
    print("[DONE] Demo data is ready.")
    print(f"- students={len(students)}")
    print(f"- teachers={len(TEACHERS)}")
    print("- admin=1")
    print(f"- classes={len(CLASS_DEFINITIONS)}")
    print("- password is read from DEMO_DEFAULT_PASSWORD or DEMO_PASSWORD when creating/updating auth passwords.")
    print("- if password is missing, existing auth users are reused and passwords are not changed.")


if __name__ == "__main__":
    main()
