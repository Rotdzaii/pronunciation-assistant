import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from supabase import create_client

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local helper
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_EMAILS = {
    "student": "student01@phoenix-demo.local",
    "teacher": "teacher01@phoenix-demo.local",
    "admin": "admin@phoenix-demo.local",
}
DEMO_EMAIL_DOMAIN = "@phoenix-demo.local"
DEMO_CLASS_PREFIX = "DEMO-PHOENIX"
DEMO_CLASS_CODES = {"DEMO-PHOENIX-A", "DEMO-PHOENIX-B", "DEMO-PHOENIX-C"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR / ".tokens.local.env")


def env(name: str, default: str | None = None, *fallback_names: str) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    for fallback_name in fallback_names:
        fallback_value = os.getenv(fallback_name)
        if fallback_value:
            return fallback_value
    joined = " or ".join((name, *fallback_names))
    raise SystemExit(f"Missing required env var: {joined}")


def extract_access_token(response: Any) -> str:
    session = getattr(response, "session", None)
    if session is not None:
        token = getattr(session, "access_token", None)
        if token:
            return token
    if isinstance(response, dict):
        token = (response.get("session") or {}).get("access_token")
        if token:
            return token
    raise RuntimeError("Login succeeded but access_token was not found.")


def login_demo_tokens() -> dict[str, str]:
    supabase_url = env("SUPABASE_URL")
    anon_key = env("SUPABASE_ANON_KEY", os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_PUBLIC_ANON_KEY"))
    password = env("DEMO_DEFAULT_PASSWORD", None, "DEMO_PASSWORD")
    supabase = create_client(supabase_url, anon_key)

    email_by_role = {
        "student": os.getenv("DEMO_STUDENT_EMAIL", DEFAULT_EMAILS["student"]),
        "teacher": os.getenv("DEMO_TEACHER_EMAIL", DEFAULT_EMAILS["teacher"]),
        "admin": os.getenv("DEMO_ADMIN_EMAIL", DEFAULT_EMAILS["admin"]),
    }

    tokens: dict[str, str] = {}
    for role, email in email_by_role.items():
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        tokens[role] = extract_access_token(response)
        print(f"[PASS] {role} login: {email}")
    return tokens


class DemoSmokeTester:
    def __init__(self, base_url: str, tokens: dict[str, str]) -> None:
        self.base_url = base_url.rstrip("/")
        self.tokens = tokens
        self.client = httpx.Client(timeout=20.0)
        self.results: list[CheckResult] = []

    def close(self) -> None:
        self.client.close()

    def record(self, name: str, passed: bool, detail: str) -> None:
        label = "PASS" if passed else "FAIL"
        print(f"[{label}] {name}: {detail}")
        self.results.append(CheckResult(name, passed, detail))

    def get(self, role: str, path: str) -> httpx.Response:
        return self.client.get(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.tokens[role]}"},
        )

    def expect_status(self, name: str, role: str, path: str, expected: set[int]) -> httpx.Response | None:
        try:
            response = self.get(role, path)
        except httpx.HTTPError as exc:
            self.record(name, False, f"request failed: {exc}")
            return None
        passed = response.status_code in expected
        detail = f"{role} GET {path} -> {response.status_code}, expected={sorted(expected)}"
        if not passed:
            detail += f", body={response.text[:500]}"
        self.record(name, passed, detail)
        return response if passed else None

    def expect_list(self, name: str, role: str, path: str) -> list[dict[str, Any]]:
        response = self.expect_status(name, role, path, {200})
        if response is None:
            return []
        try:
            payload = response.json()
        except ValueError:
            self.record(f"{name} JSON", False, "invalid JSON")
            return []
        passed = isinstance(payload, list)
        self.record(f"{name} JSON list", passed, f"items={len(payload) if isinstance(payload, list) else 'n/a'}")
        return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []

    def demo_classes(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            item for item in items
            if str(item.get("code") or "").startswith(DEMO_CLASS_PREFIX)
        ]

    def demo_users(self, users: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            user for user in users
            if str(user.get("email") or "").lower().endswith(DEMO_EMAIL_DOMAIN)
        ]

    def demo_student_rows(self, students: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            student for student in students
            if str(student.get("email") or "").lower().endswith(DEMO_EMAIL_DOMAIN)
        ]

    def demo_teacher_rows(self, teachers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            teacher for teacher in teachers
            if str(teacher.get("email") or "").lower().endswith(DEMO_EMAIL_DOMAIN)
        ]

    def expect_object(self, name: str, role: str, path: str) -> dict[str, Any] | None:
        response = self.expect_status(name, role, path, {200})
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            self.record(f"{name} JSON", False, "invalid JSON")
            return None
        passed = isinstance(payload, dict)
        self.record(f"{name} JSON object", passed, "object" if passed else "not object")
        return payload if isinstance(payload, dict) else None

    def run(self) -> int:
        student_classes = self.expect_list("student sees own classes", "student", "/student/classes")
        student_demo_classes = self.demo_classes(student_classes)
        self.record(
            "student01 sees one demo class",
            len(student_demo_classes) == 1,
            f"demo_count={len(student_demo_classes)}, codes={[item.get('code') for item in student_demo_classes]}",
        )

        teacher_classes = self.expect_list("teacher sees assigned classes", "teacher", "/teacher/classes")
        teacher_demo_classes = self.demo_classes(teacher_classes)
        teacher_demo_codes = {str(item.get("code")) for item in teacher_demo_classes}
        teacher_class_ids = {str(item.get("id")) for item in teacher_demo_classes if item.get("id")}
        self.record(
            "teacher01 sees DEMO-PHOENIX-A/B",
            teacher_demo_codes == {"DEMO-PHOENIX-A", "DEMO-PHOENIX-B"},
            f"codes={sorted(teacher_demo_codes)}",
        )

        admin_classes = self.expect_list("admin sees all classes", "admin", "/admin/classes")
        admin_demo_classes = self.demo_classes(admin_classes)
        admin_demo_class_ids = {str(item.get("id")) for item in admin_demo_classes if item.get("id")}
        admin_demo_codes = {str(item.get("code")) for item in admin_demo_classes}
        self.record(
            "admin sees exact demo classes",
            admin_demo_codes == DEMO_CLASS_CODES,
            f"codes={sorted(admin_demo_codes)}",
        )
        for class_item in admin_demo_classes:
            code = str(class_item.get("code"))
            detail = self.expect_object("admin opens demo class detail", "admin", f"/admin/classes/{class_item['id']}")
            if not detail:
                continue
            demo_students = self.demo_student_rows([item for item in detail.get("students", []) if isinstance(item, dict)])
            demo_teachers = self.demo_teacher_rows([item for item in detail.get("teachers", []) if isinstance(item, dict)])
            self.record(f"{code} has 10 demo students", len(demo_students) == 10, f"demo_students={len(demo_students)}")
            expected_teachers = 2 if code == "DEMO-PHOENIX-B" else 1
            self.record(f"{code} has expected demo teachers", len(demo_teachers) == expected_teachers, f"demo_teachers={len(demo_teachers)}")

        if student_demo_classes:
            own_class_id = str(student_demo_classes[0]["id"])
            self.expect_status("student can open own class", "student", f"/student/classes/{own_class_id}", {200})
            other_class_id = next((cid for cid in admin_demo_class_ids if cid != own_class_id), None)
            if other_class_id:
                self.expect_status("student cannot open another class", "student", f"/student/classes/{other_class_id}", {403, 404})

        if teacher_demo_classes:
            teacher_class_id = str(teacher_demo_classes[0]["id"])
            self.expect_status("teacher can open assigned class", "teacher", f"/teacher/classes/{teacher_class_id}", {200})
            students = self.expect_list("teacher sees students in class", "teacher", f"/teacher/classes/{teacher_class_id}/students")
            demo_students = self.demo_student_rows(students)
            self.record("teacher class has 10 demo students", len(demo_students) == 10, f"demo_count={len(demo_students)}, total_count={len(students)}")
            self.expect_status("teacher can open class scores", "teacher", f"/teacher/classes/{teacher_class_id}/scores", {200})
            other_teacher_class_id = next((cid for cid in admin_demo_class_ids if cid not in teacher_class_ids), None)
            if other_teacher_class_id:
                self.expect_status(
                    "teacher cannot open unassigned class",
                    "teacher",
                    f"/teacher/classes/{other_teacher_class_id}",
                    {403, 404},
                )

        users = self.expect_list("admin sees users", "admin", "/admin/users")
        demo_users = self.demo_users(users)
        demo_role_counts: dict[str, int] = {"student": 0, "teacher": 0, "admin": 0}
        for user in demo_users:
            role = str(user.get("app_role") or "")
            if role in demo_role_counts:
                demo_role_counts[role] += 1
        self.record(
            "admin sees exact demo users",
            len(demo_users) == 34,
            f"demo_count={len(demo_users)}, role_counts={demo_role_counts}",
        )
        self.record(
            "admin sees expected demo role counts",
            demo_role_counts == {"student": 30, "teacher": 3, "admin": 1},
            f"role_counts={demo_role_counts}",
        )
        self.expect_status("admin demo readiness endpoint", "admin", "/admin/demo/readiness", {200})

        total = len(self.results)
        failed = [result for result in self.results if not result.passed]
        print()
        print(f"Summary: {total - len(failed)}/{total} checks passed")
        if failed:
            for result in failed:
                print(f"- {result.name}: {result.detail}")
            return 1
        return 0


def main() -> int:
    load_env()
    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    try:
        tokens = login_demo_tokens()
    except Exception as exc:
        print(f"[FAIL] Demo login failed: {exc}", file=sys.stderr)
        return 1

    tester = DemoSmokeTester(base_url, tokens)
    try:
        return tester.run()
    finally:
        tester.close()


if __name__ == "__main__":
    raise SystemExit(main())
