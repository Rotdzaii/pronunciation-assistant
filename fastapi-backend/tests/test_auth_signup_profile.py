from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.auth import get_profile_app_role


class _ProfilesQuery:
    def __init__(self, client: "_ProfilesClient", operation: str) -> None:
        self.client = client
        self.operation = operation
        self.payload: dict[str, object] | None = None

    def eq(self, *_: object) -> "_ProfilesQuery":
        return self

    def limit(self, *_: object) -> "_ProfilesQuery":
        return self

    def execute(self) -> SimpleNamespace:
        if self.operation == "select":
            return SimpleNamespace(data=[] if self.client.role is None else [{"app_role": self.client.role}])
        assert self.payload is not None
        self.client.insert_payloads.append(self.payload)
        if self.client.role is None:
            self.client.role = str(self.payload["app_role"])
        return SimpleNamespace(data=[])


class _ProfilesClient:
    def __init__(self, role: str | None) -> None:
        self.role = role
        self.insert_payloads: list[dict[str, object]] = []

    def table(self, name: str) -> "_ProfilesClient":
        assert name == "profiles"
        return self

    def select(self, *_: object) -> _ProfilesQuery:
        return _ProfilesQuery(self, "select")

    def insert(self, payload: dict[str, object]) -> _ProfilesQuery:
        query = _ProfilesQuery(self, "insert")
        query.payload = payload
        return query


class SignupProfileTests(unittest.TestCase):
    def _role_for(self, profile_role: str | None, metadata: dict[str, object] | None = None) -> tuple[str, _ProfilesClient]:
        client = _ProfilesClient(profile_role)
        with patch("app.core.auth.sleep"):
            role = get_profile_app_role(
                client,
                "user-1",
                email="new@example.com",
                user_metadata=metadata,
                client_factory=lambda: client,
            )
        return role, client

    def test_existing_student_profile_is_read(self) -> None:
        role, client = self._role_for("student")
        self.assertEqual(role, "student")
        self.assertEqual(client.insert_payloads, [])

    def test_existing_teacher_profile_is_read(self) -> None:
        role, client = self._role_for("teacher")
        self.assertEqual(role, "teacher")
        self.assertEqual(client.insert_payloads, [])

    def test_existing_admin_profile_is_never_overwritten(self) -> None:
        role, client = self._role_for("admin", {"app_role": "teacher"})
        self.assertEqual(role, "admin")
        self.assertEqual(client.insert_payloads, [])

    def test_missing_profile_is_repaired_with_allowed_teacher_role(self) -> None:
        role, client = self._role_for(None, {"app_role": "teacher"})
        self.assertEqual(role, "teacher")
        self.assertEqual(client.insert_payloads[0]["app_role"], "teacher")

    def test_admin_metadata_cannot_create_an_admin_profile(self) -> None:
        role, client = self._role_for(None, {"app_role": "admin"})
        self.assertEqual(role, "student")
        self.assertEqual(client.insert_payloads[0]["app_role"], "student")

    def test_unknown_metadata_falls_back_to_student(self) -> None:
        role, client = self._role_for(None, {"app_role": "moderator"})
        self.assertEqual(role, "student")
        self.assertEqual(client.insert_payloads[0]["app_role"], "student")

    def test_reliability_migration_is_insert_only_and_preserves_admin_constraint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        migration = (root / "db" / "migrations" / "014_signup_profile_reliability.sql").read_text(encoding="utf-8")
        self.assertIn("begin;", migration)
        self.assertIn("commit;", migration)
        self.assertIn("on conflict (id) do nothing", migration)
        self.assertIn("('student', 'teacher', 'admin')", migration)
        self.assertIn("requested_app_role in ('student', 'teacher')", migration)


if __name__ == "__main__":
    unittest.main()
