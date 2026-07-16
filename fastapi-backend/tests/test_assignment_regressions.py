from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from app.api import assignments, practice
from app.services.assignment_grading import recalculate_assignment_grade


class _MemoryQuery:
    def __init__(self, client: "_MemorySupabase", table: str, operation: str, payload=None) -> None:
        self.client = client
        self.table = table
        self.operation = operation
        self.payload = payload
        self.filters: list[tuple[str, object]] = []
        self._limit: int | None = None

    def eq(self, column: str, value: object) -> "_MemoryQuery":
        self.filters.append((column, value))
        return self

    def limit(self, value: int) -> "_MemoryQuery":
        self._limit = value
        return self

    def order(self, *_: object, **__: object) -> "_MemoryQuery":
        return self

    def range(self, *_: object) -> "_MemoryQuery":
        return self

    def _matching(self) -> list[dict]:
        rows = self.client.rows.setdefault(self.table, [])
        return [
            row for row in rows
            if all(str(row.get(column)) == str(value) for column, value in self.filters)
        ]

    def execute(self) -> SimpleNamespace:
        self.client.query_log.append((self.table, self.operation, tuple(self.filters)))
        if self.operation == "select":
            rows = self._matching()
            if self._limit is not None:
                rows = rows[:self._limit]
            return SimpleNamespace(data=deepcopy(rows), count=len(rows))

        if self.operation == "insert":
            if self.table == "practice_history" and self.client.practice_insert_error:
                raise self.client.practice_insert_error
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            saved = [self.client._prepare_row(self.table, payload) for payload in payloads]
            self.client.rows.setdefault(self.table, []).extend(saved)
            return SimpleNamespace(data=deepcopy(saved), count=len(saved))

        if self.operation == "upsert":
            payloads = self.payload["rows"] if isinstance(self.payload["rows"], list) else [self.payload["rows"]]
            keys = [key.strip() for key in (self.payload.get("on_conflict") or "id").split(",")]
            saved: list[dict] = []
            for payload in payloads:
                existing = next(
                    (row for row in self.client.rows.setdefault(self.table, []) if all(str(row.get(key)) == str(payload.get(key)) for key in keys)),
                    None,
                )
                if existing is None:
                    existing = self.client._prepare_row(self.table, payload)
                    self.client.rows[self.table].append(existing)
                else:
                    existing.update(payload)
                    self.client._fill_defaults(self.table, existing)
                saved.append(existing)
            return SimpleNamespace(data=deepcopy(saved), count=len(saved))

        if self.operation == "update":
            saved = self._matching()
            for row in saved:
                row.update(self.payload)
                self.client._fill_defaults(self.table, row)
            return SimpleNamespace(data=deepcopy(saved), count=len(saved))

        raise AssertionError(f"Unsupported operation: {self.operation}")


class _MemorySupabase:
    """Small Supabase double that preserves table filters and derived writes."""

    def __init__(self, rows: dict[str, list[dict]] | None = None) -> None:
        self.rows = deepcopy(rows or {})
        self.query_log: list[tuple[str, str, tuple[tuple[str, object], ...]]] = []
        self.practice_insert_error: Exception | None = None
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, table: str) -> "_MemoryTable":
        return _MemoryTable(self, table)

    def rpc(self, name: str, params: dict) -> SimpleNamespace:
        self.rpc_calls.append((name, deepcopy(params)))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[]))

    def _prepare_row(self, table: str, payload: dict) -> dict:
        row = deepcopy(payload)
        row.setdefault("id", f"{table}-{len(self.rows.setdefault(table, [])) + 1}")
        self._fill_defaults(table, row)
        return row

    @staticmethod
    def _fill_defaults(table: str, row: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if table in {"assignments", "assignment_progress", "assignment_grades", "assessment_submissions", "practice_history"}:
            row.setdefault("created_at", now)
            row.setdefault("updated_at", now)
        if table == "assignment_progress":
            row.setdefault("status", "not_started")
            row.setdefault("completed_items", 0)
            row.setdefault("total_items", 0)
            row.setdefault("completed_at", None)
        if table == "assignment_grades":
            row.setdefault("grading_status", "pending")
        if table == "assignments":
            row.setdefault("description", None)
            row.setdefault("class_id", None)
            row.setdefault("student_id", None)
            row.setdefault("due_date", None)
            row.setdefault("deadline", None)
            row.setdefault("is_assessment", False)
            row.setdefault("timer_per_word_seconds", 60)


class _MemoryTable:
    def __init__(self, client: _MemorySupabase, table: str) -> None:
        self.client = client
        self.table = table

    def select(self, *_: object, **__: object) -> _MemoryQuery:
        return _MemoryQuery(self.client, self.table, "select")

    def insert(self, payload: dict | list[dict]) -> _MemoryQuery:
        return _MemoryQuery(self.client, self.table, "insert", payload)

    def upsert(self, payload: dict | list[dict], on_conflict: str | None = None) -> _MemoryQuery:
        return _MemoryQuery(self.client, self.table, "upsert", {"rows": payload, "on_conflict": on_conflict})

    def update(self, payload: dict) -> _MemoryQuery:
        return _MemoryQuery(self.client, self.table, "update", payload)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _assignment(*, assessment: bool = True) -> dict:
    return {
        "id": "assignment-1",
        "content_type": "vocabulary_set",
        "content_id": "set-1",
        "is_assessment": assessment,
        "deadline": (_now() + timedelta(days=1)).isoformat(),
    }


def _set_items() -> list[dict]:
    return [
        {"item_id": "item-1", "set_id": "set-1", "sort_order": 1, "vocabulary_items": {"id": "word-1", "word": "book"}},
        {"item_id": "item-2", "set_id": "set-1", "sort_order": 2, "vocabulary_items": {"id": "word-2", "word": "pen"}},
    ]


class TeacherAssignmentAuthorizationRegressionTests(TestCase):
    def _client(self) -> _MemorySupabase:
        return _MemorySupabase({
            "teacher_classes": [{"class_id": "class-owned", "teacher_id": "teacher-1", "status": "active"}],
            "student_classes": [{"class_id": "class-owned", "student_id": "student-1", "status": "active"}],
            "vocabulary_sets": [{"id": "set-1", "is_active": True}],
            "vocabulary_set_items": _set_items(),
        })

    def test_teacher_can_create_assignment_for_owned_class(self) -> None:
        client = self._client()
        payload = assignments.AssignmentCreatePayload(title="Owned", content_type="vocabulary_set", content_id="set-1", class_id="class-owned")
        with patch.object(assignments, "get_supabase_service_client", return_value=client):
            response = assignments.create_assignment(payload, SimpleNamespace(id="teacher-1", app_role="teacher"), SimpleNamespace())

        self.assertEqual(response.assigned_by, "teacher-1")
        self.assertEqual(len(client.rows["assignments"]), 1)
        self.assertEqual(client.rows["assignment_recipients"][0]["student_id"], "student-1")

    def test_teacher_cannot_create_assignment_for_unowned_class(self) -> None:
        client = self._client()
        payload = assignments.AssignmentCreatePayload(title="Forbidden", content_type="vocabulary_set", content_id="set-1", class_id="class-other")
        with patch.object(assignments, "get_supabase_service_client", return_value=client):
            with self.assertRaises(HTTPException) as raised:
                assignments.create_assignment(payload, SimpleNamespace(id="teacher-1", app_role="teacher"), SimpleNamespace())

        self.assertEqual(raised.exception.status_code, 403)
        self.assertFalse(client.rows.get("assignments"))

    def test_admin_has_no_implicit_cross_class_assignment_bypass(self) -> None:
        client = self._client()
        payload = assignments.AssignmentCreatePayload(title="No bypass", content_type="vocabulary_set", content_id="set-1", class_id="class-owned")
        with patch.object(assignments, "get_supabase_service_client", return_value=client):
            with self.assertRaises(HTTPException) as raised:
                assignments.create_assignment(payload, SimpleNamespace(id="admin-1", app_role="admin"), SimpleNamespace())

        self.assertEqual(raised.exception.status_code, 403)


class StudentAssignmentVisibilityRegressionTests(TestCase):
    def test_student_list_returns_assignment_from_matching_recipient(self) -> None:
        assignment = {
            **_assignment(assessment=False),
            "title": "Visible regular work",
            "description": None,
            "class_id": None,
            "assigned_by": "teacher-1",
            "due_date": None,
            "created_at": _now().isoformat(),
        }
        client = _MemorySupabase({
            # Matches Supabase's assignments(*) relation returned with a
            # matching assignment_recipients row.
            "assignment_recipients": [{
                "assignment_id": "assignment-1",
                "student_id": "student-1",
                "assigned_at": _now().isoformat(),
                "assignments": assignment,
            }],
            "assignment_progress": [{
                "assignment_id": "assignment-1", "student_id": "student-1",
                "status": "not_started", "completed_items": 0, "total_items": 2,
            }],
            "assignment_grades": [{
                "assignment_id": "assignment-1", "student_id": "student-1", "grading_status": "pending",
            }],
            "profiles": [{"id": "teacher-1", "display_name": "Teacher"}],
        })

        with patch.object(assignments, "get_supabase_service_client", return_value=client), patch.object(
            assignments, "sweep_assignment_state", return_value=None
        ):
            response = assignments.list_student_assignments(
                SimpleNamespace(id="student-1", app_role="student"), SimpleNamespace()
            )

        self.assertEqual([item.assignment_id for item in response], ["assignment-1"])
        self.assertEqual(response[0].title, "Visible regular work")
        self.assertIn(
            ("assignment_recipients", "select", (("student_id", "student-1"),)),
            client.query_log,
        )


class AssessmentPracticeConcurrencyRegressionTests(TestCase):
    def _client(self, insert_error: Exception) -> _MemorySupabase:
        assignment = _assignment()
        started = _now().isoformat()
        client = _MemorySupabase({
            "assignments": [assignment],
            "assignment_recipients": [{"id": "recipient-1", "assignment_id": "assignment-1", "student_id": "student-1"}],
            "assessment_submissions": [{"id": "submission-1", "assignment_id": "assignment-1", "student_id": "student-1", "started_at": started, "recordings": [], "is_locked": False, "submitted_at": None}],
            "vocabulary_set_items": _set_items(),
        })
        client.practice_insert_error = insert_error
        return client

    def _create(self, client: _MemorySupabase) -> HTTPException:
        payload = practice.PracticeJobCreate(target_word="book", audio_url="https://audio.invalid/book", assignment_id="assignment-1", item_id="item-1")
        with patch.object(practice, "get_supabase_service_client", return_value=client), patch.object(
            practice, "resolve_stored_pronunciation", return_value={}
        ), patch.object(practice, "sweep_assignment_state", return_value=client.rows["assessment_submissions"][0]):
            with self.assertRaises(HTTPException) as raised:
                practice.create_practice_job(payload, SimpleNamespace(id="student-1", app_role="student"), SimpleNamespace())
        return raised.exception

    def test_unique_assessment_item_conflict_maps_to_409(self) -> None:
        error = self._create(self._client(RuntimeError("duplicate key value violates unique constraint practice_history_assessment_submission_item_uidx")))
        self.assertEqual(error.status_code, 409)

    def test_unrelated_practice_insert_error_remains_500(self) -> None:
        error = self._create(self._client(RuntimeError("database connection reset")))
        self.assertEqual(error.status_code, 500)

    def test_new_practice_job_enqueues_stable_storage_path_not_signed_url(self) -> None:
        client = _MemorySupabase({
            "assignments": [{**_assignment(assessment=False)}],
            "assignment_recipients": [{"id": "recipient-1", "assignment_id": "assignment-1", "student_id": "student-1"}],
            "vocabulary_set_items": _set_items(),
        })
        payload = practice.PracticeJobCreate(
            target_word="book",
            audio_url="https://storage.example.invalid/signed/recording.webm?token=expired-later",
            audio_storage_path="student-1/fresh-recording.webm",
            assignment_id="assignment-1",
            item_id="item-1",
        )
        with patch.object(practice, "get_supabase_service_client", return_value=client), patch.object(
            practice, "resolve_stored_pronunciation", return_value={}
        ):
            practice.create_practice_job(payload, SimpleNamespace(id="student-1", app_role="student"), SimpleNamespace())

        self.assertEqual(client.rpc_calls, [(
            "enqueue_practice_job",
            {
                "p_job_id": client.rows["practice_history"][0]["id"],
                "p_student_id": "student-1",
                "p_target_word": "book",
                "p_audio_url": "student-1/fresh-recording.webm",
            },
        )])

    def test_migration_declares_partial_unique_assessment_item_index(self) -> None:
        migration = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "013_assignment_recipients_grades.sql").read_text(encoding="utf-8")
        compact = " ".join(migration.lower().split())
        self.assertIn("create unique index if not exists practice_history_assessment_submission_item_uidx on public.practice_history (assessment_submission_id, item_id) where assessment_submission_id is not null", compact)


class AssessmentGradingIsolationRegressionTests(TestCase):
    def _client(self, jobs: list[dict]) -> _MemorySupabase:
        return _MemorySupabase({
            "vocabulary_set_items": _set_items(),
            "practice_history": jobs,
            "assessment_submissions": [
                {"id": "submission-old", "assignment_id": "assignment-1", "student_id": "student-1", "is_locked": True, "submitted_at": _now().isoformat()},
                {"id": "submission-current", "assignment_id": "assignment-1", "student_id": "student-1", "is_locked": True, "submitted_at": _now().isoformat()},
            ],
        })

    def test_recalculate_uses_only_current_assessment_submission_jobs_and_is_idempotent(self) -> None:
        created = _now().isoformat()
        client = self._client([
            {"id": "old-1", "assignment_id": "assignment-1", "student_id": "student-1", "assessment_submission_id": "submission-old", "item_id": "item-1", "status": "completed", "score": 100, "created_at": created},
            {"id": "old-2", "assignment_id": "assignment-1", "student_id": "student-1", "assessment_submission_id": "submission-old", "item_id": "item-2", "status": "completed", "score": 100, "created_at": created},
            {"id": "current-1", "assignment_id": "assignment-1", "student_id": "student-1", "assessment_submission_id": "submission-current", "item_id": "item-1", "status": "completed", "score": 20, "created_at": created},
            {"id": "current-2", "assignment_id": "assignment-1", "student_id": "student-1", "assessment_submission_id": "submission-current", "item_id": "item-2", "status": "completed", "score": 40, "created_at": created},
        ])
        submission = client.rows["assessment_submissions"][1]

        first = recalculate_assignment_grade(client, _assignment(), "student-1", submission=submission)
        second = recalculate_assignment_grade(client, _assignment(), "student-1", submission=submission)

        self.assertEqual(first["grade"]["auto_score"], 30.0)
        self.assertEqual(first["grade"]["final_score"], 30.0)
        self.assertEqual(second["grade"]["auto_score"], 30.0)
        self.assertEqual(len(client.rows["assignment_grades"]), 1)
        self.assertEqual(len(client.rows["assignment_progress"]), 1)
        self.assertTrue(any(
            table == "practice_history" and ("assessment_submission_id", "submission-current") in filters
            for table, operation, filters in client.query_log if operation == "select"
        ))

    def test_completed_ai_result_without_score_requires_review_not_zero(self) -> None:
        client = self._client([{
            "id": "score-missing", "assignment_id": "assignment-1", "student_id": "student-1", "assessment_submission_id": "submission-current", "item_id": "item-1", "status": "completed", "score": None, "created_at": _now().isoformat(),
        }])
        result = recalculate_assignment_grade(client, _assignment(), "student-1", submission=client.rows["assessment_submissions"][1])

        self.assertEqual(result["grade"]["grading_status"], "needs_review")
        self.assertIsNone(result["grade"]["auto_score"])
        self.assertIsNone(result["grade"]["final_score"])

    def test_finalized_assessment_with_no_recording_still_scores_missing_words_as_zero(self) -> None:
        client = self._client([])
        result = recalculate_assignment_grade(client, _assignment(), "student-1", submission=client.rows["assessment_submissions"][1])

        self.assertEqual(result["grade"]["grading_status"], "graded")
        self.assertEqual(result["grade"]["auto_score"], 0.0)
        self.assertEqual(result["grade"]["final_score"], 0.0)


class AssignmentWebhookIdempotencyRegressionTests(TestCase):
    def test_duplicate_terminal_webhook_does_not_change_progress_or_grade(self) -> None:
        job_id = str(uuid4())
        client = _MemorySupabase({
            "practice_history": [{"id": job_id, "target_word": "book", "status": "completed", "score": 81, "feedback": {}, "assignment_id": "assignment-1", "student_id": "student-1"}],
            "assignment_progress": [{"id": "progress-1", "assignment_id": "assignment-1", "student_id": "student-1", "completed_items": 1, "total_items": 2, "status": "in_progress"}],
            "assignment_grades": [{"id": "grade-1", "assignment_id": "assignment-1", "student_id": "student-1", "auto_score": 40.5, "final_score": 40.5, "grading_status": "provisional"}],
        })
        before = deepcopy(client.rows)
        payload = practice.PracticeJobResult(job_id=job_id, status="completed", score=81)

        with patch.object(practice, "get_supabase_service_client", return_value=client):
            response = practice.update_practice_job_result(payload, x_ai_webhook_secret="secret", settings=SimpleNamespace(ai_webhook_secret="secret"))

        self.assertEqual(response.message, "Practice job result already processed")
        self.assertEqual(client.rows, before)
