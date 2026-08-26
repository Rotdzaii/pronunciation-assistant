from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.api import assignments, practice
from app.services.assignment_grading import is_past_deadline, parse_timestamp


ROOT = Path(__file__).resolve().parents[1]


class AssignmentContractTests(unittest.TestCase):
    def test_deadline_uses_utc_and_rejects_expired_work(self) -> None:
        now = datetime.now(timezone.utc)
        self.assertTrue(is_past_deadline({"deadline": (now - timedelta(seconds=1)).isoformat()}, now))
        self.assertFalse(is_past_deadline({"deadline": (now + timedelta(seconds=1)).isoformat()}, now))

    def test_timestamp_parser_handles_zulu_times(self) -> None:
        self.assertEqual(parse_timestamp("2026-01-01T00:00:00Z"), datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_student_list_precedes_uuid_assignment_route(self) -> None:
        paths = [route.path for route in assignments.router.routes]
        self.assertLess(paths.index("/assignments/student"), paths.index("/assignments/{assignment_id}"))

    def test_legacy_client_progress_endpoint_is_not_a_write_path(self) -> None:
        source = (ROOT / "app" / "api" / "assignments.py").read_text(encoding="utf-8")
        start = source.index("def update_student_progress")
        end = source.index("@router.get(\"/{assignment_id}/words\"")
        self.assertIn("HTTP_403_FORBIDDEN", source[start:end])

    def test_migration_has_recipient_grade_and_practice_links(self) -> None:
        migration = (ROOT / "db" / "migrations" / "013_assignment_recipients_grades.sql").read_text(encoding="utf-8")
        for fragment in ("assignment_recipients", "assignment_grades", "assessment_submission_id", "assignment_id uuid", "UNIQUE (assignment_id, student_id)"):
            self.assertIn(fragment, migration)

    def test_migration_backfills_recipients_and_grades(self) -> None:
        migration = (ROOT / "db" / "migrations" / "013_assignment_recipients_grades.sql").read_text(encoding="utf-8")
        self.assertIn("FROM public.assignment_progress", migration)
        self.assertIn("FROM public.assessment_submissions", migration)
        self.assertIn("INSERT INTO public.assignment_grades", migration)

    def test_practice_contract_requires_assignment_and_item_together(self) -> None:
        source = (ROOT / "app" / "api" / "practice.py").read_text(encoding="utf-8")
        self.assertIn("assignment_id and item_id must be provided together", source)
        self.assertIn("Word is not part of this assignment", source)

    def test_webhook_recalculates_derived_assignment_state(self) -> None:
        source = (ROOT / "app" / "api" / "practice.py").read_text(encoding="utf-8")
        self.assertIn("recalculate_assignment_grade", source)
        self.assertIn("result already processed", source)

    def test_assessment_start_has_unique_submission_guard(self) -> None:
        source = (ROOT / "app" / "api" / "assignments.py").read_text(encoding="utf-8")
        self.assertIn("Unique(assignment_id, student_id) is the concurrent-start guard", source)
        self.assertIn("Assessment is not available to this student", source)

    def test_grade_override_requires_a_reason_in_model_and_migration(self) -> None:
        migration = (ROOT / "db" / "migrations" / "013_assignment_recipients_grades.sql").read_text(encoding="utf-8")
        self.assertIn("teacher_override_score IS NULL OR override_reason IS NOT NULL", migration)
        self.assertIn("override_reason is required", (ROOT / "app" / "api" / "assignments.py").read_text(encoding="utf-8"))

    def test_free_practice_stays_optional(self) -> None:
        payload = practice.PracticeJobCreate(target_word="book", audio_url="https://example.invalid/audio")
        self.assertIsNone(payload.assignment_id)
        self.assertIsNone(payload.item_id)


if __name__ == "__main__":
    unittest.main()
