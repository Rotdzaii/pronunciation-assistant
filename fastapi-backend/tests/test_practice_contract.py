from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from app.api import practice


class _Query:
    def __init__(self, client: "_Client", table_name: str, operation: str) -> None:
        self.client = client
        self.table_name = table_name
        self.operation = operation

    def eq(self, *_: object) -> "_Query":
        return self

    def limit(self, *_: object) -> "_Query":
        return self

    def execute(self) -> SimpleNamespace:
        if self.operation == "select" and self.table_name == "practice_history":
            return SimpleNamespace(data=[{
                "id": self.client.job_id,
                "target_word": "book",
                "feedback": {"score_type": "heuristic_classifier_score"},
            }])
        if self.operation == "select" and self.table_name == "vocabulary_items":
            return SimpleNamespace(data=[{
                "phonetic": "bʊk",
                "target_phonemes": ["b", "ʊ", "k"],
            }])
        return SimpleNamespace(data=[])


class _Client:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.updated: dict | None = None

    def table(self, table_name: str) -> "_Client":
        self.table_name = table_name
        return self

    def select(self, *_: object) -> _Query:
        return _Query(self, self.table_name, "select")

    def update(self, value: dict) -> _Query:
        self.updated = value
        return _Query(self, self.table_name, "update")


class PracticeWebhookContractTests(unittest.TestCase):
    def test_completed_unavailable_score_is_persisted_with_http_200(self) -> None:
        job_id = str(uuid4())
        client = _Client(job_id)
        payload = practice.PracticeJobResult(
            job_id=job_id,
            status="completed",
            score=None,
            score_type="unavailable",
            problem_phonemes=[],
            feedback={
                "reliability": {"status": "invalid"},
                "display_pronunciation": "should-not-win",
                "expected_phones": ["raw"],
                "stderr": "C:\\temp\\private.log",
            },
        )
        with patch.object(practice, "_require_ai_webhook_secret"), patch.object(
            practice, "get_supabase_service_client", return_value=client
        ):
            response = practice.update_practice_job_result(
                payload,
                x_ai_webhook_secret="test",
                settings=SimpleNamespace(ai_webhook_secret="test"),
            )

        self.assertEqual(response.status, "completed")
        webhook_route = next(route for route in practice.router.routes if route.path == "/practice/webhook/ai-result")
        self.assertEqual(webhook_route.status_code, 200)
        self.assertIsNotNone(client.updated)
        self.assertIsNone(client.updated["score"])
        feedback = client.updated["feedback"]
        self.assertEqual(feedback["display_pronunciation"], "bʊk")
        self.assertEqual(feedback["expected_phones"], ["b", "ʊ", "k"])
        self.assertEqual(feedback["pronunciation_source"], "vocabulary_items")
        self.assertNotIn("stderr", feedback)

    def test_legacy_payload_fields_remain_optional(self) -> None:
        payload = practice.PracticeJobResult(
            job_id=uuid4(),
            status="failed",
        )
        self.assertEqual(payload.problem_phonemes, [])
        self.assertEqual(payload.feedback, {})


if __name__ == "__main__":
    unittest.main()
