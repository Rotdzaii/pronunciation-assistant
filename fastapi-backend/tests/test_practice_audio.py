from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from app.api import practice


class _StorageBucket:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.list_calls: list[tuple[object, object]] = []
        self.signed_calls: list[tuple[str, int]] = []

    def list(self, path=None, options=None):  # type: ignore[no-untyped-def]
        self.list_calls.append((path, options))
        return [{"name": "recording.webm"}] if self.exists else []

    def create_signed_url(self, path: str, expires_in: int) -> dict[str, str]:
        self.signed_calls.append((path, expires_in))
        return {"signedURL": "https://storage.example.invalid/signed/playback.webm"}


class _Storage:
    def __init__(self, bucket: _StorageBucket) -> None:
        self.bucket = bucket
        self.bucket_names: list[str] = []

    def from_(self, bucket_name: str) -> _StorageBucket:
        self.bucket_names.append(bucket_name)
        return self.bucket


class _Query:
    def __init__(self, client: "_Client", table_name: str) -> None:
        self.client = client
        self.table_name = table_name

    def select(self, *_: object) -> "_Query":
        return self

    def eq(self, *_: object) -> "_Query":
        return self

    def in_(self, *_: object) -> "_Query":
        return self

    def limit(self, *_: object) -> "_Query":
        return self

    def execute(self) -> SimpleNamespace:
        if self.table_name == "practice_history":
            return SimpleNamespace(data=[self.client.practice_row])
        if self.table_name == "teacher_classes":
            return SimpleNamespace(data=self.client.teacher_classes)
        if self.table_name == "student_classes":
            return SimpleNamespace(data=self.client.student_classes)
        return SimpleNamespace(data=[])


class _Client:
    def __init__(self, *, audio_url: str, exists: bool = True) -> None:
        self.practice_row = {
            "id": str(uuid4()),
            "student_id": "student-1",
            "audio_url": audio_url,
        }
        self.storage_bucket = _StorageBucket(exists=exists)
        self.storage = _Storage(self.storage_bucket)
        self.teacher_classes: list[dict[str, str]] = []
        self.student_classes: list[dict[str, str]] = []

    def table(self, table_name: str) -> _Query:
        return _Query(self, table_name)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(practice_audio_bucket="practice-audios")


class PracticeAudioEndpointTests(unittest.TestCase):
    def _request(self, client: _Client, user: SimpleNamespace):
        with patch.object(practice, "get_supabase_service_client", return_value=client):
            return practice.get_practice_history_audio(
                uuid4(),
                current_user=user,
                settings=_settings(),
            )

    def test_owner_receives_signed_url_for_storage_object_path(self) -> None:
        client = _Client(audio_url="student-1/recording.webm")
        response = self._request(client, SimpleNamespace(id="student-1", app_role="student"))

        self.assertEqual(response.audio_url, "https://storage.example.invalid/signed/playback.webm")
        self.assertEqual(response.expires_in, 300)
        self.assertEqual(client.storage.bucket_names, ["practice-audios"])
        self.assertEqual(client.storage_bucket.signed_calls, [("student-1/recording.webm", 300)])

    def test_other_student_is_denied_before_storage_is_accessed(self) -> None:
        client = _Client(audio_url="student-1/recording.webm")

        with self.assertRaises(HTTPException) as raised:
            self._request(client, SimpleNamespace(id="student-2", app_role="student"))

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(client.storage_bucket.signed_calls, [])

    def test_active_teacher_membership_can_access_student_recording(self) -> None:
        client = _Client(audio_url="student-1/recording.webm")
        client.teacher_classes = [{"class_id": "class-1"}]
        client.student_classes = [{"student_id": "student-1"}]

        response = self._request(client, SimpleNamespace(id="teacher-1", app_role="teacher"))

        self.assertTrue(response.audio_url.startswith("https://"))

    def test_legacy_http_audio_url_is_returned_without_storage_signing(self) -> None:
        legacy_url = "https://legacy.example.invalid/audio.webm"
        client = _Client(audio_url=legacy_url)

        response = self._request(client, SimpleNamespace(id="student-1", app_role="student"))

        self.assertEqual(response.audio_url, legacy_url)
        self.assertEqual(client.storage.bucket_names, [])

    def test_missing_object_returns_safe_not_found(self) -> None:
        client = _Client(audio_url="student-1/missing.webm", exists=False)

        with self.assertRaises(HTTPException) as raised:
            self._request(client, SimpleNamespace(id="student-1", app_role="student"))

        self.assertEqual(raised.exception.status_code, 404)
        self.assertNotIn("missing.webm", str(raised.exception.detail))
        self.assertNotIn("service", str(raised.exception.detail).lower())


if __name__ == "__main__":
    unittest.main()
