from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.audio.storage_resolver import AudioReferenceError, resolve_audio_reference


class AudioStorageResolverTests(unittest.TestCase):
    def test_existing_local_audio_path_is_used_for_debug(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_audio = Path(directory) / "recording.webm"
            local_audio.write_bytes(b"local-audio")

            resolved = resolve_audio_reference(
                {"audio_path": str(local_audio), "audio_url": "student/ignored.webm"},
                storage_client=None,
            )

            self.assertEqual(resolved.path, local_audio)
            self.assertEqual(resolved.reference_type, "local_path")
            self.assertFalse(resolved.cleanup_required)

    def test_http_url_downloads_to_a_cleanupable_temp_file(self) -> None:
        response = SimpleNamespace(content=b"legacy-url-audio", raise_for_status=Mock())
        with patch("app.audio.storage_resolver.requests.get", return_value=response) as get:
            resolved = resolve_audio_reference(
                {"audio_url": "https://storage.example.invalid/recording.webm?token=old"},
                storage_client=None,
            )

        self.assertEqual(resolved.reference_type, "signed_url")
        self.assertTrue(resolved.cleanup_required)
        self.assertEqual(resolved.path.read_bytes(), b"legacy-url-audio")
        get.assert_called_once_with("https://storage.example.invalid/recording.webm?token=old", timeout=30)
        response.raise_for_status.assert_called_once_with()
        resolved.cleanup()
        self.assertFalse(resolved.path.exists())

    def test_storage_object_path_downloads_from_configured_bucket(self) -> None:
        bucket = Mock()
        bucket.download.return_value = b"storage-audio"
        client = SimpleNamespace(storage=Mock())
        client.storage.from_.return_value = bucket

        resolved = resolve_audio_reference(
            {"audio_url": "student-1/uuid-recording.webm"},
            storage_client=client,
            practice_audio_bucket="custom-practice-audio",
        )

        self.assertEqual(resolved.reference_type, "storage_object_path")
        self.assertEqual(resolved.path.read_bytes(), b"storage-audio")
        client.storage.from_.assert_called_once_with("custom-practice-audio")
        bucket.download.assert_called_once_with("student-1/uuid-recording.webm")
        resolved.cleanup()
        self.assertFalse(resolved.path.exists())

    def test_storage_download_error_does_not_leak_object_path_or_bucket(self) -> None:
        bucket = Mock()
        bucket.download.side_effect = RuntimeError("missing object student-1/private.webm")
        client = SimpleNamespace(storage=Mock())
        client.storage.from_.return_value = bucket

        with self.assertRaises(AudioReferenceError) as context:
            resolve_audio_reference(
                {"audio_url": "student-1/private.webm"},
                storage_client=client,
                practice_audio_bucket="private-bucket",
            )

        self.assertEqual(str(context.exception), "Queued audio could not be downloaded from Storage.")
        self.assertNotIn("student-1", str(context.exception))
        self.assertNotIn("private-bucket", str(context.exception))

    def test_missing_audio_reference_fails_clearly(self) -> None:
        with self.assertRaisesRegex(AudioReferenceError, "Job has no audio reference"):
            resolve_audio_reference({}, storage_client=None)


if __name__ == "__main__":
    unittest.main()
