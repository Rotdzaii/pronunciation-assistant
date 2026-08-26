from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = AI_WORKER_ROOT / "scripts" / "demo_mfa_pgmq_once.py"


def load_script_module():
    module_name = "demo_mfa_pgmq_once_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load demo_mfa_pgmq_once.py for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DemoMfaPgmqOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script_module()

    def test_visibility_timeout_defaults_to_worker_environment_contract(self) -> None:
        with patch.dict(os.environ, {"QUEUE_VISIBILITY_TIMEOUT_SECONDS": "75"}, clear=False), patch.object(
            sys, "argv", ["demo_mfa_pgmq_once.py"]
        ):
            args = self.module.parse_args()
        self.assertEqual(args.visibility_timeout, 75)

    def test_visibility_timeout_rejects_non_positive_value(self) -> None:
        with patch.object(sys, "argv", ["demo_mfa_pgmq_once.py", "--visibility-timeout", "0"]):
            with self.assertRaises(SystemExit) as context:
                self.module.parse_args()
        self.assertEqual(context.exception.code, 2)

    def test_queue_read_passes_visibility_timeout_to_worker(self) -> None:
        expected_row = {"msg_id": 12, "message": {"job_id": "job-1"}}
        fake_worker = SimpleNamespace(_read_one_job=Mock(return_value=expected_row))
        result = self.module.read_one_queue_job(fake_worker, object(), "practice_jobs", 90)
        self.assertEqual(result, expected_row)
        fake_worker._read_one_job.assert_called_once_with(unittest.mock.ANY, "practice_jobs", 90)

    def test_mfa_debug_defaults_to_false(self) -> None:
        with patch.object(sys, "argv", ["demo_mfa_pgmq_once.py"]):
            args = self.module.parse_args()
        self.assertFalse(args.mfa_debug)
        self.assertIsNone(args.mfa_debug_dir)
        self.assertFalse(args.post)
        self.assertFalse(args.archive)

    def test_mfa_debug_rejects_post_and_archive(self) -> None:
        with patch.object(sys, "argv", ["demo_mfa_pgmq_once.py", "--mfa-debug", "--post"]):
            with self.assertRaises(SystemExit) as context:
                self.module.parse_args()
        self.assertEqual(context.exception.code, 2)

    def test_storage_object_path_is_not_reported_as_local_path(self) -> None:
        bucket = Mock()
        bucket.download.return_value = b"queue-audio"
        client = SimpleNamespace(storage=Mock())
        client.storage.from_.return_value = bucket

        prepared, download_status, resolved = self.module.prepare_scoring_job(
            {"audio_url": "student-1/uuid-recording.webm"},
            storage_client=client,
            practice_audio_bucket="practice-audios",
        )
        try:
            self.assertEqual(download_status, "storage_object_path")
            self.assertEqual(prepared["audio_reference_type"], "storage_object_path")
            self.assertNotEqual(download_status, "local_path")
            self.assertTrue(Path(prepared["audio_path"]).is_file())
            bucket.download.assert_called_once_with("student-1/uuid-recording.webm")
        finally:
            resolved.cleanup()


if __name__ == "__main__":
    unittest.main()
