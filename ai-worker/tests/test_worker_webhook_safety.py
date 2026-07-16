from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from app.contracts.webhook_payload import build_success_webhook_payload, validate_webhook_payload
from app.scorers import cnn_attention_scorer
import worker


def _prediction() -> dict:
    return {
        "index": 0,
        "phone": "B",
        "word": "book",
        "start": 0.1,
        "end": 0.3,
        "predicted_error_type": "substitution",
        "diagnosis_confidence": 0.9,
        "class_probabilities": {"substitution": 0.9},
    }


def _heuristic_scoring() -> dict:
    return {
        "utterance_segmental_score": 70.0,
        "scoring_method": "heuristic_gop",
        "scoring_status": "heuristic",
        "metadata": {"is_heuristic": True},
    }


class ClassifierOnlyWebhookTests(unittest.TestCase):
    def test_classifier_only_output_produces_valid_webhook(self) -> None:
        alignment = {
            "status": "success",
            "method": "mfa",
            "metadata": {"is_forced_alignment": True, "mfa_used": True},
            "words": [],
            "phones": [{"phone": "B", "start": 0.1, "end": 0.3}],
        }
        result = cnn_attention_scorer._aggregate_segment_predictions(  # noqa: SLF001
            [_prediction()], alignment_result=alignment
        )

        self.assertEqual(result["model_capability"], "error_type_classifier_only")
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")
        self.assertFalse(result["diagnosis"]["is_confirmed_error"])
        valid, issues = validate_webhook_payload(build_success_webhook_payload("job-1", result))
        self.assertTrue(valid, issues)

    def test_invalid_reliability_classifier_only_output_produces_valid_webhook(self) -> None:
        alignment = {
            "status": "failed",
            "method": "fallback_even_split",
            "metadata": {"fallback_alignment": True},
            "words": [],
            "phones": [{"phone": "B", "start": 0.1, "end": 0.3}],
        }
        result = cnn_attention_scorer._aggregate_segment_predictions(  # noqa: SLF001
            [_prediction()], alignment_result=alignment
        )

        self.assertEqual(result["reliability"]["status"], "invalid")
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")
        valid, issues = validate_webhook_payload(build_success_webhook_payload("job-2", result))
        self.assertTrue(valid, issues)


class WorkerWebhookSafetyNetTests(unittest.TestCase):
    def test_other_validation_failure_posts_valid_failed_payload_and_archives(self) -> None:
        job = {
            "job_id": "job-safety",
            "student_id": "student-1",
            "target_word": "book",
            "audio_url": "https://audio.invalid/book",
        }
        config = {
            "queue_name": "practice_jobs",
            "visibility_timeout": 60,
            "scorer_mode": "mock",
            "alignment_mode": "fallback",
            "model_version": "test",
            "confidence_threshold": 0.65,
            "webhook_url": "https://backend.invalid/webhook",
            "webhook_secret": "test-secret",
        }
        completed_result = {
            "status": "completed",
            "score": None,
            "score_type": "unavailable",
            "problem_phonemes": [],
            "feedback": {"summary": "No score", "tips": []},
            "diagnosis": {"confidence_note": "No score is available."},
            "metadata": {},
        }
        real_validate = validate_webhook_payload

        def reject_only_the_original_completed_payload(payload: dict) -> tuple[bool, list[str]]:
            if payload.get("status") == "completed":
                return False, ["Injected unrelated validation failure."]
            return real_validate(payload)

        with patch.object(worker, "_read_one_job", return_value={"msg_id": 77, "message": job}), patch.object(
            worker, "_preflight_scorer_config"
        ), patch.object(worker, "_audio_quality_gate", return_value=(None, {})), patch.object(
            worker, "_score", return_value=completed_result
        ), patch.object(worker, "validate_webhook_payload", side_effect=reject_only_the_original_completed_payload), patch.object(
            worker, "_post_webhook", return_value=SimpleNamespace(status_code=200)
        ) as post_webhook, patch.object(worker, "_archive_job") as archive_job:
            processed = worker._process_one_job(object(), config)  # noqa: SLF001

        self.assertTrue(processed)
        sent_payload = post_webhook.call_args.args[2]
        self.assertEqual(sent_payload["status"], "failed")
        self.assertIsNone(sent_payload["score"])
        valid, issues = real_validate(sent_payload)
        self.assertTrue(valid, issues)
        archive_job.assert_called_once_with(ANY, "practice_jobs", 77)


class WorkerQueuedAudioDownloadTests(unittest.TestCase):
    def test_storage_path_downloads_with_service_role_bucket(self) -> None:
        downloaded = b"stable-storage-audio"
        bucket = Mock()
        bucket.download.return_value = downloaded
        client = SimpleNamespace(storage=Mock())
        client.storage.from_.return_value = bucket

        def prepare(source: Path, target: Path) -> SimpleNamespace:
            self.assertEqual(source.read_bytes(), downloaded)
            self.assertEqual(source.suffix, ".webm")
            return SimpleNamespace(path=target)

        with patch("app.alignment.audio_preparation.prepare_audio_for_mfa", side_effect=prepare) as prepare_audio:
            with worker._prepared_audio_for_job(  # noqa: SLF001
                {"audio_url": "student-1/fresh-recording.webm"},
                storage_client=client,
                practice_audio_bucket="custom-practice-audio",
            ) as prepared:
                self.assertIsNotNone(prepared)

        client.storage.from_.assert_called_once_with("custom-practice-audio")
        bucket.download.assert_called_once_with("student-1/fresh-recording.webm")
        prepare_audio.assert_called_once()

    def test_legacy_signed_url_still_downloads_over_http(self) -> None:
        response = SimpleNamespace(content=b"legacy-signed-url-audio", raise_for_status=Mock())
        storage_client = SimpleNamespace(storage=Mock())

        def prepare(source: Path, target: Path) -> SimpleNamespace:
            self.assertEqual(source.read_bytes(), response.content)
            self.assertEqual(source.suffix, ".webm")
            return SimpleNamespace(path=target)

        with patch("app.audio.storage_resolver.requests.get", return_value=response) as get_audio, patch(
            "app.alignment.audio_preparation.prepare_audio_for_mfa", side_effect=prepare
        ):
            with worker._prepared_audio_for_job(  # noqa: SLF001
                {"audio_url": "https://storage.example.invalid/signed/recording.webm?token=legacy"},
                storage_client=storage_client,
            ):
                pass

        get_audio.assert_called_once_with(
            "https://storage.example.invalid/signed/recording.webm?token=legacy", timeout=30
        )
        response.raise_for_status.assert_called_once_with()
        storage_client.storage.from_.assert_not_called()

    def test_storage_download_failure_posts_terminal_failed_payload_and_archives(self) -> None:
        job = {
            "job_id": "job-storage-failure",
            "student_id": "student-1",
            "target_word": "book",
            "audio_url": "student-1/missing-recording.webm",
        }
        bucket = Mock()
        bucket.download.side_effect = RuntimeError("storage object missing")
        client = SimpleNamespace(storage=Mock())
        client.storage.from_.return_value = bucket
        config = {
            "queue_name": "practice_jobs",
            "visibility_timeout": 60,
            "scorer_mode": "cnn_attention_context",
            "alignment_mode": "mfa",
            "model_version": "test",
            "confidence_threshold": 0.65,
            "webhook_url": "https://backend.invalid/webhook",
            "webhook_secret": "test-secret",
            "practice_audio_bucket": "custom-practice-audio",
        }

        with patch.object(worker, "_read_one_job", return_value={"msg_id": 78, "message": job}), patch.object(
            worker, "_preflight_scorer_config"
        ), patch.object(worker, "_checkpoint_status", return_value={"checkpoint_configured": True, "checkpoint_exists": True}), patch.object(
            worker, "_post_webhook", return_value=SimpleNamespace(status_code=200)
        ) as post_webhook, patch.object(worker, "_archive_job") as archive_job:
            processed = worker._process_one_job(client, config)  # noqa: SLF001

        self.assertTrue(processed)
        client.storage.from_.assert_called_once_with("custom-practice-audio")
        bucket.download.assert_called_once_with("student-1/missing-recording.webm")
        sent_payload = post_webhook.call_args.args[2]
        self.assertEqual(sent_payload["status"], "failed")
        self.assertTrue(validate_webhook_payload(sent_payload)[0])
        archive_job.assert_called_once_with(client, "practice_jobs", 78)


if __name__ == "__main__":
    unittest.main()
