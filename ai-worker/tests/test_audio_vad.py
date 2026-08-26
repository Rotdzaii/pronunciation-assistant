from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import worker
from app.audio.preprocessing import HybridVadConfig, _resolve_speech_mask


class HybridVadDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HybridVadConfig(
            hop_ms=10.0,
            unvoiced_only_min_ratio=0.10,
            unvoiced_only_min_run_seconds=0.30,
            unvoiced_only_max_gap_seconds=0.12,
        )

    def _resolve(self, voiced: np.ndarray, unvoiced: np.ndarray) -> dict:
        return _resolve_speech_mask(
            voiced,
            unvoiced,
            sample_rate=1000,
            hop_length=10,
            cfg=self.config,
            np=np,
        )

    def test_normal_voiced_anchor_path_is_unchanged(self) -> None:
        voiced = np.zeros(100, dtype=bool)
        unvoiced = np.zeros(100, dtype=bool)
        voiced[40:45] = True
        unvoiced[35:52] = True
        result = self._resolve(voiced, unvoiced)
        self.assertEqual(result["speech_detection_mode"], "voiced_anchor")
        self.assertEqual(result["audio_quality_status"], "ok")
        self.assertIsNone(result["warning"])
        self.assertGreater(result["voiced_candidate_count"], 0)

    def test_pitch_degraded_unvoiced_only_is_warning_not_rejection(self) -> None:
        voiced = np.zeros(170, dtype=bool)
        unvoiced = np.zeros(170, dtype=bool)
        unvoiced[50:80] = True  # 0.176 ratio, 0.30 s sustained run
        result = self._resolve(voiced, unvoiced)
        self.assertEqual(result["voiced_candidate_count"], 0)
        self.assertAlmostEqual(result["unvoiced_frames_ratio"], 30 / 170, places=4)
        self.assertGreaterEqual(result["largest_unvoiced_run_seconds"], 0.30)
        self.assertEqual(result["speech_detection_mode"], "pitch_degraded_unvoiced_only")
        self.assertEqual(result["audio_quality_status"], "warning")
        self.assertEqual(result["warning"], "pitch_degraded_unvoiced_only")

    def test_silence_without_sustained_unvoiced_activity_is_invalid(self) -> None:
        result = self._resolve(np.zeros(170, dtype=bool), np.zeros(170, dtype=bool))
        self.assertEqual(result["speech_detection_mode"], "no_voiced_anchor")
        self.assertEqual(result["audio_quality_status"], "invalid")
        self.assertFalse(np.any(result["speech_mask"]))

    def test_short_burst_noise_is_rejected_even_when_ratio_is_high(self) -> None:
        voiced = np.zeros(170, dtype=bool)
        unvoiced = np.zeros(170, dtype=bool)
        unvoiced[70:88] = True  # 0.106 ratio, only 0.18 s
        result = self._resolve(voiced, unvoiced)
        self.assertGreaterEqual(result["unvoiced_frames_ratio"], 0.10)
        self.assertLess(result["largest_unvoiced_run_seconds"], 0.30)
        self.assertEqual(result["audio_quality_status"], "invalid")


class AudioQualityGateTests(unittest.TestCase):
    def test_pitch_degraded_warning_allows_mfa_and_cnn_to_continue(self) -> None:
        quality = {
            "snr_db": 22.7,
            "file_duration_seconds": 1.7,
            "voiced_anchor_duration_seconds": 0.0,
            "detected_speech_duration_seconds": 0.30,
            "voiced_duration_seconds": 0.30,
            "voiced_frames_ratio": 0.0,
            "unvoiced_frames_ratio": 0.177,
            "mean_voiced_prob": 0.019,
            "finite_f0_frames": 0,
            "finite_f0_ratio": 0.0,
            "f0_min_hz": None,
            "f0_max_hz": None,
            "pyin_voiced_flag_ratio": 0.0,
            "voiced_candidate_count": 0,
            "unvoiced_candidate_count": 30,
            "largest_unvoiced_run_seconds": 0.30,
            "speech_detection_mode": "pitch_degraded_unvoiced_only",
            "audio_quality_status": "warning",
            "warning": "pitch_degraded_unvoiced_only",
        }
        with patch("app.audio.preprocessing.estimate_snr", return_value=quality):
            rejected, returned = worker._audio_quality_gate(Path("prepared.wav"), {"scorer_mode": "cnn_attention_context"})
        self.assertIsNone(rejected)
        self.assertEqual(returned["warning"], "pitch_degraded_unvoiced_only")

    def test_silence_is_hard_rejected_without_downstream_inference(self) -> None:
        quality = {
            "snr_db": 16.7,
            "voiced_duration_seconds": 0.0,
            "voiced_frames_ratio": 0.0,
            "unvoiced_frames_ratio": 0.01,
            "mean_voiced_prob": 0.0,
            "speech_detection_mode": "no_voiced_anchor",
            "audio_quality_status": "invalid",
            "largest_unvoiced_run_seconds": 0.01,
        }
        with patch("app.audio.preprocessing.estimate_snr", return_value=quality):
            rejected, _ = worker._audio_quality_gate(Path("prepared.wav"), {"scorer_mode": "cnn_attention_context"})
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected["metadata"]["rejection_reason"], "insufficient_sustained_unvoiced_activity")
        self.assertIsNone(rejected["score"])
        self.assertEqual(rejected["problem_phonemes"], [])

    def test_pitch_degraded_activity_that_cannot_align_has_no_score_or_phone_feedback(self) -> None:
        failed = worker._build_failed_result(
            {"job_id": "job-1", "target_word": "report"},
            {
                "scorer_mode": "cnn_attention_context",
                "model_version": "test",
                "alignment_mode": "mfa",
            },
            0.5,
            worker.PhoenixWorkerError("MFA alignment failed", "alignment_failed"),
        )

        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["score"])
        self.assertEqual(failed["score_type"], "unavailable")
        self.assertEqual(failed["problem_phonemes"], [])
        self.assertIsNone(failed["feedback"].get("primary_feedback"))


if __name__ == "__main__":
    unittest.main()
