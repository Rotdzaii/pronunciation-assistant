from __future__ import annotations

import unittest

from app.contracts.ai_result_contract import MODEL_CAPABILITY, build_ai_result
from app.contracts.scoring_contract import has_public_pronunciation_score
from app.contracts.webhook_payload import build_success_webhook_payload
from app.scorers import cnn_attention_scorer


class PhoenixV2SemanticsTests(unittest.TestCase):
    def test_cnn_inventory_has_only_error_classes(self) -> None:
        self.assertEqual(cnn_attention_scorer.LABEL_ORDER, ["addition", "deletion", "substitution"])
        self.assertNotIn("correct", cnn_attention_scorer.LABEL_ORDER)

    def test_confidence_never_becomes_pronunciation_score(self) -> None:
        result = build_ai_result(
            score=95,
            predicted_error_type="substitution",
            diagnosis_confidence=0.95,
            problem_phonemes=["b"],
        )

        self.assertEqual(result["model_capability"], MODEL_CAPABILITY)
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")
        self.assertEqual(result["diagnosis"]["diagnosis_confidence"], 0.95)
        self.assertNotIn("pronunciation_score_source", result)
        self.assertFalse(has_public_pronunciation_score(result))

    def test_three_class_output_is_a_suspected_not_confirmed_error(self) -> None:
        result = cnn_attention_scorer._aggregate_segment_predictions(  # noqa: SLF001
            [{
                "index": 0,
                "phone": "B",
                "word": "book",
                "start": 0.0,
                "end": 0.2,
                "predicted_error_type": "substitution",
                "diagnosis_confidence": 0.95,
                "class_probabilities": {"substitution": 0.95},
            }],
            alignment_result={
                "status": "success",
                "method": "mfa",
                "metadata": {"is_forced_alignment": True, "mfa_used": True},
                "words": [],
                "phones": [{"phone": "B", "start": 0.0, "end": 0.2}],
            },
        )

        self.assertEqual(result["model_capability"], MODEL_CAPABILITY)
        self.assertEqual(result["predicted_error_type"], "substitution")
        self.assertFalse(result["diagnosis"]["is_confirmed_error"])
        self.assertTrue(all(not segment["is_confirmed_error"] for segment in result["segments"]))
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")

    def test_invalid_reliability_hides_every_diagnosis_field(self) -> None:
        result = cnn_attention_scorer._aggregate_segment_predictions(  # noqa: SLF001
            [{
                "index": 0,
                "phone": "B",
                "word": "book",
                "start": 0.0,
                "end": 0.2,
                "predicted_error_type": "substitution",
                "diagnosis_confidence": 0.95,
                "class_probabilities": {"substitution": 0.95},
            }],
            alignment_result={
                "status": "failed",
                "method": "fallback_even_split",
                "metadata": {"fallback_alignment": True},
                "words": [],
                "phones": [{"phone": "B", "start": 0.0, "end": 0.2}],
            },
        )

        self.assertEqual(result["reliability"]["status"], "invalid")
        self.assertIsNone(result["predicted_error_type"])
        self.assertEqual(result["problem_phonemes"], [])
        self.assertIsNone(result["diagnosis"]["suspected_problem_phone"])
        self.assertIsNone(result["diagnosis"]["diagnosis_confidence"])
        self.assertEqual(result["diagnosis"]["class_probabilities"], {})
        self.assertFalse(result["diagnosis"]["is_confirmed_error"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")

    def test_teacher_rhotic_alignment_keeps_classifier_only_semantics(self) -> None:
        result = cnn_attention_scorer._aggregate_segment_predictions(  # noqa: SLF001
            [
                {
                    "index": 0,
                    # Regression: prediction data can lose the raw phone even
                    # though the matching MFA timing segment has one.
                    "phone": None,
                    "word": "teacher",
                    "start": 0.0,
                    "end": 0.1,
                    "predicted_error_type": "substitution",
                    "diagnosis_confidence": 0.97,
                    "class_probabilities": {"substitution": 0.97},
                },
                {
                    "index": 1,
                    "phone": None,
                    "word": "teacher",
                    "start": 0.1,
                    "end": 0.2,
                    "predicted_error_type": "addition",
                    "diagnosis_confidence": 0.87,
                    "class_probabilities": {"addition": 0.87},
                },
                {
                    "index": 2,
                    "phone": None,
                    "word": "teacher",
                    "start": 0.2,
                    "end": 0.3,
                    "predicted_error_type": "deletion",
                    "diagnosis_confidence": 0.76,
                    "class_probabilities": {"deletion": 0.76},
                },
                {
                    "index": 3,
                    "phone": None,
                    "word": "teacher",
                    "start": 0.3,
                    "end": 0.5,
                    "predicted_error_type": "substitution",
                    "diagnosis_confidence": 0.65,
                    "class_probabilities": {"substitution": 0.65},
                },
            ],
            alignment_result={
                "status": "warning",
                "method": "mfa",
                "metadata": {
                    "is_forced_alignment": True,
                    "mfa_used": True,
                    "alignment_quality_warnings": [
                        "raw_audio_coverage_below_minimum",
                        "leading_silence_high",
                        "trailing_silence_high",
                    ],
                },
                "words": [],
                "phones": [
                    {"phone": "tj", "start": 0.0, "end": 0.1},
                    {"phone": "iː", "start": 0.1, "end": 0.2},
                    {"phone": "tʃ", "start": 0.2, "end": 0.3},
                    {"phone": "ɚ", "start": 0.3, "end": 0.5},
                ],
            },
        )

        self.assertEqual(result["reliability"]["status"], "valid_with_warning")
        self.assertEqual(result["reliability"]["failures"], [])
        self.assertEqual([segment["phone"] for segment in result["segments"]], ["t", "iː", "tʃ", "ɚ"])
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 0.1)
        self.assertEqual(result["diagnosis"]["suspected_problem_phone"], "t")
        self.assertEqual(result["diagnosis"]["predicted_error_type"], "substitution")
        self.assertEqual(result["diagnosis"]["diagnosis_confidence"], 0.97)
        self.assertEqual(result["model_capability"], MODEL_CAPABILITY)
        self.assertIsNone(result["score"])
        self.assertEqual(result["score_type"], "unavailable")
        self.assertFalse(result["diagnosis"]["is_confirmed_error"])
        payload = build_success_webhook_payload("teacher-job", result)
        self.assertNotIn("tj", str(payload))

    def test_legacy_score_fields_are_not_public_scores(self) -> None:
        self.assertFalse(
            has_public_pronunciation_score(
                {
                    "score": 99,
                    "score_type": "model",
                    "utterance_segmental_score": 99.0,
                    "metadata": {"is_real_gop": True},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
