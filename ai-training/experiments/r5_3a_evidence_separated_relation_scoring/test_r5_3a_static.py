"""Synthetic-only static verification for the frozen R5-3A contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from r5_3a_classifier import addition_probability, make_classifier, make_scaler
from r5_3a_features import (
    FEATURE_ORDER,
    construct_feature_matrix,
    construct_feature_row,
    validate_design_matrix,
    validate_feature_names,
)
from r5_3a_loso import event_from_decision, run_fold, run_loso
from r5_3a_threshold import (
    ThresholdEvaluation,
    apply_threshold,
    choose_best_evaluation,
    threshold_candidates,
)


RESEARCH = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research")
R5_3A = RESEARCH / "ai-training/experiments/r5_3a_evidence_separated_relation_scoring"
R5_1A = RESEARCH / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
R5_2B = RESEARCH / "ai-training/experiments/r5_2b_relation_competition"

EXPECTED = {
    "contract": "A1594B45A8EE9C542FEFF28695DA27D59D07C4A37596EAB1D4308161A7B583B3",
    "preregistration": "2239C047504B4B6CA212A40D0284DD38E626CF6E73D541E8AF036311BC8A6CB1",
    "contract_manifest": "CB3C4EC1E2402F3821BF61CA9302731D56CE6FCED2B025295FF6FC780D521D6D",
    "r5_1a_manifest": "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    "r5_2b_manifest": "37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387",
    "r5_2c_manifest": "C18621208A35074B7681EFC95261B73147B4A6E3716F17AF0AB0471F8B328F90",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def audit_manifest(root: Path, name: str) -> tuple[str, int, list[str]]:
    path = root / name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("artifacts", manifest.get("files", []))
    failures: list[str] = []
    for entry in entries:
        rel = entry.get("relative_path", entry.get("path"))
        size = entry.get("byte_size", entry.get("size_bytes"))
        target = root / rel
        if not target.is_file():
            failures.append(f"missing:{rel}")
            continue
        if target.stat().st_size != int(size):
            failures.append(f"size:{rel}")
        if sha256(target) != str(entry["sha256"]).upper():
            failures.append(f"hash:{rel}")
    return sha256(path), len(entries), failures


def first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"empty JSONL: {path}")


def synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    speakers = ["S0", "S1", "S2", "S3"]
    templates = [
        (-2.0, 0.1, -0.2, 0, True),
        (-1.0, -0.4, 0.3, 0, False),
        (2.0, 0.2, -0.1, 1, False),
        (1.1, -0.3, 0.4, 1, False),
        (-0.5, 0.5, 0.2, 0, True),
        (-0.8, -0.2, -0.6, 0, True),
    ]
    for speaker_index, speaker in enumerate(speakers):
        for row_index, (a, s, d, label, correct) in enumerate(templates):
            offset = speaker_index * 0.07
            if speaker == "S3":
                # Extreme held-out values make scaler leakage immediately visible.
                offset += 1000.0
            rows.append(
                {
                    "row_id": f"{speaker}/row_{row_index}",
                    "speaker": speaker,
                    "A": np.float64(a + offset),
                    "S": np.float64(s + 2.0 * offset),
                    "D": np.float64(d - 0.5 * offset),
                    "label": label,
                    "correct_only_negative": bool(correct and label == 0),
                    "best_insert_phone": f"P{row_index}",
                    "best_insert_boundary": row_index,
                }
            )
    return rows


class IdentityTests(unittest.TestCase):
    def test_contract_identity_and_manifest_entries(self) -> None:
        self.assertEqual(sha256(R5_3A / "R5_3A_DEVELOPMENT_CONTRACT.json"), EXPECTED["contract"])
        self.assertEqual(sha256(R5_3A / "R5_3A_PREREGISTRATION.md"), EXPECTED["preregistration"])
        actual, count, failures = audit_manifest(R5_3A, "R5_3A_CONTRACT_MANIFEST.json")
        self.assertEqual(actual, EXPECTED["contract_manifest"])
        self.assertEqual(count, 9)
        self.assertEqual(failures, [])

    def test_upstream_manifest_identities_and_entries(self) -> None:
        actual_a, count_a, failures_a = audit_manifest(R5_1A, "R5_1A_EXECUTION_MANIFEST.json")
        actual_b, count_b, failures_b = audit_manifest(R5_2B, "R5_2B_TC1_PA1_ENV_EXECUTION_MANIFEST.json")
        actual_c, count_c, failures_c = audit_manifest(R5_2B, "R5_2C_MANIFEST.json")
        self.assertEqual((actual_a, count_a, failures_a), (EXPECTED["r5_1a_manifest"], 14, []))
        self.assertEqual((actual_b, count_b, failures_b), (EXPECTED["r5_2b_manifest"], 21, []))
        self.assertEqual((actual_c, count_c, failures_c), (EXPECTED["r5_2c_manifest"], 15, []))

    def test_v4_and_checkpoint_identity_without_loading(self) -> None:
        v4 = RESEARCH / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
        checkpoint = RESEARCH / "ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
        self.assertEqual(sha256(v4), EXPECTED["v4"])
        self.assertEqual(sha256(checkpoint), EXPECTED["checkpoint"])


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old = {"source_identity": "synthetic/1", "addition_score_A_value": 10.25}
        self.new = {
            "source_identity": "synthetic/1",
            "keep_target_score_value": -2.0,
            "best_sub_score_value": 3.5,
            "best_delete_score_value": -7.0,
        }

    def test_exact_feature_formulas_and_asymmetric_order(self) -> None:
        feature = construct_feature_row(self.old, self.new)
        np.testing.assert_array_equal(feature, np.asarray([10.25, 5.5, -5.0], dtype=np.float64))
        self.assertEqual(tuple(FEATURE_ORDER), ("A", "S", "D"))

    def test_matrix_has_only_three_columns(self) -> None:
        feature = construct_feature_row(self.old, self.new).reshape(1, -1)
        validate_design_matrix(feature)
        self.assertEqual(feature.shape, (1, 3))
        with self.assertRaises(ValueError):
            validate_design_matrix(np.zeros((2, 4), dtype=np.float64))

    def test_forbidden_and_swapped_feature_names_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_names(("A", "S", "D", "C"))
        with self.assertRaises(ValueError):
            validate_feature_names(("S", "A", "D"))

    def test_nonfinite_values_rejected(self) -> None:
        bad = dict(self.new)
        bad["best_sub_score_value"] = float("nan")
        with self.assertRaises(ValueError):
            construct_feature_row(self.old, bad)

    def test_identity_mismatch_rejected(self) -> None:
        bad = dict(self.new)
        bad["source_identity"] = "synthetic/other"
        with self.assertRaises(ValueError):
            construct_feature_row(self.old, bad)

    def test_exact_matrix_join_and_order(self) -> None:
        old_rows = [self.old, {"source_identity": "synthetic/2", "addition_score_A_value": -3.0}]
        new_rows = [
            {"source_identity": "synthetic/2", "keep_target_score_value": 1.0, "best_sub_score_value": 1.25, "best_delete_score_value": 5.0},
            self.new,
        ]
        matrix, identities = construct_feature_matrix(old_rows, new_rows)
        self.assertEqual(identities, ["synthetic/2", "synthetic/1"])
        np.testing.assert_array_equal(matrix[0], np.asarray([-3.0, 0.25, 4.0]))
        np.testing.assert_array_equal(matrix[1], np.asarray([10.25, 5.5, -5.0]))

    def test_duplicate_and_missing_joins_rejected(self) -> None:
        with self.assertRaises(ValueError):
            construct_feature_matrix([self.old, self.old], [self.new])
        with self.assertRaises(ValueError):
            construct_feature_matrix([self.old], [])


class ClassifierTests(unittest.TestCase):
    def test_scaler_configuration_exact(self) -> None:
        params = make_scaler().get_params(deep=False)
        self.assertEqual(params, {"copy": True, "with_mean": True, "with_std": True})

    def test_classifier_configuration_exact(self) -> None:
        params = make_classifier().get_params(deep=False)
        self.assertEqual(params["penalty"], "l2")
        self.assertEqual(params["C"], 1.0)
        self.assertEqual(params["solver"], "lbfgs")
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["max_iter"], 1000)
        self.assertTrue(params["fit_intercept"])
        self.assertIsNone(params["random_state"])

    def test_sklearn_provenance(self) -> None:
        self.assertEqual(sklearn.__version__, "1.8.0")

    def test_probability_selects_explicit_addition_class(self) -> None:
        class FakeModel:
            classes_ = np.asarray([1, 0])

            def predict_proba(self, features: np.ndarray) -> np.ndarray:
                return np.asarray([[0.8, 0.2], [0.6, 0.4]])

            def predict(self, features: np.ndarray) -> np.ndarray:
                raise AssertionError("predict must not be used")

            def decision_function(self, features: np.ndarray) -> np.ndarray:
                raise AssertionError("decision_function must not be used")

        result = addition_probability(FakeModel(), np.zeros((2, 3), dtype=np.float64))
        np.testing.assert_array_equal(result, np.asarray([0.8, 0.6]))

    def test_missing_addition_class_rejected(self) -> None:
        class BadModel:
            classes_ = np.asarray([0, 2])

            def predict_proba(self, features: np.ndarray) -> np.ndarray:
                return np.asarray([[0.5, 0.5]])

        with self.assertRaises(ValueError):
            addition_probability(BadModel(), np.zeros((1, 3), dtype=np.float64))


class LeakageAndLosoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = synthetic_rows()

    def test_scaler_fit_is_calibration_only(self) -> None:
        _, audit = run_fold(self.rows, "S3")
        calibration = np.asarray(audit["calibration_feature_matrix"], dtype=np.float64)
        np.testing.assert_allclose(audit["scaler_mean"], calibration.mean(axis=0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(audit["scaler_var"], calibration.var(axis=0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(audit["scaler_scale"], np.sqrt(calibration.var(axis=0)), rtol=0, atol=1e-15)
        self.assertTrue(max(audit["scaler_mean"]) < 1.0)

    def test_model_fit_and_threshold_rows_are_calibration_only(self) -> None:
        _, audit = run_fold(self.rows, "S3")
        expected = [row["row_id"] for row in self.rows if row["speaker"] != "S3"]
        heldout = [row["row_id"] for row in self.rows if row["speaker"] == "S3"]
        self.assertEqual(audit["scaler_fit_row_ids"], expected)
        self.assertEqual(audit["model_fit_row_ids"], expected)
        self.assertEqual(audit["threshold_selection_row_ids"], expected)
        self.assertTrue(set(heldout).isdisjoint(audit["model_fit_row_ids"]))

    def test_heldout_uses_calibration_scaler(self) -> None:
        _, audit = run_fold(self.rows, "S3")
        heldout = np.asarray(audit["heldout_feature_matrix"], dtype=np.float64)
        expected = (heldout - np.asarray(audit["scaler_mean"])) / np.asarray(audit["scaler_scale"])
        np.testing.assert_allclose(audit["heldout_transformed"], expected, rtol=0, atol=1e-12)
        self.assertTrue(audit["same_scaler_used_for_heldout_transform"])

    def test_heldout_label_change_cannot_change_fold_threshold(self) -> None:
        _, before = run_fold(self.rows, "S3")
        changed = [dict(row) for row in self.rows]
        for row in changed:
            if row["speaker"] == "S3":
                row["label"] = 1 - row["label"]
        _, after = run_fold(changed, "S3")
        self.assertEqual(before["threshold"], after["threshold"])
        self.assertEqual(before["scaler_mean"], after["scaler_mean"])
        self.assertEqual(before["model_coefficients"], after["model_coefficients"])
        self.assertEqual(before["model_intercept"], after["model_intercept"])

    def test_loso_oof_uniqueness_and_deterministic_fold_order(self) -> None:
        outputs, audits = run_loso(self.rows, ["S0", "S1", "S2", "S3"])
        output_ids = [row["row_id"] for row in outputs]
        self.assertEqual(len(output_ids), len(self.rows))
        self.assertEqual(len(output_ids), len(set(output_ids)))
        self.assertEqual(set(output_ids), {row["row_id"] for row in self.rows})
        self.assertEqual([fold["heldout_speaker"] for fold in audits], ["S0", "S1", "S2", "S3"])

    def test_invalid_speaker_order_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_loso(self.rows, ["S0", "S1", "S2"])


class ThresholdTests(unittest.TestCase):
    def test_candidate_set_exact_and_no_arbitrary_half(self) -> None:
        scores = np.asarray([0.2, 0.8, 0.2], dtype=np.float64)
        actual = threshold_candidates(scores)
        expected = np.asarray(
            [np.nextafter(0.2, -np.inf), 0.2, 0.8, np.nextafter(0.8, np.inf)],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertNotIn(0.5, actual.tolist())

    def test_threshold_equality_is_addition(self) -> None:
        result = apply_threshold(np.asarray([0.49, 0.5, 0.51]), 0.5)
        np.testing.assert_array_equal(result, np.asarray([False, True, True]))

    def test_tie_priority_macro_f1(self) -> None:
        lower = ThresholdEvaluation(0.9, 0.7, 0.9, 0.0)
        winner = ThresholdEvaluation(0.1, 0.8, 0.1, 1.0)
        self.assertEqual(choose_best_evaluation([lower, winner]), winner)

    def test_tie_priority_addition_f1(self) -> None:
        lower = ThresholdEvaluation(0.9, 0.8, 0.2, 0.0)
        winner = ThresholdEvaluation(0.1, 0.8, 0.3, 1.0)
        self.assertEqual(choose_best_evaluation([lower, winner]), winner)

    def test_tie_priority_correct_far(self) -> None:
        lower = ThresholdEvaluation(0.9, 0.8, 0.3, 0.2)
        winner = ThresholdEvaluation(0.1, 0.8, 0.3, 0.1)
        self.assertEqual(choose_best_evaluation([lower, winner]), winner)

    def test_tie_priority_higher_threshold(self) -> None:
        lower = ThresholdEvaluation(0.1, 0.8, 0.3, 0.1)
        winner = ThresholdEvaluation(0.9, 0.8, 0.3, 0.1)
        self.assertEqual(choose_best_evaluation([lower, winner]), winner)


class EventAndSchemaTests(unittest.TestCase):
    def test_event_metadata_is_immutable_for_positive(self) -> None:
        row = {"best_insert_phone": "ZH", "best_insert_boundary": 7}
        self.assertEqual(event_from_decision(row, True), {"phone": "ZH", "boundary": 7})

    def test_negative_has_no_addition_event(self) -> None:
        row = {"best_insert_phone": "ZH", "best_insert_boundary": 7}
        self.assertIsNone(event_from_decision(row, False))

    def test_real_artifact_schema_only(self) -> None:
        old = first_jsonl(R5_1A / "r5_1a_train_scores.jsonl")
        new = first_jsonl(R5_2B / "r5_2b_tc1_pa1_env_train_scores.jsonl")
        self.assertTrue({"source_identity", "addition_score_A_value"}.issubset(old))
        self.assertTrue(
            {
                "source_identity",
                "keep_target_score_value",
                "best_sub_score_value",
                "best_delete_score_value",
            }.issubset(new)
        )
        self.assertIsInstance(old["source_identity"], str)
        self.assertIsInstance(new["source_identity"], str)
        self.assertIsInstance(old["addition_score_A_value"], (int, float))
        self.assertIsInstance(new["best_sub_score_value"], (int, float))
        self.assertIsInstance(new["best_delete_score_value"], (int, float))

    def test_frozen_population_metadata_only(self) -> None:
        identity = json.loads((R5_3A / "r5_3a_source_identity.json").read_text(encoding="utf-8"))
        provenance = identity["feature_provenance_check"]
        self.assertEqual(provenance["matched_rows"], 16582)
        self.assertEqual(provenance["positive_words"], 323)
        self.assertEqual(provenance["negative_words"], 16259)
        self.assertFalse(provenance["performance_metrics_calculated"])


def deterministic_summary(result: unittest.TestResult) -> dict[str, Any]:
    rows = synthetic_rows()
    outputs, audits = run_loso(rows, ["S0", "S1", "S2", "S3"])
    _, leakage = run_fold(rows, "S3")
    candidates = threshold_candidates(np.asarray([0.2, 0.8, 0.2], dtype=np.float64))
    old_schema = first_jsonl(R5_1A / "r5_1a_train_scores.jsonl")
    new_schema = first_jsonl(R5_2B / "r5_2b_tc1_pa1_env_train_scores.jsonl")
    return {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "sklearn_version": sklearn.__version__,
        "feature_order": list(FEATURE_ORDER),
        "synthetic_fixture": {
            "speaker_order": ["S0", "S1", "S2", "S3"],
            "row_count": len(rows),
            "fold_count": len(audits),
            "thresholds": [fold["threshold"] for fold in audits],
            "oof": outputs,
        },
        "leakage_fixture": {
            "heldout_speaker": "S3",
            "scaler_mean": leakage["scaler_mean"],
            "scaler_scale": leakage["scaler_scale"],
            "scaler_var": leakage["scaler_var"],
            "calibration_row_ids": leakage["calibration_row_ids"],
            "heldout_row_ids": leakage["heldout_row_ids"],
            "scaler_fit_row_ids": leakage["scaler_fit_row_ids"],
            "model_fit_row_ids": leakage["model_fit_row_ids"],
            "threshold_selection_row_ids": leakage["threshold_selection_row_ids"],
            "same_scaler_used_for_heldout_transform": leakage["same_scaler_used_for_heldout_transform"],
            "threshold": leakage["threshold"],
        },
        "threshold_fixture": {
            "input": [0.2, 0.8, 0.2],
            "candidate_hex": [float(value).hex() for value in candidates],
            "decision_at_0_5": apply_threshold(np.asarray([0.49, 0.5, 0.51]), 0.5).tolist(),
            "tie_priorities_verified": ["binary_macro_f1", "addition_f1", "correct_only_far", "higher_threshold"],
        },
        "event_fixture": {
            "positive": event_from_decision({"best_insert_phone": "ZH", "best_insert_boundary": 7}, True),
            "negative": event_from_decision({"best_insert_phone": "ZH", "best_insert_boundary": 7}, False),
        },
        "real_schema_only": {
            "r5_1a_fields_present": sorted({"source_identity", "addition_score_A_value"}.intersection(old_schema)),
            "r5_2b_fields_present": sorted(
                {"source_identity", "keep_target_score_value", "best_sub_score_value", "best_delete_score_value"}.intersection(new_schema)
            ),
            "rows_read_per_artifact": 1,
            "classifier_fit_on_real_rows": False,
        },
        "protocol": {
            "real_train_fitting": False,
            "real_performance_metrics": False,
            "checkpoint_inference": False,
            "audio_access": False,
            "validation_access": False,
            "test_access": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = deterministic_summary(result)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
