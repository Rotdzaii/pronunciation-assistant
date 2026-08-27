"""Deterministic synthetic/static verification for the frozen R6-0 contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import sys
import unittest
from pathlib import Path

import torch

import r6_0_boundary_labels as labels
import r6_0_boundary_mapping as mapping
import r6_0_local_features as features


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPECTED_CONTRACT_MANIFEST_SHA = "EDB1A62CD6350AFC955C7A49B51C668BD0ED4217F7BBC9AD916525769DF8718A"
EXPECTED_STRUCTURE = {
    "words": 16582,
    "expected_boundary_instances": 74426,
    "positive_boundary_identities": 324,
    "negative_boundary_identities": 74102,
    "runtime_addition_events": 342,
    "single_addition_words": 304,
    "multiple_addition_words": 19,
    "mixed_substitution_addition_words": 117,
    "mixed_deletion_addition_words": 26,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def synthetic_logits() -> tuple[torch.Tensor, list[list[float]]]:
    weight_rows = [
        [float(index + 1) for index in range(41)],
        [float((index + 1) ** 2) for index in range(41)],
        [float(41 - index) for index in range(41)],
    ]
    probabilities = [[value / sum(row) for value in row] for row in weight_rows]
    logits = torch.log(torch.tensor(probabilities, dtype=torch.float32))
    return logits, probabilities


class IdentityTests(unittest.TestCase):
    def test_01_r6_manifest_hash(self) -> None:
        self.assertEqual(sha256(EXPERIMENT_DIR / "R6_0_MANIFEST.json"), EXPECTED_CONTRACT_MANIFEST_SHA)

    def test_02_r6_manifest_entries(self) -> None:
        manifest = json.loads((EXPERIMENT_DIR / "R6_0_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["artifact_count"], 8)
        for entry in manifest["artifacts"]:
            path = EXPERIMENT_DIR / entry["relative_path"]
            self.assertEqual(path.stat().st_size, entry["byte_size"])
            self.assertEqual(sha256(path), entry["sha256"])


class TimeMappingTests(unittest.TestCase):
    def test_03_gold_boundary_formula(self) -> None:
        intervals = [(10.0, 10.1), (10.14, 10.24), (10.24, 10.31)]
        self.assertEqual(mapping.gold_boundary_time(intervals, 0), 10.0)
        self.assertEqual(mapping.gold_boundary_time(intervals, 1), 10.12)
        self.assertEqual(mapping.gold_boundary_time(intervals, 3), 10.31)

    def test_04_exact_center(self) -> None:
        self.assertEqual(mapping.nearest_output_index(10.065, 10.0, 8, 10.2), 3)

    def test_05_closer_to_lower(self) -> None:
        self.assertEqual(mapping.nearest_output_index(10.061, 10.0, 8, 10.2), 3)

    def test_06_closer_to_upper(self) -> None:
        self.assertEqual(mapping.nearest_output_index(10.081, 10.0, 8, 10.2), 4)

    def test_07_exact_tie_selects_lower(self) -> None:
        self.assertEqual(mapping.nearest_output_index(10.075, 10.0, 8, 10.2), 3)

    def test_08_near_start(self) -> None:
        self.assertEqual(mapping.nearest_output_index(20.0001, 20.0, 6, 20.2), 0)

    def test_09_near_final_frame(self) -> None:
        self.assertEqual(mapping.nearest_output_index(20.18, 20.0, 6, 20.2), 5)

    def test_10_time_reference_subtracts_once(self) -> None:
        self.assertEqual(mapping.relative_boundary_time(12.365, 12.3), 0.065)
        self.assertEqual(mapping.nearest_output_index(12.365, 12.3, 8, 12.5), 3)

    def test_11_seconds_milliseconds_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mapping.nearest_output_index(12365.0, 12.3, 8, 12.5)

    def test_12_outside_word_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mapping.nearest_output_index(9.99, 10.0, 8, 10.2)


class WindowTests(unittest.TestCase):
    def test_13_window_k0(self) -> None:
        self.assertEqual(mapping.five_step_window(0, 6), [0, 1, 2])

    def test_14_window_k1(self) -> None:
        self.assertEqual(mapping.five_step_window(1, 6), [0, 1, 2, 3])

    def test_15_window_interior(self) -> None:
        self.assertEqual(mapping.five_step_window(3, 7), [1, 2, 3, 4, 5])

    def test_16_window_t_minus_2(self) -> None:
        self.assertEqual(mapping.five_step_window(4, 6), [2, 3, 4, 5])

    def test_17_window_t_minus_1(self) -> None:
        self.assertEqual(mapping.five_step_window(5, 6), [3, 4, 5])


class FeatureTests(unittest.TestCase):
    def test_18_posterior_mass(self) -> None:
        logits, _ = synthetic_logits()
        posterior = features.posterior_from_logits(logits)
        self.assertTrue(torch.allclose(posterior.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0.0))

    def test_19_adjacent_sets(self) -> None:
        expected = [4, 7, 7, 9]
        self.assertEqual(features.adjacent_expected_phone_set(expected, 0), frozenset({4}))
        self.assertEqual(features.adjacent_expected_phone_set(expected, 1), frozenset({4, 7}))
        self.assertEqual(features.adjacent_expected_phone_set(expected, 2), frozenset({7}))
        self.assertEqual(features.adjacent_expected_phone_set(expected, 4), frozenset({9}))

    def test_20_primary_formula_and_exclusions(self) -> None:
        logits, probabilities = synthetic_logits()
        result = features.compute_local_evidence(logits, [0, 1, 2], [1, 2], 1)
        expected_rows = [sum(row[:40]) - row[1] - row[2] for row in probabilities]
        self.assertAlmostEqual(result["mean_unexpected_phone_mass"], sum(expected_rows) / 3, places=7)

    def test_21_blank_excluded_from_primary(self) -> None:
        probabilities = [0.001] * 40 + [0.96]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        logits = torch.log(torch.tensor([probabilities], dtype=torch.float32))
        result = features.compute_local_evidence(logits, [0], [0], 0)
        expected = sum(probabilities[1:40])
        self.assertAlmostEqual(result["mean_unexpected_phone_mass"], expected, places=7)

    def test_22_peak_unexpected_value_and_identity(self) -> None:
        logits, probabilities = synthetic_logits()
        result = features.compute_local_evidence(logits, [0, 1, 2], [1, 2], 1)
        candidates = [
            (value, frame, phone)
            for frame, row in enumerate(probabilities)
            for phone, value in enumerate(row[:40])
            if phone not in {1, 2}
        ]
        expected_value, expected_frame, expected_phone = max(candidates, key=lambda item: (item[0], -item[1], -item[2]))
        self.assertAlmostEqual(result["peak_unexpected_phone_posterior"], expected_value, places=7)
        self.assertEqual(result["peak_unexpected_frame_index"], expected_frame)
        self.assertEqual(result["peak_unexpected_phone_index"], expected_phone)

    def test_23_peak_tie_is_first_frame_then_lower_phone(self) -> None:
        logits = torch.zeros((2, 41), dtype=torch.float32)
        result = features.compute_local_evidence(logits, [0, 1], [39], 0)
        self.assertEqual(result["peak_unexpected_frame_index"], 0)
        self.assertEqual(result["peak_unexpected_phone_index"], 0)

    def test_24_nonblank_formulations_agree(self) -> None:
        logits, _ = synthetic_logits()
        result = features.compute_local_evidence(logits, [0, 1, 2], [1, 2], 1)
        self.assertAlmostEqual(result["mean_nonblank_mass"], result["mean_nonblank_mass_from_phone_sum"], places=6)

    def test_25_exact_feature_surface(self) -> None:
        logits, _ = synthetic_logits()
        result = features.compute_local_evidence(logits, [0, 1, 2], [1, 2], 1)
        self.assertEqual(
            set(result),
            {
                "window_indices", "adjacent_expected_phone_ids", "mean_unexpected_phone_mass",
                "peak_unexpected_phone_posterior", "peak_unexpected_phone_index",
                "peak_unexpected_frame_index", "mean_nonblank_mass",
                "mean_nonblank_mass_from_phone_sum",
            },
        )

    def test_26_local_window_only(self) -> None:
        logits, _ = synthetic_logits()
        base = features.compute_local_evidence(logits, [1], [1, 2], 1)
        changed = logits.clone()
        changed[0, 10] += 100.0
        changed[2, 10] += 100.0
        after = features.compute_local_evidence(changed, [1], [1, 2], 1)
        self.assertEqual(base, after)


class LabelTests(unittest.TestCase):
    def test_27_correct_word_all_negative(self) -> None:
        rows = labels.build_boundary_labels("correct", 2, [])
        self.assertEqual([row["addition_boundary_label"] for row in rows], [False, False, False])

    def test_28_single_addition(self) -> None:
        rows = labels.build_boundary_labels("single", 2, [{"boundary": 1}])
        self.assertEqual([row["addition_event_count"] for row in rows], [0, 1, 0])

    def test_29_multiple_distinct_boundaries(self) -> None:
        rows = labels.build_boundary_labels("multi-distinct", 3, [{"boundary": 1}, {"boundary": 3}])
        self.assertEqual([row["addition_boundary_label"] for row in rows], [False, True, False, True])

    def test_30_multiple_events_one_boundary(self) -> None:
        rows = labels.build_boundary_labels("multi-shared", 2, [{"boundary": 1}, {"boundary": 1}])
        self.assertEqual(sum(row["addition_boundary_label"] for row in rows), 1)
        self.assertEqual(rows[1]["addition_event_count"], 2)

    def test_31_mixed_substitution_does_not_create_positive(self) -> None:
        rows = labels.build_boundary_labels("mixed-sub", 2, [], {"substitution": 1, "addition": 0})
        self.assertFalse(any(row["addition_boundary_label"] for row in rows))

    def test_32_mixed_deletion_addition_uses_event_only(self) -> None:
        rows = labels.build_boundary_labels(
            "mixed-del-add", 2, [{"boundary": 2}], {"deletion": 1, "addition": 1}
        )
        self.assertEqual([row["addition_boundary_label"] for row in rows], [False, False, True])

    def test_33_synthetic_event_coverage_semantics(self) -> None:
        result = labels.synthetic_event_window_coverage(
            [{"boundary": 1}, {"boundary": 1}, {"boundary": 2}], {1}
        )
        self.assertEqual(result, {"events": 3, "covered_events": 2, "coverage": 2 / 3})


class ContractAndDistinctnessTests(unittest.TestCase):
    def test_34_structural_expectations_unchanged(self) -> None:
        contract = json.loads((EXPERIMENT_DIR / "r6_0_population_contract.json").read_text(encoding="utf-8"))
        for key, value in EXPECTED_STRUCTURE.items():
            self.assertEqual(contract[key], value)

    def test_35_future_gates_unchanged(self) -> None:
        contract = json.loads((EXPERIMENT_DIR / "r6_0_next_experiment_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["gates"]["F1"]["threshold"], 0.99)
        self.assertEqual(contract["gates"]["F2"]["threshold"], 0.65)
        self.assertEqual(contract["gates"]["F3"]["threshold"], 0.6)
        self.assertEqual(contract["gates"]["F4"]["threshold"], 9)
        self.assertEqual(contract["execution"]["primary_score"], "MEAN_UNEXPECTED_PHONE_MASS")

    def test_36_no_r5_scoring_dependency(self) -> None:
        source_paths = [
            EXPERIMENT_DIR / "r6_0_boundary_mapping.py",
            EXPERIMENT_DIR / "r6_0_local_features.py",
            EXPERIMENT_DIR / "r6_0_boundary_labels.py",
        ]
        forbidden_names = {"CTCLoss", "best_insert", "enumerate_insert", "insert_target_score"}
        for path in source_paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            self.assertFalse(any(name.startswith("r5_") for name in imported))
            identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            self.assertFalse(forbidden_names & (identifiers | attributes))


def flatten_tests(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    flattened: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            flattened.extend(flatten_tests(item))
        else:
            flattened.append(item)
    return flattened


def deterministic_evidence() -> dict[str, object]:
    logits, _ = synthetic_logits()
    feature_result = features.compute_local_evidence(logits, [0, 1, 2], [1, 2], 1)
    label_rows = labels.build_boundary_labels(
        "synthetic", 3, [{"boundary": 1}, {"boundary": 3}, {"boundary": 3}],
        {"substitution": 1, "deletion": 1},
    )
    return {
        "selected_indices": {
            "exact_center": mapping.nearest_output_index(10.065, 10.0, 8, 10.2),
            "lower": mapping.nearest_output_index(10.061, 10.0, 8, 10.2),
            "upper": mapping.nearest_output_index(10.081, 10.0, 8, 10.2),
            "tie": mapping.nearest_output_index(10.075, 10.0, 8, 10.2),
        },
        "windows": {str(index): mapping.five_step_window(index, 6) for index in (0, 1, 3, 4, 5)},
        "feature_values": feature_result,
        "label_event_counts": [row["addition_event_count"] for row in label_rows],
        "label_values": [row["addition_boundary_label"] for row in label_rows],
    }


def run_suite(summary_path: Path) -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    tests = flatten_tests(suite)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    failures = sorted(test.id() for test, _ in result.failures)
    errors = sorted(test.id() for test, _ in result.errors)
    summary = {
        "stage": "R6-0 synthetic static verification",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "test_count": result.testsRun,
        "passed": result.testsRun - len(failures) - len(errors),
        "failures": failures,
        "errors": errors,
        "test_ids": sorted(test.id() for test in tests),
        "deterministic_evidence": deterministic_evidence(),
        "checkpoint_loaded": False,
        "checkpoint_inference": False,
        "real_train_performance": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    return run_suite(args.summary)


if __name__ == "__main__":
    raise SystemExit(main())

