"""Synthetic-only static verification for the frozen R5-2B scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import r5_2b_scorer as s


EMPTY_TOLERANCE = 1e-10
OBSERVATIONS: dict[str, object] = {}


def assert_equal(actual, expected, message: str = "") -> None:
    if actual != expected:
        raise AssertionError(message or f"{actual!r} != {expected!r}")


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r} within {tolerance}")


def logits(time_steps: int) -> torch.Tensor:
    values = torch.arange(time_steps * 41, dtype=torch.float64).reshape(time_steps, 41)
    return ((values.remainder(23) - 11.0) / 9.0).contiguous()


def hypothesis(target, score: float, alignable: bool = True) -> s.HypothesisScore:
    values = tuple(target)
    scalar = float(score) if alignable else float("-inf")
    return s.HypothesisScore(
        target=values,
        input_length=max(s.minimum_ctc_steps(values), 1),
        minimum_steps=s.minimum_ctc_steps(values),
        alignable=alignable,
        raw_score=scalar * max(len(values), 1) if math.isfinite(scalar) else scalar,
        target_score=scalar,
        ctc_called=alignable and bool(values),
        empty_formula_used=alignable and not values,
    )


def scored(candidate: s.TargetCandidate, score: float, alignable: bool = True) -> s.ScoredCandidate:
    return s.ScoredCandidate(candidate, hypothesis(candidate.target, score, alignable))


def insert(boundary: int, phone: int, score: float, alignable: bool = True) -> s.ScoredCandidate:
    candidate = s.TargetCandidate("INSERT", (1, 2), boundary=boundary, phone_index=phone)
    return scored(candidate, score, alignable)


def sub(position: int, phone: int, score: float, alignable: bool = True) -> s.ScoredCandidate:
    candidate = s.TargetCandidate("SUB", (1, 2), position=position, phone_index=phone)
    return scored(candidate, score, alignable)


def delete(position: int, score: float, target=(1,), alignable: bool = True) -> s.ScoredCandidate:
    candidate = s.TargetCandidate("DELETE", tuple(target), position=position)
    return scored(candidate, score, alignable)


def keep(score: float, alignable: bool = True) -> s.ScoredCandidate:
    candidate = s.TargetCandidate("KEEP", (1, 2))
    return scored(candidate, score, alignable)


def absent_best(family: str) -> s.BestCandidateResult:
    selector = {
        "INSERT": s.select_best_insert,
        "SUB": s.select_best_sub,
        "DELETE": s.select_best_delete,
    }[family]
    return selector([])


def make_word_score(insert_score: float, keep_score: float, sub_score: float, delete_score: float):
    best_insert = s.select_best_insert([insert(2, 7, insert_score)])
    best_sub = s.select_best_sub([sub(0, 8, sub_score)])
    best_delete = s.select_best_delete([delete(0, delete_score)])
    keep_item = keep(keep_score)
    best_nonaddition = s.select_best_nonaddition(keep_item, best_sub, best_delete)
    c_value = s.compute_relation_competition_score(best_insert, best_nonaddition)
    return s.WordScore(
        keep_item, best_insert, best_sub, best_delete, best_nonaddition, c_value
    )


def test_keep_constructor() -> None:
    item = s.construct_keep([1, 2, 3])
    assert_equal(item.family, "KEEP")
    assert_equal(item.target, (1, 2, 3))


def test_insert_constructor_and_count() -> None:
    items = s.enumerate_insert_candidates([5])
    assert_equal(len(items), 80)
    assert_equal(items[0], s.TargetCandidate("INSERT", (0, 5), boundary=0, phone_index=0))
    assert_equal(items[-1], s.TargetCandidate("INSERT", (5, 39), boundary=1, phone_index=39))


def test_sub_constructor_and_count() -> None:
    expected = [3, 7]
    items = s.enumerate_sub_candidates(expected)
    assert_equal(len(items), 78)
    for item in items:
        if item.target[item.position] == expected[item.position]:
            raise AssertionError("Original expected phone appeared as SUB candidate")


def test_delete_constructor_and_count() -> None:
    items = s.enumerate_delete_candidates([1, 2, 3])
    assert_equal(len(items), 3)
    assert_equal([item.target for item in items], [(2, 3), (1, 3), (1, 2)])


def test_identical_delete_candidates_retained() -> None:
    items = s.enumerate_delete_candidates([5, 5])
    assert_equal(len(items), 2)
    assert_equal(items[0].target, items[1].target)
    assert_equal([item.position for item in items], [0, 1])


def _candidate_total_test(length: int, expected: int) -> None:
    counts = s.candidate_counts(length)
    assert_equal(counts["TOTAL"], expected)
    assert_equal(counts["TOTAL"], 80 * length + 41)


def test_total_n1() -> None:
    _candidate_total_test(1, 121)


def test_total_n2() -> None:
    _candidate_total_test(2, 201)


def test_total_n3() -> None:
    _candidate_total_test(3, 281)


def test_total_n5() -> None:
    _candidate_total_test(5, 441)


def test_alignability_no_repeat() -> None:
    assert_equal(s.minimum_ctc_steps([1, 2, 3]), 3)
    assert s.is_alignable([1, 2, 3], 3)
    assert not s.is_alignable([1, 2, 3], 2)


def test_alignability_repeats() -> None:
    assert_equal(s.adjacent_repeat_count([1, 1, 2, 2]), 2)
    assert_equal(s.minimum_ctc_steps([1, 1, 2, 2]), 6)


def test_impossible_nonempty_score() -> None:
    result = s.score_target(logits(2), [1, 1], 2)
    assert not result.alignable
    assert result.raw_score == float("-inf")
    assert result.target_score == float("-inf")
    assert not result.ctc_called


def test_alignable_nonempty_score() -> None:
    result = s.score_target(logits(3), [1, 2], 3)
    assert result.alignable and math.isfinite(result.raw_score)
    assert_close(result.target_score, result.raw_score / 2.0)


def test_empty_formula() -> None:
    tensor = logits(4)
    explicit = float(torch.log_softmax(tensor, dim=-1)[:, 40].sum().item())
    result = s.score_target(tensor, [], 4)
    assert result.alignable and result.empty_formula_used and not result.ctc_called
    assert_close(result.raw_score, explicit)
    assert_close(result.target_score, explicit)


def test_empty_ctc_crosscheck() -> None:
    tensor = logits(5)
    log_probs = torch.log_softmax(tensor, dim=-1)
    explicit = float(log_probs[:, 40].sum().item())
    loss = torch.nn.CTCLoss(blank=40, reduction="none", zero_infinity=True)(
        log_probs.unsqueeze(1),
        torch.empty(0, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
    )
    framework = -float(loss.item())
    delta = abs(explicit - framework)
    OBSERVATIONS["empty_target_crosscheck"] = {
        "explicit_all_blank_score": explicit,
        "framework_empty_ctc_score": framework,
        "absolute_delta": delta,
        "tolerance": EMPTY_TOLERANCE,
    }
    if delta > EMPTY_TOLERANCE:
        raise AssertionError(f"Empty-target mismatch {delta} > {EMPTY_TOLERANCE}")


def test_one_phone_word() -> None:
    expected = [5]
    result = s.score_word(logits(2), expected, 2)
    counts = s.candidate_counts(1)
    assert_equal(counts, {"KEEP": 1, "INSERT": 80, "SUB": 39, "DELETE": 1, "TOTAL": 121})
    assert result.best_delete.exists
    assert_equal(result.best_delete.candidate.target, ())
    assert math.isfinite(result.best_delete.score)
    OBSERVATIONS["one_phone"] = {
        "counts": counts,
        "delete_target": [],
        "delete_finite": True,
        "best_nonaddition_family": result.best_nonaddition.family,
    }


def test_best_insert_boundary_tie() -> None:
    result = s.select_best_insert([insert(1, 9, 2.0), insert(0, 12, 2.0)])
    assert_equal(result.candidate.boundary, 0)


def test_best_insert_phone_tie() -> None:
    result = s.select_best_insert([insert(1, 9, 2.0), insert(1, 4, 2.0)])
    assert_equal(result.candidate.phone_index, 4)


def test_best_sub_position_tie() -> None:
    result = s.select_best_sub([sub(1, 3, 2.0), sub(0, 9, 2.0)])
    assert_equal(result.candidate.position, 0)


def test_best_sub_phone_tie() -> None:
    result = s.select_best_sub([sub(0, 8, 2.0), sub(0, 2, 2.0)])
    assert_equal(result.candidate.phone_index, 2)


def test_best_delete_position_tie() -> None:
    result = s.select_best_delete([delete(1, 2.0), delete(0, 2.0)])
    assert_equal(result.candidate.position, 0)


def test_identical_delete_score_tie() -> None:
    candidates = s.enumerate_delete_candidates([5, 5])
    result = s.select_best_delete([scored(candidates[1], 1.25), scored(candidates[0], 1.25)])
    assert_equal(result.candidate.target, (5,))
    assert_equal(result.candidate.position, 0)


def _nonaddition(keep_score: float, sub_score: float, delete_score: float):
    return s.select_best_nonaddition(
        keep(keep_score),
        s.select_best_sub([sub(0, 3, sub_score)]),
        s.select_best_delete([delete(0, delete_score)]),
    )


def test_nonaddition_unique_keep() -> None:
    result = _nonaddition(3.0, 2.0, 1.0)
    assert_equal((result.family, result.score), ("KEEP", 3.0))


def test_nonaddition_unique_sub() -> None:
    result = _nonaddition(1.0, 3.0, 2.0)
    assert_equal((result.family, result.score), ("SUB", 3.0))


def test_nonaddition_unique_delete() -> None:
    result = _nonaddition(1.0, 2.0, 3.0)
    assert_equal((result.family, result.score), ("DELETE", 3.0))


def test_nonaddition_keep_sub_tie() -> None:
    assert_equal(_nonaddition(3.0, 3.0, 1.0).family, "KEEP")


def test_nonaddition_keep_delete_tie() -> None:
    assert_equal(_nonaddition(3.0, 1.0, 3.0).family, "KEEP")


def test_nonaddition_sub_delete_tie() -> None:
    assert_equal(_nonaddition(1.0, 3.0, 3.0).family, "SUB")


def test_c_positive() -> None:
    assert_close(make_word_score(3.0, 1.0, 2.0, 0.0).relation_competition_score, 1.0)


def test_c_zero() -> None:
    assert_close(make_word_score(2.0, 1.0, 2.0, 0.0).relation_competition_score, 0.0)


def test_c_negative() -> None:
    assert_close(make_word_score(2.0, 1.0, 3.0, 0.0).relation_competition_score, -1.0)


def test_all_inserts_impossible() -> None:
    result = s.score_word(logits(2), [1, 2], 2)
    assert not result.best_insert.exists
    assert result.best_insert.score == float("-inf")
    assert result.relation_competition_score == float("-inf")
    assert result.best_insert.candidate is None


def test_impossible_sub_cannot_win() -> None:
    result = s.select_best_sub([sub(0, 1, 99.0, alignable=False), sub(1, 2, 2.0)])
    assert result.exists
    assert_equal(result.candidate.position, 1)
    assert_equal(result.impossible_candidates, 1)


def test_keep_impossible_guard() -> None:
    try:
        s.score_word(logits(2), [1, 1], 2)
    except s.KeepAlignabilityError:
        return
    raise AssertionError("KEEP-impossible state was not detected")


def test_mechanism_sub_beats_insert() -> None:
    result = make_word_score(2.0, 1.0, 3.0, 0.0)
    assert 2.0 - 1.0 > 0.0
    assert result.relation_competition_score < 0.0
    assert_equal(result.best_nonaddition.family, "SUB")


def test_mechanism_delete_beats_insert() -> None:
    result = make_word_score(2.0, 1.0, 0.0, 3.0)
    assert 2.0 - 1.0 > 0.0
    assert result.relation_competition_score < 0.0
    assert_equal(result.best_nonaddition.family, "DELETE")


def test_mechanism_insert_beats_all() -> None:
    result = make_word_score(3.0, 1.0, 2.0, 0.0)
    assert result.relation_competition_score > 0.0


def test_event_output_positive() -> None:
    result = make_word_score(3.0, 1.0, 2.0, 0.0)
    output = s.decision_output("word", result, 0.5)
    assert_equal(output["prediction"], "ADDITION")
    assert_equal(output["addition_event"], {"word_id": "word", "phone_index": 7, "boundary": 2})
    assert output["diagnostic_is_confirmed_multiclass_relation"] is False


def test_event_output_nonaddition() -> None:
    result = make_word_score(2.0, 1.0, 3.0, 0.0)
    output = s.decision_output("word", result, 0.0)
    assert_equal(output["prediction"], "NON_ADDITION")
    assert output["addition_event"] is None
    assert_equal(output["diagnostic_best_nonaddition_family"], "SUB")
    assert output["diagnostic_is_confirmed_multiclass_relation"] is False


def test_extended_real_round_trip() -> None:
    for family in ("KEEP", "INSERT", "SUB", "DELETE", "BEST_NON_ADDITION", "C"):
        for value in (1.25, float("-inf")):
            payload = s.serialize_extended_score(value)
            encoded = json.dumps({family: payload}, allow_nan=False, sort_keys=True)
            if "Infinity" in encoded:
                raise AssertionError("Non-standard Infinity leaked into JSON")
            restored = s.deserialize_extended_score(payload)
            assert_equal(restored, value)
    OBSERVATIONS["serialization"] = {
        "families": ["KEEP", "INSERT", "SUB", "DELETE", "BEST_NON_ADDITION", "C"],
        "negative_infinity_representation": {"score_is_neg_inf": True, "score_value": None},
    }


def test_threshold_candidates() -> None:
    values = s.threshold_candidates([float("-inf"), 2.0, 1.0, 2.0])
    assert_equal(values.size, 4)
    assert_equal(values[1:-1].tolist(), [1.0, 2.0])
    assert_equal(float(values[0]), float(np.nextafter(np.float64(1.0), -np.inf)))
    assert_equal(float(values[-1]), float(np.nextafter(np.float64(2.0), np.inf)))


def test_threshold_prediction_equality() -> None:
    predictions = s.predict_addition([float("-inf"), 0.5, 1.0], 0.5)
    assert_equal(predictions.tolist(), [False, True, True])


def test_threshold_tie_priority() -> None:
    base = {"binary_macro_f1": 0.6, "addition_f1": 0.2, "correct_only_far": 0.1, "threshold": 1.0}
    higher_macro = {**base, "binary_macro_f1": 0.7, "addition_f1": 0.0}
    assert s.select_threshold_record([base, higher_macro]) is higher_macro
    higher_f1 = {**base, "addition_f1": 0.3, "correct_only_far": 0.9}
    assert s.select_threshold_record([base, higher_f1]) is higher_f1
    lower_far = {**base, "correct_only_far": 0.05, "threshold": 0.0}
    assert s.select_threshold_record([base, lower_far]) is lower_far
    higher_theta = {**base, "threshold": 2.0}
    assert s.select_threshold_record([base, higher_theta]) is higher_theta


def test_cohort_far_helpers() -> None:
    predictions = [True, False, True, True, False, True]
    masks = {
        "correct_only": [True, True, False, False, False, False],
        "substitution": [False, False, True, True, False, False],
        "deletion": [False, False, False, False, True, True],
    }
    result = s.cohort_false_addition_rates(predictions, masks)
    assert_equal(result["correct_only"], {"support": 2, "false_additions": 1, "rate": 0.5})
    assert_equal(result["substitution"], {"support": 2, "false_additions": 2, "rate": 1.0})
    assert_equal(result["deletion"], {"support": 2, "false_additions": 1, "rate": 0.5})


def test_event_exact_match() -> None:
    truth = [s.AdditionEvent("w", 5, 1)]
    prediction = [s.AdditionEvent("w", 5, 1)]
    result = s.event_metrics(truth, prediction)
    assert_equal((result["tp"], result["fp"], result["fn"]), (1, 0, 0))


def test_event_wrong_boundary() -> None:
    result = s.event_metrics([s.AdditionEvent("w", 5, 1)], [s.AdditionEvent("w", 5, 2)])
    assert_equal((result["tp"], result["fp"], result["fn"]), (0, 1, 1))


def test_event_wrong_phone() -> None:
    result = s.event_metrics([s.AdditionEvent("w", 5, 1)], [s.AdditionEvent("w", 6, 1)])
    assert_equal((result["tp"], result["fp"], result["fn"]), (0, 1, 1))


def test_event_multiple_truth_one_prediction() -> None:
    truth = [s.AdditionEvent("w", 5, 1), s.AdditionEvent("w", 6, 2)]
    prediction = [s.AdditionEvent("w", 5, 1)]
    result = s.event_metrics(truth, prediction)
    assert_equal((result["tp"], result["fp"], result["fn"]), (1, 0, 1))


def test_event_duplicate_prediction_rejected() -> None:
    prediction = [s.AdditionEvent("w", 5, 1), s.AdditionEvent("w", 6, 2)]
    try:
        s.event_metrics([], prediction)
    except ValueError:
        return
    raise AssertionError("Multiple predicted BEST_INSERT events for one word were accepted")


def test_candidate_order_deterministic() -> None:
    expected = [2, 4, 6]
    first = (
        s.enumerate_insert_candidates(expected),
        s.enumerate_sub_candidates(expected),
        s.enumerate_delete_candidates(expected),
    )
    second = (
        s.enumerate_insert_candidates(expected),
        s.enumerate_sub_candidates(expected),
        s.enumerate_delete_candidates(expected),
    )
    assert_equal(first, second)
    fixture = repr(first).encode("utf-8")
    OBSERVATIONS["candidate_order_sha256"] = hashlib.sha256(fixture).hexdigest().upper()


TESTS = [
    test_keep_constructor,
    test_insert_constructor_and_count,
    test_sub_constructor_and_count,
    test_delete_constructor_and_count,
    test_identical_delete_candidates_retained,
    test_total_n1,
    test_total_n2,
    test_total_n3,
    test_total_n5,
    test_alignability_no_repeat,
    test_alignability_repeats,
    test_impossible_nonempty_score,
    test_alignable_nonempty_score,
    test_empty_formula,
    test_empty_ctc_crosscheck,
    test_one_phone_word,
    test_best_insert_boundary_tie,
    test_best_insert_phone_tie,
    test_best_sub_position_tie,
    test_best_sub_phone_tie,
    test_best_delete_position_tie,
    test_identical_delete_score_tie,
    test_nonaddition_unique_keep,
    test_nonaddition_unique_sub,
    test_nonaddition_unique_delete,
    test_nonaddition_keep_sub_tie,
    test_nonaddition_keep_delete_tie,
    test_nonaddition_sub_delete_tie,
    test_c_positive,
    test_c_zero,
    test_c_negative,
    test_all_inserts_impossible,
    test_impossible_sub_cannot_win,
    test_keep_impossible_guard,
    test_mechanism_sub_beats_insert,
    test_mechanism_delete_beats_insert,
    test_mechanism_insert_beats_all,
    test_event_output_positive,
    test_event_output_nonaddition,
    test_extended_real_round_trip,
    test_threshold_candidates,
    test_threshold_prediction_equality,
    test_threshold_tie_priority,
    test_cohort_far_helpers,
    test_event_exact_match,
    test_event_wrong_boundary,
    test_event_wrong_phone,
    test_event_multiple_truth_one_prediction,
    test_event_duplicate_prediction_rejected,
    test_candidate_order_deterministic,
]


def run(summary_path: Path) -> int:
    OBSERVATIONS.clear()
    results = []
    failures = []
    for test in TESTS:
        try:
            test()
        except Exception as error:  # synthetic harness must preserve exact failure text
            results.append({"name": test.__name__, "status": "FAIL"})
            failures.append({"name": test.__name__, "error": f"{type(error).__name__}: {error}"})
        else:
            results.append({"name": test.__name__, "status": "PASS"})
    observations_json = json.dumps(OBSERVATIONS, sort_keys=True, separators=(",", ":"))
    summary = {
        "suite": "R5-2B synthetic static verification",
        "synthetic_only": True,
        "test_count": len(TESTS),
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
        "observations": OBSERVATIONS,
        "deterministic_fixture_sha256": hashlib.sha256(observations_json.encode("utf-8")).hexdigest().upper(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.summary))
