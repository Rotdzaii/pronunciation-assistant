from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from r5_1a_scorer import (
    BestInsertResult,
    HypothesisScore,
    InsertCandidateScore,
    KeepAlignabilityError,
    adjacent_repeat_count,
    deserialize_extended_score,
    enumerate_insert_targets,
    extended_real_roc_auc,
    is_alignable,
    minimum_ctc_steps,
    predict_addition,
    score_hypothesis,
    score_word,
    select_best_insert,
    serialize_extended_score,
    threshold_candidates,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def synthetic_logits(steps: int) -> torch.Tensor:
    base = torch.arange(steps * 41, dtype=torch.float32).reshape(steps, 41)
    return (base.remainder(17) - 8.0) / 7.0


def candidate(boundary: int, phone: int, score: float, alignable: bool = True) -> InsertCandidateScore:
    hypothesis = HypothesisScore(
        target=(phone,), input_length=1, minimum_steps=1,
        alignable=alignable, raw_score=score, target_score=score,
        ctc_called=alignable,
    )
    return InsertCandidateScore(boundary=boundary, phone_index=phone, hypothesis=hypothesis)


def test_a() -> dict[str, object]:
    target = [1, 2, 3]
    require(adjacent_repeat_count(target) == 0, "repeat count")
    require(minimum_ctc_steps(target) == 3, "minimum steps")
    require(is_alignable(target, 3), "T=3")
    require(not is_alignable(target, 2), "T=2")
    return {"adjacent_repeats": 0, "minimum_steps": 3}


def test_b() -> dict[str, object]:
    target = [1, 1, 3]
    require(adjacent_repeat_count(target) == 1, "repeat count")
    require(minimum_ctc_steps(target) == 4, "minimum steps")
    require(is_alignable(target, 4), "T=4")
    require(not is_alignable(target, 3), "T=3")
    return {"adjacent_repeats": 1, "minimum_steps": 4}


def test_c() -> dict[str, object]:
    require(adjacent_repeat_count([1, 1, 2, 2]) == 2, "repeat count")
    require(minimum_ctc_steps([1, 1, 2, 2]) == 6, "minimum steps")
    return {"adjacent_repeats": 2, "minimum_steps": 6}


def test_d() -> dict[str, object]:
    require(adjacent_repeat_count([5, 5, 5]) == 2, "repeat count")
    require(minimum_ctc_steps([5, 5, 5]) == 5, "minimum steps")
    return {"adjacent_repeats": 2, "minimum_steps": 5}


def test_e() -> dict[str, object]:
    result = score_hypothesis(synthetic_logits(2), [1, 1])
    require(not result.alignable, "must be impossible")
    require(result.raw_score == float("-inf"), "raw score")
    require(result.target_score == float("-inf"), "target score")
    require(not result.ctc_called, "guard must precede CTCLoss")
    return {"raw_is_neg_inf": True, "target_is_neg_inf": True, "ctc_called": False}


def test_f() -> dict[str, object]:
    result = score_hypothesis(synthetic_logits(3), [1, 2])
    require(result.alignable and result.ctc_called, "alignable target must call CTCLoss")
    require(math.isfinite(result.raw_score) and math.isfinite(result.target_score), "finite scores")
    require(result.target_score == result.raw_score / 2.0, "TARGET normalization")
    return {"finite": True, "normalization_exact": True}


def test_g() -> dict[str, object]:
    result = select_best_insert([candidate(0, 3, -3.0), candidate(2, 8, -1.0), candidate(1, 4, -2.0)])
    require(result == BestInsertResult(True, 2, 8, -1.0, 3, 3, 0), "unique maximum")
    return {"boundary": 2, "phone": 8, "score": -1.0}


def test_h() -> dict[str, object]:
    result = select_best_insert([candidate(2, 9, -1.0), candidate(1, 12, -1.0)])
    require((result.boundary, result.phone_index) == (1, 12), "lower boundary tie")
    return {"boundary": 1, "phone": 12}


def test_i() -> dict[str, object]:
    result = select_best_insert([candidate(2, 9, -1.0), candidate(2, 4, -1.0)])
    require((result.boundary, result.phone_index) == (2, 4), "lower phone tie")
    return {"boundary": 2, "phone": 4}


def test_j() -> dict[str, object]:
    result = select_best_insert([
        candidate(0, 0, float("-inf"), alignable=False),
        candidate(1, 7, -10.0, alignable=True),
    ])
    require((result.boundary, result.phone_index) == (1, 7), "impossible candidate cannot win")
    require(result.impossible_candidates == 1, "impossible count")
    return {"finite_winner": True, "impossible_candidates": 1}


def test_k() -> dict[str, object]:
    result = score_word(synthetic_logits(1), [1])
    require(result.keep.alignable, "KEEP alignable")
    require(not result.best_insert.best_insert_exists, "no BEST_INSERT")
    require(result.best_insert.boundary is None and result.best_insert.phone_index is None, "null identity")
    require(result.best_insert.best_target_score == float("-inf"), "best score")
    require(result.addition_score == float("-inf"), "A")
    require(result.best_insert.total_candidates == 80, "all candidates retained")
    require(result.best_insert.impossible_candidates == 80, "all impossible")
    return {"best_insert_exists": False, "candidate_count": 80, "addition_score_is_neg_inf": True}


def test_l() -> dict[str, object]:
    try:
        score_word(synthetic_logits(2), [1, 1])
    except KeepAlignabilityError as error:
        require(error.status == "R5_1A_EXECUTION_BLOCKED_KEEP_ALIGNABILITY", "status")
        return {"raised": error.status}
    raise AssertionError("KEEP impossibility did not stop")


def test_zero_infinity_regression() -> dict[str, object]:
    logits = synthetic_logits(2)
    log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(1)
    target = torch.tensor([1, 1], dtype=torch.long)
    loss = torch.nn.CTCLoss(blank=40, reduction="none", zero_infinity=True)(
        log_probs, target, torch.tensor([2]), torch.tensor([2])
    )
    underlying = float(loss.item())
    require(underlying == 0.0, "fixture must reproduce misleading zero loss")
    guarded = score_hypothesis(logits, [1, 1])
    require(not guarded.ctc_called and guarded.raw_score == float("-inf"), "guard interception")
    return {"underlying_zero_infinity_loss": underlying, "guarded_ctc_called": False, "guarded_score_is_neg_inf": True}


def test_extended_real_auc() -> dict[str, object]:
    finite_labels = [0, 1, 0, 1, 0, 1]
    finite_scores = [0.1, 0.4, 0.2, 0.3, 0.3, 0.3]
    ours = extended_real_roc_auc(finite_labels, finite_scores)
    trusted = float(roc_auc_score(finite_labels, finite_scores))
    require(math.isclose(ours, trusted, rel_tol=0.0, abs_tol=1e-15), "finite cross-check")
    one_negative_infinity = extended_real_roc_auc([0, 1, 1], [float("-inf"), 0.0, 1.0])
    require(one_negative_infinity == 1.0, "one negative infinity")
    tied_negative_infinity = extended_real_roc_auc(
        [0, 1, 0, 1], [float("-inf"), float("-inf"), 0.0, 1.0]
    )
    require(tied_negative_infinity == 0.625, "tied negative infinity")
    finite_ties = extended_real_roc_auc([0, 1, 0, 1], [0.0, 0.0, 1.0, 1.0])
    require(finite_ties == 0.5, "finite mixed-class ties")
    mixed_ties = extended_real_roc_auc(
        [1, 0, 1, 0, 1, 0], [float("-inf"), float("-inf"), 0.0, 0.0, 1.0, 1.0]
    )
    require(mixed_ties == 0.5, "mixed positive/negative ties")
    return {
        "finite_crosscheck": ours, "trusted_standard": trusted,
        "one_neg_inf": one_negative_infinity, "tied_neg_inf": tied_negative_infinity,
        "finite_ties": finite_ties, "mixed_class_ties": mixed_ties,
    }


def test_threshold_candidates() -> dict[str, object]:
    scores = np.asarray([float("-inf"), 0.1, 0.1, 0.3], dtype=np.float64)
    candidates = threshold_candidates(scores)
    require(candidates.size == 4, "edge plus unique count")
    require(np.isfinite(candidates).all(), "negative infinity excluded")
    require(candidates[0] == np.nextafter(np.float64(0.1), -np.inf), "lower edge")
    require(candidates[-1] == np.nextafter(np.float64(0.3), np.inf), "upper edge")
    prediction = predict_addition(scores, 0.1)
    require(prediction.tolist() == [False, True, True, True], "A>=theta")
    return {"candidate_count": 4, "negative_infinity_excluded": True, "prediction_rule": "A>=theta"}


def test_serialization() -> dict[str, object]:
    negative_payload = serialize_extended_score(float("-inf"))
    require(negative_payload == {"score_value": None, "score_is_neg_inf": True}, "negative representation")
    negative_json = json.dumps(negative_payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    negative_roundtrip = deserialize_extended_score(json.loads(negative_json))
    require(negative_roundtrip == float("-inf"), "negative round trip")
    finite = float(np.float64(0.16184102947061696))
    finite_payload = serialize_extended_score(finite)
    finite_json = json.dumps(finite_payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    finite_roundtrip = deserialize_extended_score(json.loads(finite_json))
    require(finite_roundtrip == finite, "finite exact round trip")
    require("Infinity" not in negative_json, "standard JSON")
    return {
        "negative_payload": negative_payload, "negative_roundtrip_is_neg_inf": True,
        "finite_value": finite, "finite_roundtrip_exact": True,
        "standards_compatible_json": True,
    }


def test_candidate_enumeration() -> dict[str, object]:
    candidates = list(enumerate_insert_targets([1, 2, 3]))
    require(len(candidates) == 160, "40*(N+1)")
    require(candidates[0] == (0, 0, (0, 1, 2, 3)), "first candidate")
    require(candidates[-1] == (3, 39, (1, 2, 3, 39)), "last candidate")
    return {"candidate_count": 160, "order": "boundary_then_phone"}


TESTS: list[tuple[str, Callable[[], dict[str, object]]]] = [
    ("A", test_a), ("B", test_b), ("C", test_c), ("D", test_d),
    ("E", test_e), ("F", test_f), ("G", test_g), ("H", test_h),
    ("I", test_i), ("J", test_j), ("K", test_k), ("L", test_l),
    ("ZERO_INFINITY_REGRESSION", test_zero_infinity_regression),
    ("EXTENDED_REAL_ROC_AUC", test_extended_real_auc),
    ("THRESHOLD_CANDIDATES", test_threshold_candidates),
    ("SERIALIZATION_ROUNDTRIP", test_serialization),
    ("CANDIDATE_ENUMERATION", test_candidate_enumeration),
]


def run_suite() -> dict[str, object]:
    results: dict[str, object] = {}
    failed = []
    for name, function in TESTS:
        try:
            details = function()
            results[name] = {"status": "PASS", "details": details}
        except Exception as error:  # test report must preserve exact failure
            results[name] = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
            failed.append(name)
    return {
        "status": "PASS" if not failed else "FAIL",
        "test_count": len(TESTS),
        "passed": len(TESTS) - len(failed),
        "failed": len(failed),
        "failed_tests": failed,
        "tests": results,
        "synthetic_only": True,
        "real_checkpoint_loaded": False,
        "audio_accessed": False,
        "phoenix_performance_metrics_calculated": False,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_suite()
    content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": result["status"], "passed": result["passed"], "failed": result["failed"],
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest().upper(),
    }, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
