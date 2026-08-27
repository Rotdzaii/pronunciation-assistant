"""Deterministic synthetic regression suite for the R5-2B-TC1 guard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch

import r5_2b_tc1_numeric_guard as guard


EXPECTED_SCORER_SHA = "2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3"
EXPECTED_FAILED_MANIFEST_SHA = "92163F8DE3ECEAFEE2950AF57302AA565C0F3612A2D6F0D36462552FA66BA09E"
EXPECTED_STATIC_SHA = "F0E54A0A4DBC1378F424E6C8FE8B0A9861CDD72DCBE72C045B96660324723197"
EXPECTED_TC0_RAW_SHA = "4AA9BA6E103425091438E00AD11F25724CE0E34F7E29EBD44B6156458D768F72"

OBSERVATIONS: dict[str, object] = {}
CONTEXT: dict[str, object] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_equal(actual, expected, message: str = "") -> None:
    if actual != expected:
        raise AssertionError(message or f"Expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, tolerance: float = 1e-15) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected!r}, got {actual!r}")


def upward_decimal(value: Decimal) -> float:
    result = float(value)
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def independent_reference(n: int, abs_sum: float, dtype: str) -> tuple[float, float]:
    epsilon = {"float32": Decimal(1) / Decimal(2**23), "float64": Decimal(1) / Decimal(2**52)}[dtype]
    with localcontext() as context:
        context.prec = 100
        u = epsilon / Decimal(2)
        product = Decimal(n - 1) * u
        gamma = Decimal(0) if n == 1 else product / (Decimal(1) - product)
        bound = gamma * Decimal.from_float(abs_sum)
    return upward_decimal(gamma), upward_decimal(bound)


def record_bound(case: str, dtype: str, terms: list[float], backend: str) -> guard.ReductionBound:
    result = guard.reduction_bound_from_terms(terms, dtype, backend=backend)
    expected_gamma, expected_bound = independent_reference(
        result.term_count, result.operand_abs_sum, dtype
    )
    assert_equal(result.machine_epsilon, float(np.finfo(dtype).eps))
    assert_equal(result.unit_roundoff, result.machine_epsilon / 2.0)
    assert_equal(result.gamma, expected_gamma)
    assert_equal(result.bound, expected_bound)
    OBSERVATIONS.setdefault("bounds", {})[case] = {
        "dtype": result.dtype,
        "backend": result.backend,
        "eps": result.machine_epsilon,
        "u": result.unit_roundoff,
        "n": result.term_count,
        "gamma": result.gamma,
        "operand_abs_sum": result.operand_abs_sum,
        "per_reduction_bound": result.bound,
    }
    return result


def fixture_terms(dtype: torch.dtype) -> tuple[torch.Tensor, list[float]]:
    steps = 4
    logits = (
        (torch.arange(steps * 41, dtype=dtype).reshape(steps, 41).remainder(17) - 8.0)
        / 7.0
    )
    log_probs = torch.log_softmax(logits, dim=-1)
    terms = [float(value) for value in log_probs[:, 40].tolist()]
    return logits, terms


def frozen_case(device: str, dtype: str) -> dict[str, object]:
    raw = CONTEXT["tc0_raw"]
    return next(item for item in raw["cases"] if item["device"] == device and item["dtype"] == dtype)


def test_float32_n1_bound() -> None:
    result = record_bound("float32_n1", "float32", [3.25], "cpu")
    assert_equal(result.gamma, 0.0)
    assert_equal(result.bound, 0.0)


def test_float32_n2_bound() -> None:
    record_bound("float32_n2", "float32", [-1.25, 2.5], "cpu")


def test_float32_n4_bound() -> None:
    record_bound("float32_n4", "float32", [-4.0, 3.0, -2.0, 1.0], "cpu")


def test_float32_larger_n_bound() -> None:
    terms = [float((index % 11) - 5) / 7.0 for index in range(128)]
    record_bound("float32_n128", "float32", terms, "cpu")


def test_float64_n1_bound() -> None:
    result = record_bound("float64_n1", "float64", [-7.5], "cpu")
    assert_equal(result.gamma, 0.0)
    assert_equal(result.bound, 0.0)


def test_float64_n4_bound() -> None:
    record_bound("float64_n4", "float64", [1.0, -2.0, 3.0, -4.0], "cpu")


def test_mixed_backend_same_dtype() -> None:
    terms = [1.0, -2.0, 3.0, -4.0]
    a = record_bound("mixed_backend_a", "float32", terms, "cuda")
    b = record_bound("mixed_backend_b", "float32", terms, "cpu")
    combined = guard.combined_reduction_bound(a, b)
    assert combined >= a.bound + b.bound
    OBSERVATIONS["mixed_backend_same_dtype"] = {
        "dtype": "float32",
        "authoritative_backend": "cuda",
        "diagnostic_backend": "cpu",
        "combined_bound": combined,
    }


def test_invalid_gamma_domain_rejected() -> None:
    invalid_n = 2**24 + 1
    try:
        guard.forward_error_bound(invalid_n, 1.0, "float32")
    except guard.InvalidGammaDomainError:
        OBSERVATIONS["invalid_gamma_domain"] = {"n": invalid_n, "status": "REJECTED"}
    else:
        raise AssertionError("Invalid gamma domain was accepted")


def test_historical_tc0_float32() -> None:
    scorer = CONTEXT["scorer"]
    logits, cpu_terms = fixture_terms(torch.float32)
    cpu_result = scorer.score_target(logits, [], 4).target_score
    cpu_frozen = frozen_case("cpu", "float32")
    cuda_frozen = frozen_case("cuda", "float32")
    assert_equal(cpu_result, -15.667130470275879)
    assert_equal(cpu_result, cpu_frozen["paths"]["frozen_scorer"])
    assert_equal(cpu_terms, cpu_frozen["blank_log_prob_values"])
    cuda_value = float(cuda_frozen["paths"]["frozen_scorer"])
    cuda_terms = [float(value) for value in cuda_frozen["blank_log_prob_values"]]
    result = guard.guard_empty_target_equivalence(
        cuda_value,
        cpu_result,
        cuda_terms,
        cpu_terms,
        authoritative_dtype="float32",
        diagnostic_dtype="float32",
        authoritative_backend="cuda",
        diagnostic_backend="cpu",
    )
    assert_equal(result.absolute_difference, 9.5367431640625e-7)
    assert_close(result.correction_bound, 5.603003224082621e-6, tolerance=2e-15)
    OBSERVATIONS["historical_tc0_float32"] = {
        "cpu_value_fresh": cpu_result,
        "cuda_value_frozen_verified_tc0": cuda_value,
        "absolute_difference": result.absolute_difference,
        "combined_bound": result.correction_bound,
        "classification": "EQUIVALENT",
        "cuda_runtime_available": torch.cuda.is_available(),
        "cuda_value_source": "verified_frozen_tc0_artifact",
    }


def test_original_static_fixture() -> None:
    path = CONTEXT["original_static_rerun"]
    assert_equal(sha256(path), EXPECTED_STATIC_SHA)
    payload = json.loads(path.read_text(encoding="utf-8"))
    check = payload["observations"]["empty_target_crosscheck"]
    assert_equal(check["explicit_all_blank_score"], -19.43882697170362)
    assert_equal(check["framework_empty_ctc_score"], -19.43882697170362)
    assert_equal(check["absolute_delta"], 0.0)
    OBSERVATIONS["original_static_fixture"] = {
        "sha256": sha256(path),
        "explicit": check["explicit_all_blank_score"],
        "framework": check["framework_empty_ctc_score"],
        "difference": check["absolute_delta"],
        "classification": "PASS",
    }


def test_float64_control() -> None:
    scorer = CONTEXT["scorer"]
    logits, cpu_terms = fixture_terms(torch.float64)
    cpu_value = scorer.score_target(logits, [], 4).target_score
    cuda_frozen = frozen_case("cuda", "float64")
    cuda_value = float(cuda_frozen["paths"]["frozen_scorer"])
    cuda_terms = [float(value) for value in cuda_frozen["blank_log_prob_values"]]
    result = guard.guard_empty_target_equivalence(
        cuda_value,
        cpu_value,
        cuda_terms,
        cpu_terms,
        authoritative_dtype="float64",
        diagnostic_dtype="float64",
        authoritative_backend="cuda",
        diagnostic_backend="cpu",
    )
    float32_bound = OBSERVATIONS["historical_tc0_float32"]["combined_bound"]
    assert_equal(cpu_value, -15.66712968974069)
    assert_equal(cuda_value, -15.66712968974069)
    assert result.correction_bound < float32_bound
    assert_equal(result.scientific_score, cuda_value)
    OBSERVATIONS["float64_control"] = {
        "cpu_value_fresh": cpu_value,
        "cuda_value_frozen_verified_tc0": cuda_value,
        "absolute_difference": result.absolute_difference,
        "combined_bound": result.correction_bound,
        "smaller_than_float32": True,
        "classification": "PASS",
    }


def standard_terms() -> list[float]:
    return [-4.0, -3.0, -2.0, -1.0]


def standard_bound() -> float:
    item = guard.reduction_bound_from_terms(standard_terms(), "float32", backend="cpu")
    return guard.combined_reduction_bound(item, item)


def compare(authoritative: float, diagnostic: float) -> guard.EmptyTargetGuardResult:
    terms = standard_terms()
    return guard.guard_empty_target_equivalence(
        authoritative,
        diagnostic,
        terms,
        terms,
        authoritative_dtype="float32",
        diagnostic_dtype="float32",
        authoritative_backend="cuda",
        diagnostic_backend="cpu",
    )


def test_exact_equality() -> None:
    result = compare(-10.0, -10.0)
    assert_equal(result.absolute_difference, 0.0)
    assert result.equivalent
    OBSERVATIONS["exact_equality"] = {"difference": 0.0, "bound": result.correction_bound, "status": "PASS"}


def test_within_bound() -> None:
    bound = standard_bound()
    delta = bound / 2.0
    result = compare(0.0, delta)
    assert result.absolute_difference < result.correction_bound
    OBSERVATIONS["within_bound"] = {"delta": delta, "bound": result.correction_bound, "status": "PASS"}


def test_beyond_bound_rejected() -> None:
    bound = standard_bound()
    delta = bound * 2.0
    try:
        compare(0.0, delta)
    except guard.EmptyTargetMismatchError as error:
        assert_equal(error.status, guard.MISMATCH_STATUS)
        OBSERVATIONS["beyond_bound"] = {"delta": delta, "bound": bound, "status": error.status}
    else:
        raise AssertionError("Beyond-bound perturbation was accepted")


def test_boundary_equality_accepted() -> None:
    bound = standard_bound()
    result = compare(0.0, bound)
    assert_equal(result.absolute_difference, result.correction_bound)
    OBSERVATIONS["boundary_equality"] = {"delta": bound, "bound": result.correction_bound, "operator": "<=", "status": "PASS"}


def assert_nonfinite_rejected(authoritative: float, diagnostic: float, name: str) -> None:
    try:
        compare(authoritative, diagnostic)
    except guard.NonFiniteEmptyTargetError as error:
        assert_equal(error.status, guard.NONFINITE_STATUS)
        OBSERVATIONS.setdefault("nonfinite", {})[name] = error.status
    else:
        raise AssertionError(f"Nonfinite case {name} was accepted")


def test_nan_rejected() -> None:
    assert_nonfinite_rejected(float("nan"), -10.0, "nan")


def test_positive_infinity_rejected() -> None:
    assert_nonfinite_rejected(-10.0, float("inf"), "positive_infinity")


def test_negative_infinity_rejected() -> None:
    assert_nonfinite_rejected(float("-inf"), -10.0, "negative_infinity")


def test_authoritative_output_immutable() -> None:
    authoritative = -10.0
    diagnostic = authoritative + standard_bound() / 2.0
    authoritative_bytes = struct.pack("!d", authoritative)
    result = compare(authoritative, diagnostic)
    assert authoritative != diagnostic
    assert_equal(result.scientific_score, authoritative)
    assert_equal(struct.pack("!d", result.scientific_score), authoritative_bytes)
    assert result.scientific_score != diagnostic
    OBSERVATIONS["authoritative_immutability"] = {
        "authoritative": authoritative,
        "diagnostic": diagnostic,
        "returned_scientific_score": result.scientific_score,
        "byte_identical": True,
        "diagnostic_replaced_output": False,
    }


def test_relation_score_immutable() -> None:
    scorer = CONTEXT["scorer"]
    best_insert = scorer.BestCandidateResult(True, "INSERT", None, -2.0, 1, 1, 0)
    best_nonaddition = scorer.BestNonAdditionResult("KEEP", -3.25, None)
    before = scorer.compute_relation_competition_score(best_insert, best_nonaddition)
    before_bytes = struct.pack("!d", before)
    compare(-10.0, -10.0 + standard_bound() / 2.0)
    after = scorer.compute_relation_competition_score(best_insert, best_nonaddition)
    assert_equal(after, before)
    assert_equal(struct.pack("!d", after), before_bytes)
    OBSERVATIONS["relation_score_immutability"] = {
        "before": before,
        "after": after,
        "byte_identical": True,
    }


def test_scorer_sha_unchanged() -> None:
    actual = sha256(CONTEXT["scorer_path"])
    assert_equal(actual, EXPECTED_SCORER_SHA)
    OBSERVATIONS["scorer_sha"] = actual


def test_failed_execution_preserved() -> None:
    actual = sha256(CONTEXT["failed_manifest"])
    assert_equal(actual, EXPECTED_FAILED_MANIFEST_SHA)
    OBSERVATIONS["failed_execution_manifest_sha"] = actual


TESTS = [
    test_float32_n1_bound,
    test_float32_n2_bound,
    test_float32_n4_bound,
    test_float32_larger_n_bound,
    test_float64_n1_bound,
    test_float64_n4_bound,
    test_mixed_backend_same_dtype,
    test_invalid_gamma_domain_rejected,
    test_historical_tc0_float32,
    test_original_static_fixture,
    test_float64_control,
    test_exact_equality,
    test_within_bound,
    test_beyond_bound_rejected,
    test_boundary_equality_accepted,
    test_nan_rejected,
    test_positive_infinity_rejected,
    test_negative_infinity_rejected,
    test_authoritative_output_immutable,
    test_relation_score_immutable,
    test_scorer_sha_unchanged,
    test_failed_execution_preserved,
]


def run(args: argparse.Namespace) -> int:
    OBSERVATIONS.clear()
    scorer_path = args.scorer.resolve()
    assert_equal(sha256(scorer_path), EXPECTED_SCORER_SHA)
    assert_equal(sha256(args.tc0_raw), EXPECTED_TC0_RAW_SHA)
    CONTEXT.clear()
    CONTEXT.update(
        scorer=load_module("frozen_r5_2b_scorer_tc1", scorer_path),
        scorer_path=scorer_path,
        failed_manifest=args.failed_manifest.resolve(),
        tc0_raw=json.loads(args.tc0_raw.read_text(encoding="utf-8")),
        original_static_rerun=args.original_static_rerun.resolve(),
    )
    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for test in TESTS:
        try:
            test()
        except Exception as error:
            results.append({"name": test.__name__, "status": "FAIL"})
            failures.append({"name": test.__name__, "error": f"{type(error).__name__}: {error}"})
        else:
            results.append({"name": test.__name__, "status": "PASS"})
    observations_text = json.dumps(OBSERVATIONS, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload = {
        "suite": "R5-2B-TC1 synthetic static regression",
        "synthetic_only": True,
        "test_count": len(TESTS),
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
        "observations": OBSERVATIONS,
        "deterministic_observations_sha256": hashlib.sha256(observations_text.encode("utf-8")).hexdigest().upper(),
        "protocol": {
            "train_audio_accessed": False,
            "checkpoint_inference_run": False,
            "train_performance_calculated": False,
            "validation_accessed": False,
            "test_accessed": False,
        },
    }
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--failed-manifest", type=Path, required=True)
    parser.add_argument("--tc0-raw", type=Path, required=True)
    parser.add_argument("--original-static-rerun", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
