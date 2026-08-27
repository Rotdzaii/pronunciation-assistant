"""R5-2B-TC1 empty-target numerical-equivalence diagnostic guard.

This additive helper implements only the frozen TC1 execution diagnostic.  The
authoritative R5-2B scorer value is returned unchanged as ``scientific_score``.
No diagnostic value can replace or modify that scientific value.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from typing import Iterable


NONFINITE_STATUS = "R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_NONFINITE_EMPTY_TARGET"
MISMATCH_STATUS = "R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET_MISMATCH"

_MACHINE_EPSILON = {
    "float32": 2.0**-23,
    "float64": 2.0**-52,
}


class EmptyTargetDiagnosticError(RuntimeError):
    """Base class for TC1 diagnostic failures."""

    status: str


class NonFiniteEmptyTargetError(EmptyTargetDiagnosticError):
    status = NONFINITE_STATUS


class EmptyTargetMismatchError(EmptyTargetDiagnosticError):
    status = MISMATCH_STATUS


class InvalidGammaDomainError(ValueError):
    """Raised when (n-1)u is outside the valid gamma domain."""


@dataclass(frozen=True)
class ReductionBound:
    dtype: str
    backend: str
    machine_epsilon: float
    unit_roundoff: float
    term_count: int
    gamma: float
    operand_abs_sum: float
    bound: float


@dataclass(frozen=True)
class EmptyTargetGuardResult:
    scientific_score: float
    diagnostic_score: float
    absolute_difference: float
    correction_bound: float
    equivalent: bool
    authoritative: ReductionBound
    diagnostic: ReductionBound

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scientific_score_source"] = "frozen_r5_2b_scorer"
        payload["diagnostic_can_replace_scientific_score"] = False
        return payload


def machine_epsilon(dtype: str) -> float:
    """Return the IEEE-754 machine epsilon frozen by TC1."""

    normalized = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    if normalized not in _MACHINE_EPSILON:
        raise ValueError(f"Unsupported floating-point dtype: {dtype}")
    return _MACHINE_EPSILON[normalized]


def unit_roundoff(dtype: str) -> float:
    """Return u = machine epsilon / 2 (standard unit-roundoff convention)."""

    return machine_epsilon(dtype) / 2.0


def _decimal_to_float_upward(value: Decimal) -> float:
    """Round a nonnegative Decimal to a conservative finite binary64 value."""

    if value < 0 or not value.is_finite():
        raise ValueError("A bound component must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError("Forward-error bound overflowed binary64")
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def exact_operand_abs_sum(terms: Iterable[float]) -> tuple[int, float]:
    """Return count and a conservatively rounded sum of absolute float operands."""

    values = [float(item) for item in terms]
    if not values:
        raise ValueError("At least one reduction term is required")
    if any(not math.isfinite(item) for item in values):
        raise ValueError("Reduction operands must be finite")
    with localcontext() as context:
        context.prec = 100
        exact = sum((abs(Decimal.from_float(item)) for item in values), Decimal(0))
    return len(values), _decimal_to_float_upward(exact)


def forward_error_bound(
    term_count: int,
    operand_abs_sum: float,
    dtype: str,
    *,
    backend: str = "unspecified",
) -> ReductionBound:
    """Compute gamma_(n-1) * ABS_SUM with conservative upward rounding."""

    n = int(term_count)
    if n < 1 or n != term_count:
        raise ValueError("term_count must be a positive integer")
    abs_sum = float(operand_abs_sum)
    if not math.isfinite(abs_sum) or abs_sum < 0.0:
        raise ValueError("operand_abs_sum must be finite and nonnegative")
    normalized = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    eps = machine_epsilon(normalized)
    u = eps / 2.0

    with localcontext() as context:
        context.prec = 100
        decimal_u = Decimal.from_float(u)
        product = Decimal(n - 1) * decimal_u
        if product >= Decimal(1):
            raise InvalidGammaDomainError("Invalid gamma domain: (n-1)u must be < 1")
        gamma_exact = Decimal(0) if n == 1 else product / (Decimal(1) - product)
        bound_exact = gamma_exact * Decimal.from_float(abs_sum)
    gamma = _decimal_to_float_upward(gamma_exact)
    bound = _decimal_to_float_upward(bound_exact)
    return ReductionBound(
        dtype=normalized,
        backend=str(backend),
        machine_epsilon=eps,
        unit_roundoff=u,
        term_count=n,
        gamma=gamma,
        operand_abs_sum=abs_sum,
        bound=bound,
    )


def reduction_bound_from_terms(
    terms: Iterable[float], dtype: str, *, backend: str = "unspecified"
) -> ReductionBound:
    n, abs_sum = exact_operand_abs_sum(terms)
    return forward_error_bound(n, abs_sum, dtype, backend=backend)


def combined_reduction_bound(first: ReductionBound, second: ReductionBound) -> float:
    """Conservatively add two independent reduction bounds."""

    with localcontext() as context:
        context.prec = 100
        exact = Decimal.from_float(first.bound) + Decimal.from_float(second.bound)
    return _decimal_to_float_upward(exact)


def guard_empty_target_equivalence(
    authoritative_score: float,
    diagnostic_score: float,
    authoritative_terms: Iterable[float],
    diagnostic_terms: Iterable[float],
    *,
    authoritative_dtype: str,
    diagnostic_dtype: str,
    authoritative_backend: str,
    diagnostic_backend: str,
) -> EmptyTargetGuardResult:
    """Validate numerical equivalence and preserve A as the scientific output."""

    authoritative = float(authoritative_score)
    diagnostic = float(diagnostic_score)
    if not math.isfinite(authoritative) or not math.isfinite(diagnostic):
        raise NonFiniteEmptyTargetError(NONFINITE_STATUS)

    bound_a = reduction_bound_from_terms(
        authoritative_terms, authoritative_dtype, backend=authoritative_backend
    )
    bound_b = reduction_bound_from_terms(
        diagnostic_terms, diagnostic_dtype, backend=diagnostic_backend
    )
    correction_bound = combined_reduction_bound(bound_a, bound_b)
    difference = abs(authoritative - diagnostic)
    if difference > correction_bound:
        raise EmptyTargetMismatchError(MISMATCH_STATUS)

    return EmptyTargetGuardResult(
        scientific_score=authoritative,
        diagnostic_score=diagnostic,
        absolute_difference=difference,
        correction_bound=correction_bound,
        equivalent=True,
        authoritative=bound_a,
        diagnostic=bound_b,
    )
