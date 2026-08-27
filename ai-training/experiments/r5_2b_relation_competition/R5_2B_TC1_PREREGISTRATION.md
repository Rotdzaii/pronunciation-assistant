# R5-2B-TC1 Empty-Target Guard Technical Correction Preregistration

## Status and scope

This is a technical-correction contract only. It preserves the frozen R5-2B scorer, scientific contract, population, hypothesis families, relation-competition score, LOSO procedure, thresholds, eight gates, and event semantics.

The original R5-2B attempt remains `R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET`. It exposed no TRAIN audio, inference, performance, thresholds, or gates. TC0 subsequently established `FLOATING_POINT_EQUIVALENCE_ONLY`: the same four float32 operands were reduced on CUDA and CPU with different valid rounding outcomes.

## Authoritative score

The imported frozen scorer identified by SHA-256 `2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3` is authoritative. For an empty target, its returned value for

`sum_t log_softmax(logits[t], dim=-1)[40]`

is the scientific score. An independently reduced diagnostic must never replace, average, promote, or otherwise modify that value.

## Diagnostic preconditions

The independent comparison is valid only when both paths use the same finite blank log-probability operands, the same `T`, the same blank index, and numerically identical element values after any lossless device transfer. The implementation must record devices, dtypes, term count, absolute operand sum, both reduction results, observed difference, both forward-error bounds, and the combined correction bound.

If operands, slicing, blank index, or term count differ, the reduction-equivalence rule does not apply and the future execution must stop as an empty-target mismatch.

## Floating-point policy

For reduction `j` in `{A,B}`, let `eps_j` be the machine epsilon of its floating-point dtype and define unit roundoff `u_j = eps_j / 2`. For `n <= 1`, set `gamma_j = 0`. For `n >= 2`, require `(n-1)u_j < 1` and define:

`gamma_j = ((n-1)u_j) / (1 - (n-1)u_j)`.

Let `ABS_SUM_j` be the nonnegative high-precision sum of the magnitudes of the actual represented operands reduced by path `j`. The implementation must compute it using deterministic compensated/high-precision host accumulation and round the reported bound upward, not downward.

Define:

`BOUND_j = gamma_j * ABS_SUM_j`

and, because A and B are independent reductions:

`CORRECTION_BOUND = BOUND_A + BOUND_B`.

Finite results are diagnostically equivalent iff:

`abs(A-B) <= CORRECTION_BOUND`.

The two-bound sum is required because either reduction may deviate from the exact real sum in the opposite direction. A fixed empirical tolerance is prohibited.

## Historical regression after policy freeze

For the four-term TC0 float32 fixture, the single-reduction bound is `2.8015016120413103e-6`; therefore the independently derived two-reduction bound is `5.603003224082621e-6`. The historical difference `9.5367431640625e-7` passes.

The original float64 static fixture has explicit and framework values `-19.43882697170362`, difference `0.0`, and also passes. Neither fixture defined the policy.

## Future failure behavior

NaN or either infinity where a finite empty-target result is expected must stop with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_NONFINITE_EMPTY_TARGET`.

A finite difference beyond the derived bound must stop with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET_MISMATCH`. The bound may not be loosened after observation.

## Required synthetic regression

Before any corrected TRAIN execution, a separately authorized TC1 synthetic static regression must verify scorer identity, unchanged scientific values, bound implementation, float32 CPU/CUDA equivalence, float64 and exact fixtures, deliberate beyond-bound rejection, all three nonfinite rejections, diagnostic non-substitution, and deterministic repeated output.

This contract does not implement that regression and does not authorize TRAIN, VALIDATION, or TEST.
