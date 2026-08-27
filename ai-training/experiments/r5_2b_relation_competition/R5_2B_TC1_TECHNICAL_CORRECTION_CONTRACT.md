# R5-2B-TC1 Empty-Target Execution-Guard Technical Correction Contract

## Frozen status

`R5_2B_TC1_TECHNICAL_CORRECTION_CONTRACT_FROZEN`

This contract is technical only. The R5-2B scientific scorer, contract, hypothesis, population, LOSO protocol, thresholds, gates, and event semantics remain unchanged. The first execution remains preserved as `R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET` with no scientific performance result.

## TC0 evidence

TC0 established `FLOATING_POINT_EQUIVALENCE_ONLY`. The same four float32 blank log-probabilities produced `-15.667129516601562` under CUDA reduction and `-15.667130470275879` under CPU reduction, a difference of `9.5367431640625e-7`. No operands, T, blank index, normalization, or mathematical semantics differed.

## Authoritative scientific value

The unchanged imported scorer, SHA-256 `2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3`, is authoritative. Its empty-target return value is the scientific score. The independent value is diagnostic metadata only and cannot overwrite, average, promote, or otherwise alter the scorer result.

## General equivalence policy

For each independent finite reduction `j`:

- `eps_j = machine epsilon(dtype_j)`
- `u_j = eps_j / 2`
- `gamma_j = 0` for `n <= 1`
- otherwise `gamma_j = ((n-1)u_j)/(1-(n-1)u_j)`, requiring `(n-1)u_j < 1`
- `BOUND_j = gamma_j * ABS_SUM_j`

`ABS_SUM_j` is the deterministic compensated/high-precision host sum of the magnitudes of the actual represented operands, and implementation arithmetic for each bound is rounded conservatively upward.

Because A and B are independent reductions, freeze:

`CORRECTION_BOUND = BOUND_A + BOUND_B`

and accept only when:

`abs(A-B) <= CORRECTION_BOUND`.

The two-bound sum is mathematically required because either reduction may round away from the exact real sum in the opposite direction. No empirical absolute or relative tolerance is allowed.

## Regression classification

The TC0 fixture has a single-reduction float32 bound `2.8015016120413103e-6` and two-reduction bound `5.603003224082621e-6`. Its historical difference `9.5367431640625e-7` passes. The original static fixture difference `0.0` also passes. These checks were applied after the general rule was selected.

## Failure behavior

NaN or either infinity where a finite empty-target likelihood is required stops with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_NONFINITE_EMPTY_TARGET`.

A finite difference beyond the derived bound stops with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET_MISMATCH`. The bound may not be increased after observation.

## Required next stage

Before TRAIN is eligible, a separate synthetic TC1 regression must verify the unchanged scorer SHA and scientific output, the bound implementation, float32 CPU/CUDA and float64 fixtures, exact equality, deliberate beyond-bound rejection, all nonfinite rejection cases, diagnostic non-substitution, and deterministic repeated results.

This contract neither implements that correction nor runs the regression, TRAIN, VALIDATION, or TEST.
