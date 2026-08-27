# R5-2B-TC1 Synthetic Static Regression Report

## Result

`R5_2B_TC1_STATIC_REGRESSION_PASS`

All 25 frozen pass criteria were satisfied. The complete 22-test TC1 suite ran twice with byte-identical outputs. No real audio, checkpoint inference, LOSO, or Phoenix performance metric was used.

## Identity verification

Ten required top-level identities matched. All 42 entries across the frozen R5-2B contract, original static-verification, failed-execution, TC0, and TC1 contract manifests matched both recorded byte sizes and SHA-256 values.

The frozen scorer remained:

`2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3`

The failed execution manifest remained:

`92163F8DE3ECEAFEE2950AF57302AA565C0F3612A2D6F0D36462552FA66BA09E`

## Numeric guard

The additive guard implements the frozen TC1 policy:

- `u = machine_epsilon(dtype) / 2`
- `gamma_(n-1) = ((n-1)u) / (1-(n-1)u)` with `(n-1)u < 1`
- `BOUND_j = gamma_(n-1) * SUM_i abs(x_i)`
- `CORRECTION_BOUND = BOUND_A + BOUND_B`
- finite values are equivalent exactly when `abs(A-B) <= CORRECTION_BOUND`

Bound components are conservatively rounded upward. No fixed absolute tolerance, observation-derived tolerance, or relative tolerance is used. For `n=1`, gamma and the pure reduction bound are exactly zero. An invalid gamma domain is rejected deterministically.

## Historical controls

The frozen TC0 float32 values were:

- CPU: `-15.667130470275879`
- CUDA: `-15.667129516601562`
- Difference: `9.5367431640625e-7`
- Combined bound: `5.603003224082621e-6`

Classification: `EQUIVALENT / PASS`.

The CPU same-tensor calculation was rerun from the original deterministic synthetic logits formula. The available Python runtime was CPU-only, so the CUDA value and identical operands were replayed from the hash-verified frozen TC0 artifact; no new CUDA or checkpoint inference was attempted.

The float64 CPU/CUDA values were both `-15.66712968974069`; their combined bound was `1.0436404866803105e-14`, materially below the float32 bound.

The unchanged original 50-test static fixture reran with SHA-256:

`F0E54A0A4DBC1378F424E6C8FE8B0A9861CDD72DCBE72C045B96660324723197`

Its explicit and framework empty-target values remained `-19.43882697170362`, with zero difference.

## Failure controls

- Exact equality: accepted.
- Strictly within-bound perturbation: accepted.
- Equality at the bound: accepted using the frozen `<=` operator.
- Beyond-bound perturbation: rejected with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET_MISMATCH`.
- NaN, positive infinity, and negative infinity: rejected with `R5_2B_TC_EXECUTION_TECHNICAL_FAILURE_NONFINITE_EMPTY_TARGET`.

## Scientific immutability

For a within-bound difference, the returned scientific score remained byte-identical to authoritative value A. The diagnostic value did not overwrite or modify it. A synthetic relation score `C = BEST_INSERT - BEST_NON_ADDITION` was byte-identical before and after the guard.

The scientific scorer, R5-2B contract, failed execution, population, threshold protocol, and eight gates were not changed.

## Determinism

- Runs: 2
- Tests per run: 22
- Passed per run: 22
- Run 1 SHA-256: `349237D74A8DD2FA9883F1B2E32FD4B0C2968C9B5A904F696C224C19A0056975`
- Run 2 SHA-256: `349237D74A8DD2FA9883F1B2E32FD4B0C2968C9B5A904F696C224C19A0056975`
- Byte-identical: yes

## Protocol

- TRAIN audio accessed: no
- Checkpoint inference: no
- TRAIN performance calculated: no
- VALIDATION accessed: no
- TEST accessed: no
- Frozen scorer modified: no
- Failed execution modified: no
