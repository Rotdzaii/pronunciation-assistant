# R5-2B-TC0 Empty-Target Numerical Equivalence Audit

## Result

Technical interpretation: `FLOATING_POINT_EQUIVALENCE_ONLY`.

Final status: `R5_2B_TC0_NUMERICAL_EQUIVALENCE_CONFIRMED`.

The failed R5-2B execution remains frozen as `R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET`. TC0 produced no scientific metrics and evaluated none of the eight development gates.

## Code-path provenance

| Field | Frozen scorer | Failed execution adapter |
|---|---|---|
| Source | Same deterministic `[4,41]` CUDA logits | Same tensor |
| T / blank | `4 / 40` | `4 / 40` |
| Input dtype/device | float32 / CUDA | float32 / CUDA |
| `log_softmax` | CUDA float32 | CUDA float32 |
| Blank operands | `log_probs[:,40]` | Identical values in `log_probs[0][:,40]` |
| Before sum | Remains on CUDA | Detached and copied to CPU |
| Sum | CUDA float32 `Tensor.sum` | CPU float32 `Tensor.sum` |
| Python conversion | After sum | After sum |
| Normalization | None | None |

The scorer source is `r5_2b_scorer.py:139-164`. The failed adapter guard is `r5_2b_train_execution_driver.py:385-395`.

## Controlled same-tensor evidence

For CUDA float32, the scorer and same-device explicit sum were both `-15.667129516601562`. Copying the identical four log-probability operands to CPU before summation yielded `-15.667130470275879`, exactly reproducing the historical absolute difference `9.5367431640625e-7`. Framework empty-target CTC returned the CPU-reduction value.

For CPU float32, all primary paths agreed at `-15.667130470275879`. For CPU and CUDA float64, all primary paths agreed at `-15.66712968974069`.

Separately recomputing `log_softmax` on the same device produced no difference. The discrepancy appeared only when the same float32 operands were reduced on different devices/backends.

## Accumulation analysis

The four CUDA float32 blank log-probabilities were:

`[-4.125710964202881, -3.2495081424713135, -4.680192470550537, -3.61171817779541]`.

Their float64-promoted/Python compensated sum was `-15.667129755020142`, between the CUDA and CPU float32 reductions. With unit roundoff `u = eps/2`, float32 `eps = 1.1920928955078125e-7`, and four terms, the standard sequential-summation bound `gamma_3 * sum(abs(x_i))` is approximately `2.8015e-6`. The observed `9.5367e-7` difference is within that independently derived bound.

This bound is based on dtype and reduction length, not selected to fit the historical discrepancy.

## Static provenance

The original frozen static test was rerun without modification. It reproduced:

- Explicit: `-19.43882697170362`
- Framework: `-19.43882697170362`
- Absolute difference: `0.0`
- Summary SHA-256: `F0E54A0A4DBC1378F424E6C8FE8B0A9861CDD72DCBE72C045B96660324723197`

This exactly matches the frozen static-run identity.

## Contract impact and correction policy

Both failed paths implement the same frozen mathematical quantity. The scientific contract and frozen scorer remain valid and unchanged.

A separate technical correction is justified only in the execution-orchestration layer. A future authorized correction may use the imported frozen scorer as the authoritative empty-target score and retain an independent cross-check solely as a diagnostic using a preregistered dtype-aware forward-error bound. It may not change the formula, scorer, contract, model, population, thresholds, gates, or scientific interpretation.

TC0 does not implement that correction and does not authorize another TRAIN execution.

## Protocol audit

- Training: no
- Checkpoint load/inference: no/no
- TRAIN audio/performance: no/no
- LOSO or threshold search: no/no
- VALIDATION/TEST: no/no
- Contract/scorer/failed execution modified: no/no/no
