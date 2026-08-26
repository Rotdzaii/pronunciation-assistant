# R5-1 Frozen TRAIN Development — Pre-Metric Technical Stop

The frozen source identities passed. The first and only guarded driver invocation stopped before audio loading, model inference, hypothesis scoring, threshold calibration, or performance metrics.

## Preserved initial stop

The initial implementation incorrectly required `label_quality == "clean"` in addition to the canonical V4 `relation == "addition"`. That bookkeeping error produced zero mapped positive words. No scores or metrics had been exposed, so an additive metadata-only correction audit was permitted under the frozen technical-failure policy. The original driver and its failed row-accounting artifact remain unchanged.

## Corrected frozen population

The additive correction reproduced the preregistered population exactly:

- 16,582 runtime-evaluable TRAIN words
- 323 addition-positive words
- 16,259 addition-negative words
- 423 clean source addition events
- 342 clean addition events inside the runtime population
- 19 multiple-addition positive words
- 117 mixed substitution/addition words
- 26 mixed deletion/addition words

The 81 source events outside runtime scoring comprise 49 without a manual word, 14 without an expected sequence, and 18 in words excluded for unresolved co-occurring evidence.

## Independent frozen-contract blocker

The preregistration requires every candidate to satisfy:

`len(H) + adjacent-identical-target count <= encoder T`

and requires zero theoretically unalignable candidates. The exact metadata audit found:

- 2,993,622 total KEEP/INSERT hypotheses
- 2,977,040 INSERT hypotheses
- 196 unalignable INSERT candidates
- 65 affected TRAIN words
- 132 candidates requiring 4 steps where `T=3`
- 64 candidates requiring 3 steps where `T=2`

`CTCLoss(zero_infinity=True)` would turn an infinite loss for these impossible alignments into zero loss. The frozen contract explicitly forbids treating that output as a high sequence score. Skipping candidates, assigning negative infinity, adding acoustic context, or changing the population would alter frozen semantics.

## Result

R5-1 did not reach a scientific performance result. Continuous AUC, LOSO thresholds, OOF binary metrics, exact-event metrics, and the six gates were not calculated. `ROBUST_THETA_NOT_AUTHORIZED`.

Technical stop: `R5_1_EXECUTION_TECHNICAL_FAILURE_CTC_ALIGNABILITY`.

VALIDATION and TEST remained untouched. No neural training occurred, and no R4 or R5-0 artifact was modified.
