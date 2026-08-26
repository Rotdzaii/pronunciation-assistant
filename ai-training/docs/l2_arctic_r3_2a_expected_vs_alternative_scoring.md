# R3-2A Expected-vs-Best-Alternative Scoring Audit

Status: **RESEARCH_ONLY**, **VALIDATION_ONLY**, **NO_TRAINING**,
**TEST_CLOSED**, **NOT_RUNTIME_CONNECTED**.

Final status: `R3_2A_SCORING_WEAK`. Raw-score verdict:
`NOT_YET_SUITABLE` for calibration.

## Reproduction and evidence

The audit used only the selected R3-1D checkpoint at epoch 47, SHA-256
`5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E`,
and the 28,212 locked validation rows for ABA, HKK, HQTV, LXC, MBMPS, and
SVBI. The selected 40-class metrics and HARD_ARGMAX downstream metrics
reproduced within tolerance; all deltas were zero apart from floating-point
round-off of 5.6e-17 in two binary F1 values.

An initial evaluator probe used a full-tensor weighted-CE reducer instead of
the legacy per-batch reducer. It stopped before score export as designed. The
probe is preserved under `r3_2a_attempt1_invalid_loss_reducer`. Correcting
only the reducer reproduced legacy validation loss exactly; no tolerance,
checkpoint, preprocessing, or model output was changed.

The final audit exports one evidence row per validation sample with expected
posterior, best-alternative phone/probability, expected and alternative
logits, and expected margin. Relation origin is analysis-only and was not a
model input. TEST audio was not resolved, extracted, inferred, or inspected.

## Score separation

Expected posterior achieved substitution ROC-AUC 0.717874 and PR-AUC
0.247386. Correct/substitution median posterior was 0.339865/0.123098.

Expected margin achieved substitution ROC-AUC 0.739384 and PR-AUC 0.263482.
Correct/substitution median margin was 0.165174/-1.296994. Margin therefore
provided the stronger threshold-independent separation.

## Registered global-threshold results

HARD_ARGMAX produced Macro-F1 0.483013 and substitution
precision/recall/F1 0.165477/0.789275/0.273594.

The globally selected expected-posterior threshold was 0.043371. It produced
Macro-F1 0.598685 and substitution precision/recall/F1
0.272268/0.296322/0.283786.

The globally selected expected-margin threshold was -2.338524. It produced
Macro-F1 0.605703 and substitution precision/recall/F1
0.287203/0.303197/0.294983.

Both continuous scores improved Macro-F1, but both failed the registered
minimum substitution recall of 0.50. Neither therefore passed the meaningful
improvement gate. The correct final status is `R3_2A_SCORING_WEAK`.

Using the selected global margin threshold, per-speaker Macro-F1 ranged from
0.542846 to 0.645105, but substitution recall ranged from 0.189873 to
0.347262 and substitution precision from 0.159574 to 0.471664. This minority
class instability, together with failure of the scoring gate, yields
`NOT_YET_SUITABLE` for calibration. No 0-100 mapping was created.
