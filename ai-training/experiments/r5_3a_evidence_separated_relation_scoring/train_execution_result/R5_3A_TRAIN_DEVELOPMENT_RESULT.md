# R5-3A Frozen TRAIN Development Result

## Status

`R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED`

## Exact population and method

- Exact joined TRAIN rows: 16,582; positive 323; negative 16,259.
- Feature order: `[A,S,D]`; all finite; no forbidden feature used.
- Exact 12-fold TRAIN speaker LOSO with calibration-only StandardScaler, fixed balanced L2 LogisticRegression, and class-1 `predict_proba`.
- All 12 folds converged within max_iter=1000.

## Continuous metrics

- Addition/all-negative AUC: 0.78258481085112752
- Addition/correct-only AUC: 0.82699905982957167
- Addition/substitution-containing AUC: 0.71439877957205189
- Addition/deletion-containing AUC: 0.61122009669681732

## OOF decision metrics

- TP/FP/FN/TN: 70 / 587 / 253 / 15672
- Binary Macro-F1: 0.55837860871417921
- Addition P/R/F1: 0.106544901065449 / 0.21671826625386997 / 0.14285714285714285
- Correct/SUB/DELETE FAR: 0.024468085106382979 / 0.052578565672844482 / 0.099073414112615818
- Exact-event F1: 0.044044044044044044

## Frozen gates

6 / 8 PASS.

- G1: PASS
- G2: PASS
- G3: PASS
- G4: PASS
- G5: PASS
- G6: FAIL
- G7: FAIL
- G8: PASS

Robust threshold: `R5_3A_ROBUST_THETA_NOT_AUTHORIZED`

## Protocol

This was the single authorized real TRAIN classifier execution. No neural checkpoint was loaded, no audio was read, and no new acoustic score was created. VALIDATION and TEST were not resolved or accessed. The frozen contract and static implementation were unchanged. No rerun or retuning occurred.
