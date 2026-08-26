# R4-4D1 Locked VALIDATION-Only Technical Re-execution V2

Final: **R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL**

All frozen driver, correction-contract, numerical-contract, preregistration, checkpoint, V4, matched-control, and threshold identities passed verification. The canonical matched mapping was 1,434/1,434 before inference. TRAIN scoring and recalibration did not run. RAW theta remained `2.197946548461914`.

## Primary result

- Accuracy: 0.9355279645
- Balanced Accuracy: 0.6965871373
- Binary Macro-F1: 0.6437496738
- Deletion precision / recall / F1: 0.2531486146 / 0.4398249453 / 0.3213429257
- Binary confusion `[[TN,FP],[FN,TP]]`: `[[24237,1186],[512,402]]`
- Three-relation Macro-F1: 0.4622633292
- Correct / substitution / deletion F1: 0.7507534740 / 0.3146935878 / 0.3213429257
- Correct false-deletion: 0.0463113494
- Substitution false-deletion: 0.0495495495

## Matched control

- Binary Macro-F1: 0.6133438208
- Deletion precision / recall / F1: 0.7203791469 / 0.4239888424 / 0.5338015803
- Confusion: `[[599,118],[413,304]]`

## Continuous transfer

- Deletion vs non-deletion ROC-AUC / PR-AUC: 0.8877398574 / 0.2338867573
- Deletion vs substitution ROC-AUC / PR-AUC: 0.8876424291 / 0.6978412219

Confirmation failed gates A, B, C, and F. Strong Partial failed its Binary Macro-F1 and deletion-F1 requirements. Both frozen continuous threshold-transfer criteria passed, fixing the final classification without subjective override.

The hypothesis scorer rescued 2,491/3,087 greedy false deletions (80.6932%) while retaining 230/409 greedy true deletions (56.2347%). BEST_SUB_PHONE Top-1 accuracy was 0.616541 at 0.748874 coverage.

Validation runtime was 27.8899 seconds on CUDA: 7.0489 seconds feature extraction, 1.3479 seconds inference, and 16.8084 seconds for 1,061,208 hypothesis scores.

Locked v2 execution count: **ONE**. Neural training: **NO**. Threshold changed: **NO**. R4 TEST accessed: **NO**.
