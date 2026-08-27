# R4-4D1 Locked TRAIN-Calibrated Hypothesis Validation

Final: **R4_4D1_SOURCE_VERIFICATION_FAIL**

The single authorized execution verified every frozen SHA, scored the complete TRAIN population, selected the frozen TRAIN-only RAW-score threshold, wrote/closed/hashed/reopened the threshold artifact, and then scored the complete VALIDATION population once.

The run stopped at the mandatory matched-control guard before validation metrics or prediction exports were written. Of the 1,434 frozen matched identities, 233 did not map under the frozen driver's row identity convention. No intersection or reduced control was evaluated.

## Frozen TRAIN result

- Threshold candidates: 56,263
- Eligible thresholds: 12,493
- Selected threshold: 2.197946548461914
- Threshold SHA-256: `36F6FD5AB6B7E98A607D499445E455DCAB8C3DD4ACDD19F252DC472FCDD07E94`
- Binary Macro-F1: 0.6284987892241034
- Balanced Accuracy: 0.7081837561549786
- Deletion precision / recall / F1: 0.2102954080140392 / 0.46567357512953367 / 0.2897441063872658
- Correct false-deletion: 0.04865727200212709
- Substitution false-deletion: 0.05471280040906767
- Three-relation Macro-F1: 0.48467653339199873

## Failure evidence

The matched-control CSV and the prior R4-4C2 export encode `source_csv_row` as `source_index + 2`. The frozen R4-4D1 driver emitted raw `source_index`. A static identity check reproduces exactly 233 missing identities under the driver convention and zero missing identities after accounting for the two-line offset.

Because the frozen driver cannot be modified and a second execution is not authorized, the in-memory VALIDATION scores were not regenerated. Validation metrics, gates, and the scientific R4-4D1 hypothesis outcome therefore remain unevaluated.

R4 TEST accessed: **NO**. Neural training: **NO**. Post-hoc threshold change: **NO**. Locked execution count: **ONE**.
