# R5-1C Post-Hoc Validation Transfer Failure Audit

Status: `R5_1C_POSTHOC_FAILURE_AUDIT_COMPLETE`

The authoritative scientific status remains `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED`.

## Speaker failure

- Speakers above the historical correct-only FAR reference: 5/6
- Speaker FAR median/min/max: 0.088493 / 0.033025 / 0.104228

## Score location

- All-word median TRAIN/VALIDATION: -0.310241 / -0.159461
- Addition median TRAIN/VALIDATION: 0.276697 / 0.404830
- Correct-only median TRAIN/VALIDATION: -0.442513 / -0.293713

## Ranking versus decision

- AUC all delta: -0.022202418
- AUC correct-only delta: -0.030645590
- Correct-only FAR delta: +0.048335149

## Exploratory interpretation

`MIXED_CALIBRATION_AND_CLASS_OVERLAP`

Useful ranking transferred, but speaker-dependent score location and a broad rise in false additions at the fixed threshold indicate calibration shift. Simultaneous overlap across positive and negative score regions and across negative cohorts indicates intrinsic class overlap also remained material.

This interpretation is post-hoc and exploratory. It does not authorize recalibration, TEST access, or a new model.
