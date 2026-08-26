# R3-2B Speaker-Transfer Margin Threshold Audit

Status: **RESEARCH_ONLY**, **NO_TRAINING**, **VALIDATION_EVIDENCE_ONLY**,
**NO 0-100 CALIBRATION**, **TEST_CLOSED**, **NOT_RUNTIME_CONNECTED**.

Final status: `R3_2B_TRANSFER_PASS`. Binary readiness:
`BINARY_MARGIN_TRANSFER_READY`.

## Reproduction and protocol

The audit verified the R3-1D checkpoint SHA and the locked R3-2A evidence
SHA before analysis. The evidence contains exactly 28,212 sequential rows for
ABA, HKK, HQTV, LXC, MBMPS, and SVBI. HARD_ARGMAX metrics, expected-margin
ROC-AUC/PR-AUC, and all registered margin distribution summaries reproduced
within 1e-9 tolerance.

Six leave-one-speaker-out folds were run. Each fold selected one global
expected-margin threshold from the other five speakers, constrained to
calibration substitution recall at least 0.50. The held-out speaker had no
influence on its threshold. No per-phone, per-pair, or held-out-speaker
threshold was fit.

## Primary out-of-fold result

Every validation row appeared in exactly one held-out prediction set. OOF
accuracy was 0.774280, balanced accuracy 0.654656, Macro-F1 0.590068, and
substitution precision/recall/F1 0.229385/0.503953/0.315269. Relative to
HARD_ARGMAX, Macro-F1 improved by 0.107055 and substitution F1 by 0.041675.

All registered gates passed: Macro-F1 gain at least 0.05, OOF substitution
recall at least 0.50, substitution-F1 gain at least 0.03, every held-out
speaker substitution recall at least 0.35, and Macro-F1 improvement for all
six speakers.

The six thresholds ranged from -1.378984 to -1.207395, with mean/median
-1.295094/-1.295098, population SD 0.051993, and IQR 0.043007. In context of
the broad margin distributions and successful held-out recall, this is
`THRESHOLD_STABLE`.

## Known limitations

DH-to-D remains `DH_D_SCORE_FAILURE_MODE`: support 300, OOF detection recall
0.216667, 235 false negatives, and median margin +0.051111. It behaves much
worse than overall substitution recall 0.503953 and the mean/median recall of
other large pairs, 0.503037/0.482759.

V4 substitution exclusion rate has a moderate negative descriptive
association with held-out recall and Macro-F1 across only six speakers.
SVBI combines the highest exclusion rate with the lowest recall, while ABA
has high exclusion but recall above the OOF average. This is an association,
not evidence of causality.

The prospective all-validation threshold is -1.293920 under the same recall
constraint and optimization rule. It is marked
`VALIDATION_CALIBRATED_CANDIDATE`, `NOT_TESTED`, and
`NOT_PRODUCTION_CALIBRATED`. No TEST data was accessed and no 0-100 mapping
was created.
