# R4-4D0 CTC Sequence-Hypothesis Feasibility Audit

Final: **R4_4D0_CTC_HYPOTHESIS_SIGNAL_STRONG**

This was a TRAIN-only frozen-checkpoint audit. No neural training, validation candidate scoring, or R4 TEST access occurred.

## Primary TRAIN results

| Family | Del vs non-del ROC | PR | Del vs sub ROC | PR | Clean zero-pref |
|---|---:|---:|---:|---:|---:|
| RAW | 0.905407 | 0.207316 | 0.904593 | 0.659342 | 0.137168 |
| TARGET | 0.875496 | 0.228741 | 0.884558 | 0.664103 | 0.063162 |
| TIME | 0.889480 | 0.192990 | 0.889639 | 0.629133 | 0.137168 |

Selected score family: **RAW**

Raw length-bias verdict: **RAW_SCORE_LENGTH_BIASED**

## Feasibility gates

- deletion_vs_nondeletion_roc_at_least_0_65: PASS
- deletion_vs_substitution_roc_at_least_0_60: PASS
- speaker_direction_consistent_at_least_75_percent: PASS
- clean_zero_threshold_deletion_preference_at_most_25_percent: PASS

## Closure

- VALIDATION candidate hypothesis scores: NO
- VALIDATION thresholds/normalization selection: NO
- R4 TEST accessed: NO
- Neural training: NO
