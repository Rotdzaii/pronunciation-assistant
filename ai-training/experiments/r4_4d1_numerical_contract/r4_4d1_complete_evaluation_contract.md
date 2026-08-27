# R4-4D1A Complete Numerical Evaluation Contract

Status: **FROZEN BEFORE TRAIN THRESHOLD CALIBRATION AND VALIDATION**

This is an additive contract. It does not modify `r4_4d1_preregistered_score_design.json` (SHA-256 `7FD870DF2321465D716EAFB1B66E7D6116C153E3D8BCD494494F3FAE44ACC784`). The original preregistration plus this document define the future locked R4-4D1 experiment.

## Frozen sources

- V4: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- R4-4C2 checkpoint: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`
- R4-4C2 manifest: `BE82690C366049E659F5A872256BF23616566E18D37A7AFB8A1E7169629F1DB9`
- R4-4D0 manifest: `A8443C85AA9C03879E3A907CC3AB05CC6154BB365BED70A59B4ED8E9FBA1A920`
- R4-4D0 scorer: `DC141D38091C9FC60BA2F0A8447FF918DC98B5D99499D336A524CABEB8F4948C`

## Populations

TRAIN calibration contains 16,259 words and 56,304 expected-phone rows: 48,893 correct, 5,867 substitution, and 1,544 deletion. Speakers are BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, and ZHAA.

The one future VALIDATION evaluation contains 7,728 words and 26,337 expected-phone rows: 22,759 correct, 2,664 substitution, and 914 deletion. Speakers are ABA, HKK, HQTV, LXC, MBMPS, and SVBI.

R4 TEST speakers ASI, ERMS, SKA, THV, TXHC, and YDCK remain closed.

## Frozen acoustic model

Use only the frozen R4-4C2 CNN plus one packed bidirectional GRU. Input is the full MFA word span, mono 16 kHz, with 64 Slaney-normalized mel bins, FFT 512, a 400-sample Hann window, 160-sample hop, and per-word maximum-relative dB. The CNN channels are 16→32→64→96. The BiGRU has one bidirectional layer with hidden size 96. The head is `Linear(192,41)`, with blank index 40.

No training, checkpoint change, feature change, phone boundary, or duration feature is permitted.

## Exact RAW CTC score

For frozen logits of shape `[T,41]`:

```text
log_probs = log_softmax(logits, dim=-1)

NLL(H) = CTCLoss(
    log_probs,
    H,
    input_lengths=[T],
    target_lengths=[len(H)],
    blank=40,
    reduction="none",
    zero_infinity=True,
)

RAW_SCORE(H) = -NLL(H)
```

An empty target is valid and must use PyTorch's exact CTC behavior. A non-finite score stops the future run. No custom empty-target formula is permitted.

There is no target-length normalization, time normalization, length or duration correction, phone prior, speaker prior, or greedy-omission feature.

## Local hypotheses and decision score

For expected sequence `E=[E0,...,E(n-1)]` and expected position `i`:

- `H_KEEP = E`
- `H_DELETE(i) = E` with `Ei` removed
- `H_SUB(i,q) = E` with `Ei` replaced by each of the other 39 canonical phones

The manual observed phone is never given to the scorer. `BEST_SUB_SCORE_i` is the maximum substitution score. Exact ties select the lowest frozen canonical phone index.

```text
KEEP_i   = RAW_SCORE(H_KEEP)
DELETE_i = RAW_SCORE(H_DELETE(i))
SUB_i    = BEST_SUB_SCORE_i

D_i = DELETE_i - max(KEEP_i, SUB_i)
```

`D_i` and all threshold values and comparisons use float64. This is the only deletion-decision scalar.

## TRAIN-only threshold calibration

Let `U` be the sorted unique finite TRAIN `D_i` values. Candidates, in ascending order, are:

1. `np.nextafter(min(U), -np.inf)`
2. every value in `U`
3. `np.nextafter(max(U), +np.inf)`

`D_i >= threshold` predicts deletion. The first edge predicts all deletion; the last predicts no deletion.

A threshold is eligible only when all conditions hold:

- TRAIN deletion recall ≥ 0.45
- TRAIN substitution false-deletion rate ≤ 0.25
- Every TRAIN speaker with at least 30 deletion rows has deletion recall ≥ 0.25

There is no correct false-deletion eligibility constraint.

Among eligible candidates select, using exact float64 comparisons:

1. highest binary Macro-F1
2. higher deletion F1
3. higher deletion precision
4. higher three-relation Macro-F1
5. higher threshold

The final tie-break is conservative against false deletion. If no candidate is eligible, return `R4_4D1_TRAIN_CALIBRATION_NO_ELIGIBLE_THRESHOLD` and stop before VALIDATION.

The selected threshold, full candidate metrics, and selection trace must be written and hashed before any VALIDATION acoustic or hypothesis scoring.

## Final relation decision

```text
if D_i >= theta:
    deletion
elif KEEP_i >= SUB_i:
    correct
else:
    substitution, observed phone = BEST_SUB_PHONE_i
```

Equality at the deletion threshold predicts deletion. Equality between KEEP and SUB predicts correct. No second threshold exists.

## Metrics

Binary truth and predictions map correct and substitution to non-deletion (0), and deletion to deletion (1). The confusion matrix is `[[TN,FP],[FN,TP]]`.

```text
Accuracy = (TP+TN)/(TP+TN+FP+FN)
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
Balanced Accuracy = (TNR+TPR)/2
Deletion Precision = TP/(TP+FP)
Deletion Recall = TP/(TP+FN)
Deletion F1 = harmonic mean(Deletion Precision, Deletion Recall)
Non-deletion Precision = TN/(TN+FN)
Non-deletion Recall = TN/(TN+FP)
Non-deletion F1 = harmonic mean(Non-deletion Precision, Non-deletion Recall)
Binary Macro-F1 = mean(Non-deletion F1, Deletion F1)
```

Every zero denominator produces 0; no class is dropped.

Correct false-deletion rate is predicted deletion among all true-correct rows. Substitution false-deletion rate is predicted deletion among all true-substitution rows. There are no exclusions.

Three-relation class order is correct, substitution, deletion. Report precision, recall, F1, and support for every class, and their unweighted Macro-F1. Confusion rows are truth and columns are prediction. Zero denominators produce 0.

Substitution-phone coverage is true substitutions predicted substitution divided by all true substitutions. Top-1 accuracy compares `BEST_SUB_PHONE` with the manual observed canonical phone within that covered population. Zero coverage gives accuracy 0.

## Matched control

Use exactly:

`ai-training/experiments/r4_4a_ctc_sequence_feasibility/validation_word_eligible_matched_control.csv`

SHA-256: `D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798`

It contains 717 deletion rows and 717 matched non-deletion rows: 1,434 rows, 32 phones, and all six validation speakers. Every identity must match exactly one future prediction. Do not rebuild, resample, intersect, or silently shrink it. Identity failure is a contract failure.

Report full binary metrics. Hard matched gates are Macro-F1 ≥ 0.60 and deletion F1 ≥ 0.55.

## Eight confirmation gates

All must pass:

1. Binary Macro-F1 ≥ 0.70
2. Deletion recall ≥ 0.45
3. Deletion F1 ≥ 0.40
4. Substitution false-deletion ≤ 0.25
5. Matched Macro-F1 ≥ 0.60
6. Matched deletion F1 ≥ 0.55
7. Every validation speaker with at least 30 deletions has deletion recall ≥ 0.25
8. Three-relation Macro-F1 ≥ 0.40

## Fixed diagnostics

Report each validation speaker's expected-phone rows, deletion support, binary Macro-F1, balanced accuracy, deletion P/R/F1, and correct/substitution false-deletion rates.

For every expected phone report support, deletion support, and deletion P/R/F1. Aggregate D/T/R/L versus all other phones without a minimum-support filter.

Position categories are single-phone (`n=1`), initial (`n>1, i=0`), final (`n>1, i=n-1`), and medial (`n>1, 0<i<n-1`).

Multi-error categories are:

- pure deletion: deletion count > 0 and substitution count = 0
- substitution plus deletion: both counts > 0
- multiple deletion: deletion count > 1

Multiple-deletion words may overlap either of the first two categories.

After predictions are frozen, report diagnostic continuous ROC-AUC and PR-AUC for deletion versus non-deletion and deletion versus substitution. They cannot affect the score, threshold, checkpoint, or result.

Length buckets are 1, 2, 3, 4, 5, and 6+ expected phones. Duration buckets are `<150 ms`, `150–250 ms`, `250–400 ms`, `400–600 ms`, and `≥600 ms`. For each report rows, deletion support, deletion P/R/F1, and predicted deletion among true non-deletion rows. Neither length nor duration may enter prediction.

Report `D_i` mean, median, p10, p25, p75, and p90 for correct, substitution, and deletion.

## Greedy rescue diagnostic

Among frozen R4-4C2 greedy-predicted deletions:

- true-deletion retention is the fraction of its true deletions still predicted deletion by R4-4D1;
- false-deletion rescue is the fraction of its false deletions changed to correct or substitution.

This is diagnostic only.

## Historical comparators

| Method | Binary Macro-F1 | Deletion F1 |
|---|---:|---:|
| Duration-only | 0.668146 | 0.364164 |
| R4-1 | 0.657336 | 0.341612 |
| R4-2A | 0.566997 | 0.197525 |
| R4-3B | 0.503101 | 0.025465 |
| R4-4C2 greedy | 0.555712 | 0.185464 |

No comparator is refit.

## Threshold-transfer reporting

Report the TRAIN threshold and TRAIN/VALIDATION `D_i` medians for correct, substitution, and deletion, plus each absolute median shift. Report locked VALIDATION continuous AUCs. Do not adjust the threshold. Any qualitative stability wording remains descriptive because no additional numerical stability bands are frozen.

## Final classification precedence

Apply exactly:

1. `R4_4D1_SOURCE_VERIFICATION_FAIL` for any source mismatch.
2. `R4_4D1_TRAIN_CALIBRATION_NO_ELIGIBLE_THRESHOLD` when no TRAIN threshold is eligible.
3. `R4_4D1_HYPOTHESIS_DELETION_CONFIRMED` when all eight confirmation gates pass.
4. `R4_4D1_HYPOTHESIS_DELETION_STRONG_PARTIAL` when all of these pass: binary Macro-F1 ≥ 0.65; deletion F1 ≥ 0.35; deletion recall ≥ 0.40; substitution false-deletion ≤ 0.25; matched Macro-F1 ≥ 0.58; matched deletion F1 ≥ 0.50; three-relation Macro-F1 ≥ 0.38; and every validation speaker with at least 30 deletions has recall ≥ 0.20.
5. `R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL` when neither prior validation status applies but validation deletion-vs-nondeletion ROC-AUC ≥ 0.75 and deletion-vs-substitution ROC-AUC ≥ 0.70.
6. `R4_4D1_HYPOTHESIS_SIGNAL_NOT_CONFIRMED` otherwise.

There is no subjective override.

## Prohibitions and closure

After VALIDATION, do not modify the threshold, comparison operators, score family, normalization, length or duration handling, phone/speaker behavior, checkpoint, or model. Do not combine greedy and RAW decisions.

This freeze task calculates no TRAIN threshold, performs no VALIDATION hypothesis scoring, trains no neural model, and accesses no R4 TEST data. R4 TEST remains closed for the future R4-4D1 validation experiment.
