# R5-1B Frozen Development-Validation Transfer Preregistration

Status: `R5_1B_VALIDATION_CONTRACT_FROZEN`

Frozen: 2026-08-15, before any R5-1B validation audio access, inference, score, prediction, or performance metric.

## Research question and evidence role

R5-1B asks whether the fully frozen R5-1A exact CTC addition scorer and its TRAIN-derived threshold transfer to the six unseen VALIDATION speakers without validation calibration.

This is iterative development-validation transfer evidence, not independent final confirmation. Epoch 35 of the R4-4C2 acoustic checkpoint was selected using phone error rate on the same VALIDATION speakers. R5 addition labels did not select that checkpoint, and the R5-1A formula, threshold, metrics, and gates were frozen without VALIDATION addition performance, but the acoustic checkpoint provenance prevents treating R5-1B as an independent final evaluation.

## Frozen identities

- R5-1A contract: `A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB`
- R5-1A preregistration: `2CE9F25B91139B9EA38E2AB552B11C29AA1397B252CE957833D1E3A80D689141`
- R5-1A static manifest: `EED04655AD957A66BF9A13149F812BD0CD74B2D362BDFD4CECD12579DAAA3B5E`
- R5-1A execution manifest: `C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6`
- Static-verified scorer: `4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90`
- V4: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- R4-4C2 checkpoint: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`

The frozen static and execution manifests were fully reverified by SHA-256 and byte size before this preregistration was written.

## Speakers and execution policy

VALIDATION consists only of ABA, HKK, HQTV, LXC, MBMPS, and SVBI. TRAIN metrics remain frozen and cannot be rerun for tuning. TEST speakers ASI, ERMS, SKA, THV, TXHC, and YDCK remain closed.

A future R5-1B execution requires separate authorization. It must import the exact frozen R5-1A scorer and use the frozen model, preprocessing, feature extraction, downsampling, vocabulary, blank handling, alignability semantics, TARGET normalization, hypothesis construction, BEST_INSERT logic, tie-break, and extended-real ROC-AUC behavior. There is no training or checkpoint selection.

## Frozen scorer and decision

For expected sequence `E=[e_1,...,e_N]`:

- `H_KEEP=E`.
- `H_INSERT(E,b,p)=E[:b]+[p]+E[b:]` for every `b` in `0..N` and canonical phone index `p` in `0..39`.
- For an alignable target, `RAW_SCORE(H)` is negative CTC loss and `TARGET_SCORE(H)=RAW_SCORE(H)/max(len(H),1)`.
- `MIN_CTC_STEPS(H)=len(H)+ADJACENT_REPEAT_COUNT(H)` and a target is alignable iff encoder length `T>=MIN_CTC_STEPS(H)`.
- Impossible targets have score negative infinity and remain counted.
- BEST_INSERT maximizes alignable INSERT TARGET_SCORE, breaking exact ties by lower boundary and then lower phone index.
- If BEST_INSERT exists, `A=BEST_INSERT_TARGET_SCORE-KEEP_TARGET_SCORE`; otherwise `A=-infinity`.

The only threshold is the exact TRAIN speaker-LOSO median:

`ROBUST_THETA = 0.7485884030659993`

`A >= ROBUST_THETA` predicts addition. No validation threshold search, calibration, neighbor inspection, or speaker/phone/position threshold is permitted.

## Population and event semantics

Apply exactly the R5-1A runtime population rule: unique manual-word containment, nonempty canonical expected sequence, no unresolved or malformed row in the word, and a reliable frozen MFA word span. Before performance interpretation, report source words, runtime words, positive and negative words, source and runtime addition events, multiple-addition words, mixed substitution/addition and deletion/addition words, and every exclusion reason. The R5-0 count of 296 clean VALIDATION addition events is source support, not a required runtime-event count.

A clean addition is a resolved frozen V4 addition row with expected placeholder `<SIL>`, a canonical observed added phone, and a valid interval. The expected sequence uses resolved correct, substitution, and deletion rows in source order; addition rows do not enter it. An event boundary is the number of expected rows preceding the addition row: boundary 0 is BEFORE_FIRST, a boundary strictly between 0 and N is BETWEEN, and boundary N is AFTER_FINAL.

Events without unique word containment or a nonempty expected sequence remain source-support evidence but cannot enter runtime position or exact-event metrics. Mixed-error positive words remain included. Every event in a multiple-addition word remains a separate true event. The scorer emits at most one event per word, so unmatched additional true events are false negatives.

An exact event match requires the same word, canonical phone, and expected-sequence boundary. Matching is deterministic one-to-one multiset matching. Timestamps are not part of the primary match.

## Metrics and frozen gates

Continuous metrics use the frozen extended-real Mann-Whitney ROC-AUC implementation. Fixed-threshold word metrics include TP, FP, FN, TN, accuracy, balanced accuracy, Binary Macro-F1, and addition precision/recall/F1. False-addition rates are reported for correct-only, substitution-containing non-addition, and deletion-containing non-addition words. Exact-event metrics and BEFORE_FIRST, BETWEEN, and AFTER_FINAL strata are also reported.

All six gates must pass at full precision:

1. Addition vs all non-addition ROC-AUC >= 0.70.
2. Addition vs correct-only ROC-AUC >= 0.70.
3. Fixed-threshold Binary Macro-F1 > 0.548179.
4. Fixed-threshold addition F1 > 0.129246.
5. Correct-only false-addition rate <= 0.054352.
6. Exact-event F1 > 0.026688.

If all pass, the future scientific status is `R5_1B_VALIDATION_TRANSFER_PASS`. If any fail after valid metrics exist, the status is `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED`, with no retuning.

If continuous ranking passes but fixed-threshold decisions fail, the frozen interpretation is that ranking evidence transferred but TRAIN-derived decision calibration did not transfer sufficiently. A validation-optimal threshold must not be calculated.

## Technical stops

Before scientific interpretation, identity, row-accounting, KEEP-alignability, numerical, or implementation failures receive their corresponding frozen technical status. No word may be removed because KEEP is impossible, and no non-finite alignable result may be clipped or repaired.

## TEST closure and anti-drift

TEST paths, audio, inference, examples, scores, predictions, and metrics remain closed. Even a future R5-1B transfer pass authorizes only creation of a dedicated locked TEST preregistration, not immediate TEST inference. A transfer failure does not authorize TEST.

After validation performance is visible, no threshold, score family, normalization, insertion penalty, search method, model, population, gate, or comparison operator may change within R5-1B.

## Contract-stage protocol audit

- Training: NO
- Model inference: NO
- VALIDATION audio accessed: NO
- VALIDATION inference: NO
- VALIDATION performance calculated: NO
- TEST accessed: NO
- R5-1A modified: NO
- R4 modified: NO

