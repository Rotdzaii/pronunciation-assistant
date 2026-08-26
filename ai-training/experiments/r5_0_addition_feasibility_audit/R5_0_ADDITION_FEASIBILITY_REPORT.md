# R5-0 Addition Data & Runtime Feasibility Audit

Research-only feasibility audit. No neural training, threshold tuning, VALIDATION model inference, or TEST audio/inference occurred.

## Frozen identities

- V4: `ai-training\datasets\l2-arctic\metadata\all_speakers_expected_observed_v4.csv` — `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- R4-4C2 checkpoint: `ai-training\experiments\r4_4c2_bigru_ctc_seed42\R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt` — `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`
- R5-0 preregistration: `14CBFADAC0BE35D53C01DC966A030A71F24EC4D86EA88384BC449544CE12AEF7`
- R4 closure manifest: `3E21936C6175F9DEA5FAE3346E96EECD3AEF18732072C42B0B2FCAE4161D6174`

## Annotation semantics

An addition is a dedicated manual `phones` IntervalTier row encoded as `sil,<observed_phone>,a`. `sil` is a placeholder rather than an expected canonical phone; the observed canonical label is the added phone. The interval has start/end timestamps. Multiple additions and additions mixed with substitution/deletion are retained. Malformed added-phone labels remain unresolved and are reported, not silently repaired.

The insertion boundary is the number of expected-sequence phones preceding the addition interval: 0 is BEFORE_FIRST, N is AFTER_FINAL, and an interior boundary is BETWEEN.

## Data support

- Global clean additions: **1044** / 118449 eligible relation rows (0.881392%).
- Tagged addition annotations: 1092; excluded/invalid: 48.
- TRAIN: 423 additions; 12 speakers with additions; 12 speakers with >=5.
- VALIDATION support only: 296 additions; 6 speakers with additions; 6 speakers with >=5.
- TEST aggregate support only: 325 additions; 6 speakers with additions; 6 speakers with >=5.
- TRAIN mapped positions: BEFORE_FIRST=50, BETWEEN=183, AFTER_FINAL=127.
- Imbalance: **severe**; correct:addition=97.343:1, substitution:addition=11.840:1, deletion:addition=3.274:1.

## Feasibility gates

- TRAIN support: **PASS**
- VALIDATION support: **PASS**
- TEST aggregate support: **PASS**

## Frozen TRAIN CTC greedy insertion baseline

- Word precision/recall/F1: 0.088235 / 0.241486 / 0.129246
- Binary Macro-F1: 0.548179
- Exact event precision/recall/F1: 0.018240 / 0.049708 / 0.026688
- Correct-only false insertion word rate: 0.054352
- ADDITION_INSERTION_RATE_DELTA: 0.187134

### Additive technical correction 001

The initial post-run audit found two reporting-scope defects after the primary metrics were visible: the hallucinated-phone ranking covered all eligible TRAIN words instead of correct-only words, and the added-phone distribution covered only manual-word-mapped events instead of all clean TRAIN source events. The original manifest and affected artifacts were preserved. A frozen additive correction repeated the identical TRAIN-only inference solely to recover the missing subgroup breakdown and reproduced the primary word confusion (TP=78, FP=806, FN=245, TN=15,453), event confusion (TP=17, FP=915, FN=325), and delta exactly. It recovered all 423 clean TRAIN added-phone labels and the correct-only hallucination ranking. No score, gate, status, VALIDATION evidence, or TEST evidence changed.

## Runtime MFA and CTC provenance

Current runtime MFA does not expose an explicit arbitrary extra-phone slot: it aligns the canonical prompt transcript, and its parser only returns MFA-emitted intervals. It may still provide word context, but explicit added-phone identity/location requires a separate acoustic sequence mechanism. The frozen CTC insertion alignment does provide a deterministic phone-plus-boundary representation.

Reuse classification: **R5_CTC_REUSE_DEVELOPMENT_ALLOWED_CONFIRMATION_REQUIRES_NEW_PROTOCOL**. R4 excluded addition-containing words and did not use addition labels, so TRAIN-only diagnostic reuse is acceptable. Epoch 35 was nevertheless selected on these VALIDATION speakers by PER, so future work on the same VALIDATION split is iterative development rather than independent confirmation.

## Final decision

**R5_0_PASS_EXISTING_CTC_FEASIBLE**

This is a feasibility outcome, not confirmation that Phoenix detects additions.

## Protocol closure

- Training: NO
- Threshold tuning: NO
- VALIDATION performance consumed: NO
- TEST audio accessed: NO
- TEST inference: NO
- TEST performance consumed: NO
- R4 modified: NO
