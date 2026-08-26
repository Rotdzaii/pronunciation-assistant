# Phoenix R4 Deletion Research Closure

Final status: **R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED**

## Scientific conclusion

Phoenix deletion detection was **not confirmed** under the preregistered validation requirements. The frozen CNN+BiGRU CTC model produced useful continuous deletion-related acoustic evidence, but its final TARGET-normalized hypothesis decision layer did not achieve the required validation precision/recall/F1 and matched-control gates.

Phoenix demonstrated measurable acoustic evidence for pronunciation deletion through a self-trained CNN+BiGRU CTC sequence model and CTC hypothesis scoring. However, the final frozen validation protocol did not meet the predefined deletion confirmation criteria. Therefore deletion detection is retained as a current research limitation and was not evaluated on the untouched final R4 TEST split.

This conclusion must not be represented as deletion solved, production-ready deletion detection, a validated deletion classifier, or TEST-confirmed deletion performance. Phoenix findings for correct and substitution remain separate from this unconfirmed deletion branch.

## Final development result

The final R4-4D2B TARGET method used the frozen R4-4C2 acoustic checkpoint and the TRAIN-speaker-LOSO threshold `0.16184102947061696`.

| Metric | Result | Frozen requirement | Outcome |
|---|---:|---:|---|
| Binary Macro-F1 | 0.652102 | >= 0.70 | FAIL |
| Deletion recall | 0.373085 | >= 0.45 | FAIL |
| Deletion F1 | 0.331390 | >= 0.40 | FAIL |
| Substitution false-deletion | 0.031907 | <= 0.25 | PASS |
| Matched Macro-F1 | 0.593491 | >= 0.60 | FAIL |
| Matched deletion F1 | 0.485769 | >= 0.55 | FAIL |
| Supported-speaker recall | all passed | >= 0.25 | PASS |
| Three-relation Macro-F1 | 0.465742 | >= 0.40 | PASS |

Only 3 of 8 frozen gates passed. The primary final limitation is **INSUFFICIENT ROBUST DELETION DECISION PERFORMANCE**, not complete absence of acoustic signal: TARGET deletion-vs-nondeletion ROC-AUC was 0.854124 and deletion-vs-substitution ROC-AUC was 0.862234.

## Findings retained

- CNN+BiGRU CTC substantially improved phone-sequence acoustic modeling over CNN-only CTC.
- CTC hypothesis scoring strongly reduced greedy false deletions.
- Continuous deletion-versus-substitution discrimination was meaningfully above chance.
- TRAIN-speaker-held-out TARGET normalization/calibration improved robustness over the original RAW threshold.
- Despite those positive findings, final TARGET validation failed the confirmation contract.

## Historical comparison

| Method | Binary Macro-F1 | Deletion F1 |
|---|---:|---:|
| Duration-only | 0.668146 | 0.364164 |
| R4-1 | 0.657336 | 0.341612 |
| R4-2A | 0.566997 | 0.197525 |
| R4-3B | 0.503101 | 0.025465 |
| R4-4C2 greedy | 0.555712 | 0.185464 |
| R4-4D1 RAW | 0.643750 | 0.321343 |
| R4-4D2B TARGET | 0.652102 | 0.331390 |

R4-4D2B did **not** surpass the duration-only baseline on either headline deletion metric.

## TEST and future policy

The R4 TEST speakers were never accessed. No TEST path was resolved, audio read, posterior computed, hypothesis scored, or metric calculated. TEST was not run because no R4 development candidate passed the final validation confirmation contract.

The current R4 deletion branch is closed. It must not continue through another threshold, normalization, RAW/TARGET blend, recurrent or Transformer-family architecture, duration correction, or phone/speaker-specific threshold. Any future deletion work must begin as a new research generation with a new hypothesis and an appropriate independent evaluation protocol.

Closure actions performed here were documentation and hashing only. Neural training: **NO**. Validation rerun: **NO**. R4 TEST access: **NO**.
