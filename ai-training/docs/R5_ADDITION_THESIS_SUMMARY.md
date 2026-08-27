# R5 Pronunciation Addition Research — Thesis Summary

## Research objective

The Phoenix R5 research program investigated whether an English phone added by a learner could be detected relative to the expected canonical phone sequence of a word. Unlike substitution, an addition has no corresponding expected-phone slot. The research therefore had to establish both whether addition annotations were sufficiently supported and whether a runtime acoustic representation could identify an inserted phone and its boundary.

R5 used speaker-disjoint L2-ARCTIC splits. TRAIN supported development and speaker leave-one-out calibration. VALIDATION supported one frozen iterative transfer evaluation. The six-speaker TEST split remained untouched.

## Why addition is difficult

Addition is rare: the global clean-event prevalence was 0.881392%. A detector can therefore create many false alarms even when it recovers some true additions. Localization is also structurally difficult because MFA aligns the canonical transcript and cannot create an arbitrary extra-phone slot. Finally, the selected scorer produces one best insertion per word, while some words contain multiple true additions.

## Data and runtime feasibility

R5-0 found 1,044 clean addition events: 423 in TRAIN, 296 in VALIDATION, and 325 in aggregate TEST support metadata. Every preregistered split-support gate passed.

A frozen greedy CTC diagnostic on TRAIN provided directional but weak evidence. Word-level precision, recall, and F1 were 0.088235, 0.241486, and 0.129246; Binary Macro-F1 was 0.548179; exact-event F1 was 0.026688; and the correct-only false insertion rate was 0.054352. The insertion-positive rate difference between true-addition and correct-only words was +0.187134. Thus greedy CTC reacted more often to true additions, but direct decoded insertions were too noisy for a reliable detector.

## Exact CTC insertion-hypothesis scoring

R5-1 replaced greedy omission/insertion interpretation with an explicit sequence comparison. For expected sequence `E`, the model scored `KEEP = E` and every single-phone `INSERT(E, b, p)` hypothesis, where `b` is one of the expected-sequence insertion boundaries and `p` is one of the 40 canonical phones. The best TARGET-normalized insertion score was compared with the TARGET-normalized KEEP score:

```text
A = TARGET_SCORE(BEST_INSERT) - TARGET_SCORE(KEEP)
```

The initial frozen R5-1 execution stopped before performance evaluation because 196 insertion hypotheses across 65 words were CTC-impossible. With `zero_infinity=True`, directly interpreting the returned zero loss could give an impossible sequence a misleading score. This was a technical contract failure, not a negative scientific result.

R5-1A was therefore created as a separate pre-metric alignability-safe contract. It defined:

```text
MIN_CTC_STEPS(H) = len(H) + ADJACENT_REPEAT_COUNT(H)
```

When encoder length `T` is smaller than this minimum, the target is impossible and receives `TARGET_SCORE = -infinity`. Synthetic static verification confirmed alignability logic, finite CTC scoring, tie behavior, extended-real ROC-AUC, threshold generation, serialization, and determinism before real-data evaluation.

## TRAIN development result

R5-1A evaluated 16,582 runtime-evaluable TRAIN words: 323 positives and 16,259 negatives. Twelve-fold TRAIN-speaker leave-one-speaker-out calibration produced out-of-fold decisions.

| Metric | TRAIN value |
|---|---:|
| Addition vs non-addition ROC-AUC | 0.7734833025081417 |
| Addition vs correct-only ROC-AUC | 0.8023528214095370 |
| Binary Macro-F1 | 0.5551978767901391 |
| Addition precision | 0.1005665722379603 |
| Addition recall | 0.2198142414860681 |
| Addition F1 | 0.1379980563654033 |
| Correct-only false-addition rate | 0.03491295938104449 |
| Exact-event F1 | 0.04389312977099236 |

All six preregistered TRAIN development gates passed. The median LOSO threshold was frozen at `0.7485884030659993` without using VALIDATION addition performance.

## Iterative VALIDATION transfer

R5-1B applied the unchanged scorer and threshold exactly once to 7,955 runtime-evaluable VALIDATION words: 227 positives and 7,728 negatives. No validation threshold search or recalibration was permitted.

| Metric | VALIDATION value |
|---|---:|
| Addition vs non-addition ROC-AUC | 0.7512808848879525 |
| Addition vs correct-only ROC-AUC | 0.7717072309229656 |
| Binary Macro-F1 | 0.5438876363494940 |
| Addition precision | 0.0933521923620934 |
| Addition recall | 0.2907488986784141 |
| Addition F1 | 0.1413276231263383 |
| Correct-only false-addition rate | 0.0832481079975455 |
| Exact-event F1 | 0.0378151260504202 |

Four of six transfer gates passed. Binary Macro-F1 failed to exceed the frozen greedy comparator of 0.548179, and correct-only false-addition rate exceeded the maximum 0.054352. Final transfer status was **R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED**.

## Failure interpretation

The continuous AUC values remained useful on VALIDATION, so the acoustic ranking evidence did transfer. The fixed threshold, however, produced substantially more false additions on correct words. A post-hoc frozen-artifact audit found that five of six validation speakers exceeded the historical correct-only FAR reference. Median scores shifted upward between TRAIN and VALIDATION by approximately +0.1508 for all words, +0.1281 for true additions, and +0.1488 for correct-only words. Correct-only cases formed 407 of the 641 false-positive words.

The cautious interpretation is **MIXED_CALIBRATION_AND_CLASS_OVERLAP**: score location varied across speakers, while addition and non-addition distributions also overlapped materially. It would be incorrect to attribute the failure only to threshold calibration or to claim that the model contained no addition-related evidence.

### Plain-language explanation

The model learned a useful signal for ranking words by how likely they were to contain an added phone. That ranking remained useful for new speakers. However, the score scale changed between speakers and addition and non-addition examples still overlapped. The same TRAIN threshold therefore labeled too many correct VALIDATION words as additions.

### Giải thích ngắn bằng tiếng Việt

> Mô hình có học được tín hiệu của âm thêm và vẫn xếp hạng khá tốt khi đổi người nói. Tuy nhiên cùng một ngưỡng từ tập TRAIN lại báo nhầm addition quá nhiều trên VALIDATION. Kiểm tra sau đó cho thấy vừa có lệch điểm giữa người nói, vừa có sự chồng lấn giữa addition và các trường hợp không phải addition.

## Later relation-evidence generations

R5-2B compared BEST_INSERT against the strongest KEEP, single-SUB, or single-DELETE hypothesis. This hard competition strongly reduced substitution- and deletion-negative false additions, but it also suppressed true Addition evidence, reduced continuous discrimination, and triggered compensating LOSO thresholds. Its frozen result passed only 2 of 8 gates and was not confirmed. R5-2C classified the mechanism as `MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION`.

R5-3A then retained the three evidence channels separately: `A = BEST_INSERT - KEEP`, `S = BEST_SUB - KEEP`, and `D = BEST_DELETE - KEEP`. A fixed balanced linear classifier was evaluated once in 12-fold TRAIN-speaker LOSO over 16,582 words. It recovered Addition/all AUC 0.7825848108511275, Addition/correct AUC 0.8269990598295717, Binary Macro-F1 0.5583786087141792, Addition F1 0.14285714285714285, correct-only FAR 0.02446808510638298, and exact-event F1 0.044044044044044044. However, substitution FAR was 0.05257856567284448 and deletion FAR was 0.09907341411261582, so G6 and G7 failed and no robust threshold was authorized.

The standardized median coefficients were positive for A (0.9953044129046558), S (0.1523077166275505), and D (0.5849599738415144), with the same positive sign in all 12 folds. This is consistent with the failure to retain SUB/DELETE suppression, but it does not establish a causal relationship between those channels and Addition.

## INSERT-candidate recoverability

R5-4B successfully materialized the complete frozen TRAIN INSERT landscape: 2,977,040 candidates for 16,582 words. All structural counts reproduced, and candidate-derived BEST_INSERT identity and exact winning score matched the frozen scorer for every word. This was a technical/provenance pass, not a scientific Addition-performance result.

R5-4A then measured exact-event recoverability using only those frozen candidates and already-frozen TRAIN truth. All 304 single-addition words mapped exactly. Top-1, Top-3, Top-5, and Top-10 recoverability were 0.134868, 0.240132, 0.279605, and 0.388158; median truth rank was 19 and mean rank was 57.10. Only the mapping gate passed, yielding 1/4 and `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_NOT_CONFIRMED`.

The candidate family therefore contains partial localization information, but the correct event is not generally near enough to the top to satisfy the frozen feasibility criteria. BETWEEN additions were more recoverable than word-edge additions, and phone and speaker variation was substantial, but these are descriptive TRAIN findings rather than production rules. The evidence does not justify a new generation based only on top-k, marginalization, or alternative pooling of the same frozen INSERT family.

## Limitations

- Addition prevalence was extremely low, and absolute precision remained low.
- The scorer returned only one `BEST_INSERT` event per word, structurally limiting recall for multiple-addition words.
- MFA could not explicitly localize arbitrary added phones.
- Speaker-dependent score calibration remained unstable.
- Addition and non-addition score distributions retained intrinsic overlap.
- The R4-4C2 acoustic checkpoint used current VALIDATION speakers for PER-based epoch selection.
- R5-1B therefore provided iterative transfer evidence, not independent final confirmation.
- R5-1C's mixed-failure interpretation was post-hoc and exploratory.

## Scientific contribution and claim boundary

R5 contributed a verified alignability-safe exact CTC insertion scorer, evidence that explicit sequence hypotheses improve on greedy insertion diagnostics under the frozen TRAIN criteria, and a transparent demonstration that continuous ranking and deployable fixed-threshold decisions are separate scientific questions.

Allowed claims:

- “Addition-related continuous ranking evidence transferred to the iterative VALIDATION split.”
- “Fixed-threshold addition decision transfer was not confirmed.”

Unsupported claims include “Phoenix accurately detects additions,” “R5 addition detection was confirmed,” and “only threshold calibration caused the failure.”

R5-1, R5-2, R5-3A, and R5-4A are closed not-confirmed generations or diagnostics. R5-4B is closed as a successful technical materialization stage, not scientific confirmation. The latest diagnostic closure status is **R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED**. R5 VALIDATION was consumed by R5-1B and was not accessed in R5-2, R5-3A, or R5-4. R5 TEST was not accessed. Any future addition work must use a fundamentally different, separately preregistered mechanism that acknowledges consumed VALIDATION and preserves TEST until independently authorized.

## Appendix — frozen experiment and identity record

| Item | Frozen value |
|---|---|
| R5-0 status | `R5_0_PASS_EXISTING_CTC_FEASIBLE` |
| R5-1 technical status | `R5_1_EXECUTION_TECHNICAL_FAILURE_CTC_ALIGNABILITY` |
| R5-1A static status | `R5_1A_STATIC_VERIFICATION_PASS` |
| R5-1A TRAIN status | `R5_1A_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS` |
| R5-1B VALIDATION status | `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED` |
| R5-1C status | `R5_1C_POSTHOC_FAILURE_AUDIT_COMPLETE` |
| R5-1D closure | `R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED` |
| R5-2B TRAIN status | `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED` |
| R5-2C interpretation | `MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION` |
| R5-3A TRAIN status | `R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED` |
| R5-3A closure | `R5_3A_EVIDENCE_SEPARATION_GENERATION_CLOSED_NOT_CONFIRMED` |
| R5-3A gates | `6 / 8 PASS; G6/G7 FAIL` |
| R5-3A robust threshold | `R5_3A_ROBUST_THETA_NOT_AUTHORIZED` |
| R5-4B closure | `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS` |
| R5-4B materialization gates | `10 / 10 PASS` |
| R5-4A closure | `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED` |
| R5-4A feasibility gates | `1 / 4 PASS; G2/G3/G4 FAIL` |
| Frozen threshold | `0.7485884030659993` |
| R5-1A contract SHA | `A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB` |
| Frozen scorer SHA | `4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90` |
| R5-1A execution manifest SHA | `C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6` |
| R5-1B contract SHA | `BD169175C0777B1C37506E95350CB6AD90A992045527396DDE5DA4419779AC4A` |
| R5-1B execution manifest SHA | `B1E053F8B80F6EDCBB4195BA632B1215E9FA42A7AA0AF5C2950D92AFF6E0771E` |
| R5-1C manifest SHA | `67BCB0D2655F68100E7BB657FD193E63300802B398D395DBA5EF6D120E4C7241` |
| Closure manifest SHA | `C8E71EDE56902D594A60F1194ABF8A72AB0A7EFBE4F212F57BC71FCEF21B1D69` |
| R5-2B execution manifest SHA | `37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387` |
| R5-2C manifest SHA | `C18621208A35074B7681EFC95261B73147B4A6E3716F17AF0AB0471F8B328F90` |
| R5-3A execution manifest SHA | `1EC6DED9ECA6617AE683C9EF316B4435A11F6898B380E4592D779061BF73CE51` |
| R5-4B execution manifest SHA | `FD1E4E66168654EC54778A489E0B443BD9C691DD1392233C4284CDF6CDF07B11` |
| R5-4A resumed-audit manifest SHA | `328E2732A3705593EA29D6C2682956E3B5D19F2175B4F7E2C55F888175A15046` |

Closure-documentation protocol: training **NO**; inference **NO**; classifier fitting **NO**; new metrics **NO**; threshold search/recalibration **NO**; VALIDATION access **NO**; TEST access **NO**.
