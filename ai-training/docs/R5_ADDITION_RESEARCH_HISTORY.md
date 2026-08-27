# Phoenix R5 Addition Research History

## 1. Research objective

R5 investigated whether Phoenix could detect an added canonical phone that is absent from a word's expected phone sequence. The program separated three questions:

1. whether L2-ARCTIC contains enough clean addition evidence for speaker-disjoint research;
2. whether an acoustic sequence model produces useful continuous evidence for an insertion;
3. whether a decision threshold derived from TRAIN speakers transfers robustly to unseen speakers.

The latest completed candidate-recoverability diagnostic is closed with final status:

`R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED`

R5-4B is separately closed as `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS`. These statuses preserve useful evidence from every generation without treating a technical materialization pass as scientific confirmation. R5-1 retained continuous insertion-ranking evidence but failed fixed-threshold VALIDATION transfer. R5-2 reduced substitution/deletion false additions but damaged overall Addition discrimination. R5-3A recovered aggregate TRAIN discrimination but failed the preregistered substitution- and deletion-negative FAR gates. R5-4A then found partial localization signal but insufficient general exact-event recoverability. No R5 addition generation is eligible for TEST or production claims.

## 2. Dataset, splits, and evidence policy

R5 used the frozen V4 L2-ARCTIC expected/observed metadata. A clean addition is represented by a resolved V4 addition relation whose expected phone is the `<SIL>` placeholder and whose observed phone is a canonical added phone. Its insertion boundary is the number of expected-phone rows preceding that addition row.

| Split | Speakers | R5 role |
|---|---|---|
| TRAIN | BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA | Feasibility, frozen scorer development, and 12-fold speaker LOSO calibration |
| VALIDATION | ABA, HKK, HQTV, LXC, MBMPS, SVBI | Exactly one iterative fixed-threshold transfer evaluation in R5-1B |
| TEST | ASI, ERMS, SKA, THV, TXHC, YDCK | Untouched candidate independent holdout; never accessed |

R5-0 was allowed to inspect only aggregate TEST support counts from frozen metadata. No TEST sample list, phone distribution, position distribution, audio path, inference, prediction, or performance was consumed. R5-1A used TRAIN only. R5-1B consumed addition performance on VALIDATION exactly once under a frozen threshold. R5-1C used only already frozen TRAIN and VALIDATION artifacts.

The R4-4C2 acoustic checkpoint had selected epoch 35 using PER on the same current VALIDATION speakers. Consequently, R5-1B is iterative transfer evidence, not fully independent confirmation, even though R5 addition labels did not select the R4 checkpoint and the R5 scorer and threshold were frozen without VALIDATION addition performance.

## 3. Research timeline

The timeline preserves the actual hypothesis, evidence, and decision sequence. The final method was not treated as obvious in advance.

| Stage | Hypothesis or purpose | Main evidence | Decision |
|---|---|---|---|
| R5-0 | Are addition data, runtime representation, and frozen CTC insertion behavior feasible? | 1,044 global clean events; all split support gates passed; greedy word F1 0.129246; exact-event F1 0.026688; insertion-rate delta +0.187134 | `R5_0_PASS_EXISTING_CTC_FEASIBLE` |
| R5-1 contract | Can exact TARGET-normalized CTC INSERT likelihood outperform greedy insertion? | Frozen KEEP versus all single INSERT hypotheses and TRAIN-speaker LOSO protocol | Authorize one TRAIN-only execution |
| R5-1 execution | Execute the frozen initial scorer | 196 CTC-impossible INSERT hypotheses across 65 words before metrics | `R5_1_EXECUTION_TECHNICAL_FAILURE_CTC_ALIGNABILITY`; no scientific result |
| R5-1A contract | Define mathematically safe impossible-target behavior before performance | `MIN_CTC_STEPS`, impossible score `-infinity`, unchanged population/gates | New named pre-metric technical contract |
| R5-1A static verification | Prove implementation semantics on synthetic inputs | Alignability, tie, extended-real AUC, threshold, serialization, and determinism tests passed | `R5_1A_STATIC_VERIFICATION_PASS` |
| R5-1A TRAIN execution | Test continuous discrimination, LOSO decisions, and event localization | AUC 0.773483/0.802353; Binary MF1 0.555198; addition F1 0.137998; exact-event F1 0.043893 | 6/6 gates; `R5_1A_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS` |
| R5-1B contract | Freeze one no-recalibration transfer evaluation | Fixed scorer and `ROBUST_THETA = 0.7485884030659993`; same six gates | Authorize exactly one iterative VALIDATION execution |
| R5-1B execution | Does the TRAIN-derived threshold transfer? | AUC 0.751281/0.771707; Binary MF1 0.543888; correct-only FAR 0.083248 | 4/6 gates; `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED` |
| R5-1C | Describe the already frozen failure without rescue | Broad FAR inflation, score-location shifts, and false positives across cohorts | Post-hoc `MIXED_CALIBRATION_AND_CLASS_OVERLAP` |
| R5-1D | Close the current generation while retaining positive and negative evidence | Verified closure package and explicit future/test policy | `R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED` |
| R5-2B | Test hard relation competition: `BEST_INSERT - max(KEEP, BEST_SUB, BEST_DELETE)` | SUB FAR 0.017123 and DELETE FAR 0.006415, but AUC 0.687841/0.669239, Binary MF1 0.530403, and Addition F1 0.086379 | 2/8 gates; `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED` |
| R5-2C | Diagnose the frozen R5-2B failure | Relation competition suppressed confounded negatives but also true Additions; LOSO thresholds compensated downward | `MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION` |
| R5-3A | Keep Addition, substitution, and deletion evidence separate as `[A,S,D]` and fit one fixed balanced linear model in TRAIN-speaker LOSO | Aggregate discrimination and decisions recovered, but SUB FAR 0.052579 and DELETE FAR 0.099073 failed their gates | 6/8 gates; `R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED` |
| R5-3A closure | Preserve the valid negative result without rerun or retuning | No robust threshold; VALIDATION not accessed; TEST untouched | `R5_3A_EVIDENCE_SEPARATION_GENERATION_CLOSED_NOT_CONFIRMED` |
| R5-4B | Materialize every frozen TRAIN INSERT candidate without truth-based selection or performance analysis | 16,582 words and 2,977,040 candidates persisted; exact BEST_INSERT identity and score reproduced for every word | 10/10 provenance gates; `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS` |
| R5-4A resumed | Test whether the exact truth event is recoverable near the top of the unchanged INSERT ranking | Mapping 304/304; Top-5 0.279605; Top-10 0.388158; median truth rank 19 | 1/4 feasibility gates; `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED` |

## 4. R5-0 — Addition Data and Runtime Feasibility Audit

### 4.1 Hypothesis

Before building a detector, R5-0 asked whether clean addition annotations were sufficiently supported, whether addition position could be represented relative to the expected sequence, whether frozen greedy CTC contained directional insertion evidence, and whether MFA could expose an added-phone location.

### 4.2 Data support

| Quantity | Frozen value |
|---|---:|
| Global clean addition events | 1,044 |
| TRAIN clean source events | 423 |
| VALIDATION clean source events | 296 |
| TEST clean source events, aggregate support only | 325 |
| Global addition prevalence | 0.881392% |

All TRAIN, VALIDATION, and aggregate TEST support gates passed. Addition evidence existed across the speaker-disjoint splits and covered the required insertion-position classes. These were feasibility findings, not detection-performance claims.

### 4.3 Frozen greedy TRAIN comparator

| Metric | Value |
|---|---:|
| Word-level precision | 0.088235 |
| Word-level recall | 0.241486 |
| Word-level addition F1 | 0.129246 |
| Binary Macro-F1 | 0.548179 |
| Exact-event F1 | 0.026688 |
| Correct-only false insertion rate | 0.054352 |
| `ADDITION_INSERTION_RATE_DELTA` | +0.187134 |

Greedy CTC insertions were more frequent on true-addition words than on correct-only words, establishing directional signal. Absolute precision and exact event localization remained weak, so direct greedy insertion was too noisy.

MFA could align expected transcript phones but could not explicitly create an arbitrary extra-phone slot. R5 therefore retained expected-sequence boundaries and CTC hypotheses as the viable representation. Final status: **R5_0_PASS_EXISTING_CTC_FEASIBLE**.

## 5. R5-1 — Initial exact CTC insertion hypothesis scoring

### 5.1 Frozen hypothesis

For expected canonical sequence `E = [e1, ..., eN]`, the scorer retained `KEEP = E` and enumerated every single insertion:

```text
INSERT(E, b, p) = E[:b] + [p] + E[b:]
b = 0 ... N
p = 0 ... 39
```

It used the R4 TARGET normalization, compared the highest-scoring INSERT with KEEP, and froze a 12-fold TRAIN-speaker LOSO threshold protocol. No new acoustic model was introduced.

### 5.2 Pre-metric technical stop

The attempted execution discovered 196 INSERT hypotheses across 65 words that were impossible under CTC's minimum-step requirement. The initial contract retained `zero_infinity=True`; for an impossible target this can replace infinite loss with zero loss, which would be misleading after negating the loss into a hypothesis score.

Execution stopped before TRAIN performance metrics or gate decisions. Final status:

`R5_1_EXECUTION_TECHNICAL_FAILURE_CTC_ALIGNABILITY`

This is a technical-stop generation, not a model or performance failure. Its contract and stop evidence remain preserved.

## 6. R5-1A — Alignability-safe exact scoring

### 6.1 New pre-metric contract

Because changing impossible-target behavior would alter the original frozen contract, R5-1 was not edited. R5-1A was created as a new named contract before performance was observed.

For target `H`:

```text
ADJACENT_REPEAT_COUNT(H) = count of i where H[i] == H[i-1]
MIN_CTC_STEPS(H) = len(H) + ADJACENT_REPEAT_COUNT(H)
ALIGNABLE iff T >= MIN_CTC_STEPS(H)
```

An impossible hypothesis has probability zero and therefore:

```text
RAW_SCORE(H) = -infinity
TARGET_SCORE(H) = -infinity
```

For an alignable hypothesis:

```text
RAW_SCORE(H) = -CTCLoss(log_softmax(logits), H, blank=40,
                        reduction="none", zero_infinity=True)
TARGET_SCORE(H) = RAW_SCORE(H) / max(len(H), 1)
```

`BEST_INSERT` considers only alignable candidates, with lower insertion boundary and then lower canonical-phone index as exact tie-breaks. The frozen continuous score is:

```text
A = TARGET_SCORE(BEST_INSERT) - TARGET_SCORE(KEEP)
```

If all INSERT candidates are impossible, no insertion event exists and `A = -infinity`. The word remains in the binary population. The scorer emits at most one `BEST_INSERT` event per word.

### 6.2 Static verification

Before real audio inference, synthetic tests verified the minimum-step formula, impossible-target interception, finite alignable scoring, BEST_INSERT tie rules, all-INSERT-impossible and KEEP-impossible behavior, the `zero_infinity=True` regression, deterministic extended-real Mann-Whitney ROC-AUC, finite threshold candidates, standards-compatible `-infinity` serialization, and determinism. No Phoenix performance data was used.

Final static status: **R5_1A_STATIC_VERIFICATION_PASS**.

### 6.3 Frozen TRAIN development

The runtime-evaluable TRAIN population contained 16,582 words: 323 addition-positive words and 16,259 negatives. Multiple-addition and mixed-error words remained included.

| Metric | TRAIN value |
|---|---:|
| Addition vs non-addition ROC-AUC | 0.7734833025081417 |
| Addition vs correct-only ROC-AUC | 0.8023528214095370 |
| OOF Binary Macro-F1 | 0.5551978767901391 |
| Addition precision | 0.1005665722379603 |
| Addition recall | 0.2198142414860681 |
| Addition F1 | 0.1379980563654033 |
| Correct-only false-addition rate | 0.03491295938104449 |
| Exact-event F1 | 0.04389312977099236 |

All six preregistered TRAIN development gates passed. The ordinary float64 median of the 12 LOSO fold thresholds was frozen as:

`ROBUST_THETA = 0.7485884030659993`

Final status: **R5_1A_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS**. This was development evidence and did not authorize TEST.

## 7. R5-1B — Fixed-threshold iterative VALIDATION transfer

### 7.1 Frozen transfer question

R5-1B asked whether the unchanged R5-1A scorer and TRAIN-derived `ROBUST_THETA` transferred to the six VALIDATION speakers without calibration, threshold search, model changes, or scoring changes. Exactly one validation execution was authorized.

The runtime-evaluable population contained 7,955 words: 227 positives and 7,728 negatives.

### 7.2 Frozen result

| Metric | VALIDATION value |
|---|---:|
| Addition vs non-addition ROC-AUC | 0.7512808848879525 |
| Addition vs correct-only ROC-AUC | 0.7717072309229656 |
| Fixed threshold | 0.7485884030659993 |
| Binary Macro-F1 | 0.5438876363494940 |
| Addition precision | 0.0933521923620934 |
| Addition recall | 0.2907488986784141 |
| Addition F1 | 0.1413276231263383 |
| Correct-only false-addition rate | 0.0832481079975455 |
| Exact-event F1 | 0.0378151260504202 |

Four of six gates passed. The failed gates were:

- G3: Binary Macro-F1 had to exceed 0.548179.
- G5: correct-only false-addition rate had to be at most 0.054352.

The continuous AUCs remained above their gates, while the fixed threshold produced too many false addition decisions, especially on correct words. Final status:

`R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED`

No validation-optimal threshold was calculated and no recalibration occurred.

## 8. R5-1C — Post-hoc frozen failure audit

R5-1C was descriptive and exploratory. It used only the frozen R5-1A and R5-1B score/prediction artifacts. It ran no model inference, generated no new predictions, searched no threshold, and did not change the R5-1B result.

Five of six VALIDATION speakers exceeded the historical correct-only FAR reference. Aggregate median score locations shifted by approximately +0.1508 for all words, +0.1281 for true additions, and +0.1488 for correct-only words.

The 641 frozen false-positive words were distributed as follows:

| Negative cohort | False positives | Share |
|---|---:|---:|
| Correct-only | 407 | 63.49% |
| Substitution-only | 167 | 26.05% |
| Deletion-only | 44 | 6.86% |
| Substitution + deletion | 23 | 3.59% |

The evidence was not consistent with a single isolated speaker or a calibration-only explanation. The frozen exploratory interpretation was:

`MIXED_CALIBRATION_AND_CLASS_OVERLAP`

This post-hoc label does not replace the preregistered R5-1B scientific status.

Audit status: **R5_1C_POSTHOC_FAILURE_AUDIT_COMPLETE**.

## 9. R5-1D — Current generation closure

R5-1D verified all upstream identities, preserved the complete causal history, and closed the current scoring generation without inference, training, recalibration, threshold search, or TEST access.

Authoritative closure status:

`R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED`

The closure preserves three distinct conclusions:

1. **Data feasibility was sufficient.** R5-0 established adequate clean addition support for the research program.
2. **Continuous addition evidence was supported.** Exact CTC insertion-hypothesis scoring retained useful addition-ranking evidence from TRAIN to iterative VALIDATION.
3. **Fixed-threshold decision transfer was not confirmed.** The TRAIN-derived global threshold did not provide adequate false-addition control on unseen VALIDATION speakers.

Preferred interpretation:

> The frozen CTC scorer retained meaningful addition-ranking evidence, but its score distributions shifted across speakers and addition/non-addition overlap remained substantial. Consequently, the TRAIN-derived global threshold did not transfer with adequate false-addition control.

The current generation is not eligible to open TEST.

## 10. What R5 established and did not establish

Positive findings retained:

- addition annotations are sufficiently supported across speakers;
- frozen CTC can represent explicit insertion hypotheses;
- alignability-safe exact CTC scoring is technically valid;
- continuous addition ranking was above chance on TRAIN and VALIDATION;
- exact hypothesis scoring exceeded the frozen greedy comparators on all TRAIN development gates;
- exact-event evidence exceeded the greedy comparator;
- correct-word false additions were controlled on TRAIN;
- the transfer failure became visible only when applying the fixed global threshold to VALIDATION.

Claims not supported:

- Phoenix accurately detects additions;
- R5 addition detection was confirmed;
- the scorer is production-ready;
- TEST confirmed addition performance;
- calibration alone explains the failure.

Allowed concise claims:

- “Addition-related continuous ranking evidence transferred to the iterative VALIDATION split.”
- “Fixed-threshold addition decision transfer was not confirmed.”

## 11. Plain-language explanation

The model learned a useful acoustic ranking signal for possible added phones, and that ranking remained useful on new speakers. However, the score scale moved between speakers and addition and non-addition cases still overlapped substantially. At the same fixed TRAIN threshold, too many correct validation words were therefore labeled as additions.

### Giải thích ngắn bằng tiếng Việt

> Mô hình có học được tín hiệu của âm thêm và vẫn xếp hạng khá tốt khi đổi người nói. Tuy nhiên cùng một ngưỡng từ tập TRAIN lại báo nhầm addition quá nhiều trên VALIDATION. Kiểm tra sau đó cho thấy vừa có lệch điểm giữa người nói, vừa có sự chồng lấn giữa addition và các trường hợp không phải addition.

## 12. Limitations

- Addition is extremely imbalanced; global prevalence was only 0.881392%.
- Absolute addition precision remained low on TRAIN and VALIDATION.
- The scorer emits one `BEST_INSERT` maximum per word.
- Multiple-addition words cannot have all true events recovered, structurally limiting event recall.
- MFA cannot explicitly localize arbitrary added-phone slots.
- The R4-4C2 checkpoint used current VALIDATION speakers for PER-based epoch selection.
- R5-1B is iterative transfer evidence, not independent final confirmation.
- Score calibration varies across speakers.
- Material intrinsic addition/non-addition overlap remains.
- R5-1C interpretation is post-hoc and exploratory.

## 13. VALIDATION and TEST status

R5 VALIDATION addition performance was consumed in R5-1B. Future R5 work must acknowledge that history and must not present this split as untouched independent confirmation for a scorer influenced by R5-1 evidence.

R5 TEST speakers ASI, ERMS, SKA, THV, TXHC, and YDCK remain untouched. No TEST audio was read, no TEST inference was run, and no TEST performance was consumed. Preserving TEST was intentional because R5-1B did not pass its frozen transfer contract.

## 14. Future research policy

R5-1, R5-2, and R5-3A are closed. They must not continue through changed thresholds, validation-optimal or per-speaker/phone/boundary rules, altered feature fusion, changed equality semantics, RAW/TIME fallback, speaker normalization, penalties, beam search, alternate checkpoints/seeds, retraining, or selective removal of difficult cohorts.

This does not close all future addition research permanently. Any future generation must:

1. use a new stage/generation name;
2. state a new scientific hypothesis motivated by the complete frozen R5-1/R5-2/R5-3A evidence;
3. specify whether it targets calibration robustness, class separability, or both;
4. preregister before evaluation;
5. acknowledge consumed VALIDATION;
6. preserve untouched TEST until a new protocol legitimately reaches it;
7. freeze numeric gates and stop rules before new evaluation metrics.

## 15. R5-2 hard relation competition

R5-2 tested whether an INSERT should count only when it beat the strongest same-word non-addition explanation in the same exact CTC likelihood space. The score subtracted the maximum of KEEP, one-phone SUB, and one-phone DELETE evidence from BEST_INSERT. On the frozen 16,582-word TRAIN population, only G6 and G7 passed. Substitution-negative FAR fell from 0.05016116035455278 to 0.017123287671232876 and deletion-negative FAR fell from 0.03349964362081254 to 0.006414825374198147. However, Addition/all and Addition/correct AUC fell to 0.6878413803490975 and 0.669239360205041, Binary Macro-F1 fell to 0.5304030323273805, Addition F1 fell to 0.08637873754152824, and exact-event F1 fell to 0.03253796095444685.

The frozen R5-2C audit found that SUB or DELETE won the non-addition competition for 291 of 323 true Addition words. Forty R5-1A true positives became R5-2B false negatives, while threshold compensation introduced 249 correct-word TN-to-FP transitions. BEST_INSERT identity remained unchanged for all 16,582 words. The supported post-hoc classification is `MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION`: relation evidence was useful, but hard-max subtraction was not confirmed as an effective fusion rule.

## 16. R5-3A evidence-separated linear fusion

R5-3A retained three independent features: `A = BEST_INSERT - KEEP`, `S = BEST_SUB - KEEP`, and `D = BEST_DELETE - KEEP`. A fixed, balanced L2 Logistic Regression with calibration-fold-only standardization was evaluated exactly once in 12-fold TRAIN-speaker LOSO. All folds converged in seven iterations.

| Metric | Frozen R5-3A TRAIN value |
|---|---:|
| Addition/all-negative AUC | 0.7825848108511275 |
| Addition/correct-only AUC | 0.8269990598295717 |
| Addition/substitution-containing AUC | 0.7143987795720519 |
| Addition/deletion-containing AUC | 0.6112200966968173 |
| Binary Macro-F1 | 0.5583786087141792 |
| Addition F1 | 0.14285714285714285 |
| Correct-only FAR | 0.02446808510638298 |
| Substitution-negative FAR | 0.05257856567284448 |
| Deletion-negative FAR | 0.09907341411261582 |
| Exact-event F1 | 0.044044044044044044 |

G1, G2, G3, G4, G5, and G8 passed. G6 and G7 failed, so the final result was 6/8 and `R5_3A_ROBUST_THETA_NOT_AUTHORIZED`. Relative to R5-1A, aggregate AUC, Binary Macro-F1, Addition F1, correct-only FAR, and exact-event F1 improved slightly, but substitution FAR worsened and deletion FAR worsened substantially. The standardized coefficient medians were A 0.9953044129046558, S 0.1523077166275505, and D 0.5849599738415144; every coefficient was positive in all 12 folds. This is consistent with failure to retain relation-specific suppression, but it is not evidence that S or D causally creates Addition.

The allowed conclusion is narrow: evidence separation recovered useful overall Addition discrimination compared with R5-2B and slightly exceeded R5-1A on several aggregate TRAIN metrics, but the fixed linear fusion did not successfully use substitution/deletion evidence as suppressive confounder information. R5-3A is therefore closed as `R5_3A_EVIDENCE_SEPARATION_GENERATION_CLOSED_NOT_CONFIRMED`. VALIDATION was not accessed for R5-2 or R5-3A, TEST remains untouched, and any future Addition work requires a separately named preregistered generation.

## 17. R5-4 INSERT-candidate materialization and recoverability

R5-4 separated a technical provenance question from a scientific feasibility question. R5-4B first materialized the complete frozen TRAIN INSERT landscape without using Addition truth during scoring. It persisted 2,977,040 candidate rows for all 16,582 runtime words: 2,976,844 alignable candidates and 196 impossible candidates affecting 65 words. No word lacked a finite candidate. Candidate-derived BEST_INSERT phone/boundary and exact winning score matched the frozen R5-2B result for 16,582/16,582 words. All ten materialization gates passed. This establishes artifact completeness and scorer reproduction only; it does not establish Addition performance.

The resumed R5-4A audit then joined only already-frozen TRAIN truth events to the frozen R5-4B candidate shards. All 304 single-addition words mapped exactly, while 19 multiple-addition words containing 38 events were analyzed separately. The primary single-addition result was:

| Metric | Frozen R5-4A value |
|---|---:|
| Top-1 recoverability | 0.13486842105263158 |
| Top-3 recoverability | 0.24013157894736842 |
| Top-5 recoverability | 0.27960526315789475 |
| Top-10 recoverability | 0.3881578947368421 |
| Mean truth rank | 57.10197368421053 |
| Median truth rank | 19.0 |
| Q25 / Q75 / Q90 | 4.0 / 75.75 / 172.7 |
| Maximum truth rank | 364 |
| Mean reciprocal rank | 0.2198158912466136 |

Only the mapping gate passed. Top-5 failed the frozen 0.40 gate, Top-10 failed the 0.55 gate, and median rank failed the <=10 gate. The result was 1/4 and `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_NOT_CONFIRMED`.

Recoverability was position-, phone-, and speaker-dependent. BETWEEN events were materially stronger (N=156, Top-10 0.5577, median rank 9) than BEFORE_FIRST events (N=44, Top-10 0.1364, median 158.5) and AFTER_FINAL events (N=104, Top-10 0.2404, median 28). Adequately supported D, N, R, T, IH, and IY groups were relatively stronger; AX, G, K, EH, and W were relatively weak, with sample-size caution. Speaker variation was substantial, but weakness occurred across several sizeable cohorts and was not attributable to one speaker. Multiple-addition events mapped 38/38 and had per-event Top-10 0.5, but one single-INSERT hypothesis cannot encode multiple simultaneous additions.

The defensible conclusion is that the exact CTC INSERT family contains partial localization information and that candidates below BEST_INSERT recover additional events. However, the correct event is not generally near enough to the top to meet the preregistered feasibility criteria. Low historical exact-event F1 therefore cannot be explained mainly by top-1 selection alone. The frozen evidence does not justify a new generation based only on top-k, marginalization, or alternative pooling of this same candidate family. R5-4A closes as `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED`; R5-4B closes as `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS`. VALIDATION and TEST were not accessed.

## Appendix A — Frozen metrics and gates

The same six numeric gates were used for R5-1A development and R5-1B transfer:

1. addition vs all non-addition ROC-AUC >= 0.70;
2. addition vs correct-only ROC-AUC >= 0.70;
3. Binary Macro-F1 > 0.548179;
4. addition F1 > 0.129246;
5. correct-only false-addition rate <= 0.054352;
6. exact-event F1 > 0.026688.

R5-1A TRAIN passed 6/6. R5-1B VALIDATION passed G1, G2, G4, and G6, and failed G3 and G5.

## Appendix B — Frozen identities

| Artifact | SHA-256 |
|---|---|
| V4 metadata | `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D` |
| R4-4C2 checkpoint | `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085` |
| R5-1A contract | `A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB` |
| R5-1A frozen scorer | `4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90` |
| R5-1A execution manifest | `C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6` |
| R5-1B contract | `BD169175C0777B1C37506E95350CB6AD90A992045527396DDE5DA4419779AC4A` |
| R5-1B execution manifest | `B1E053F8B80F6EDCBB4195BA632B1215E9FA42A7AA0AF5C2950D92AFF6E0771E` |
| R5-1C audit manifest | `67BCB0D2655F68100E7BB657FD193E63300802B398D395DBA5EF6D120E4C7241` |
| R5-1 closure conclusion | `5DD35196D07A1479DF6B6E26864AEF62DE0A7F11A73206646800232F09D8D7DF` |
| R5-1 closure final status | `AEF81158C8F5890305D69DC72412F02ABF1D59712C78E6F2650F156E9A4EF9C6` |
| R5-1 closure manifest | `C8E71EDE56902D594A60F1194ABF8A72AB0A7EFBE4F212F57BC71FCEF21B1D69` |

## Appendix C — Protocol history

- R5-0 performed data and TRAIN-only diagnostic feasibility work; TEST was limited to frozen aggregate support counts.
- R5-1 stopped before metrics and must not be reported as a scientific failure.
- R5-1A was a new pre-metric contract, not a silent correction of R5-1.
- R5-1A static verification used only synthetic inputs.
- R5-1A real-data development used TRAIN only and one 12-fold speaker LOSO evaluation.
- R5-1B used exactly one fixed-threshold VALIDATION execution and no validation calibration.
- R5-1C was post-hoc and used existing artifacts only.
- R5-1D and this consolidation were documentation-only.
- R5 TEST remained closed throughout.

## Appendix D — Authoritative closure package

The authoritative closure package is `ai-training/experiments/r5_1_addition_scoring_closure/`. Its final status is **R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED**, and its manifest SHA-256 is `C8E71EDE56902D594A60F1194ABF8A72AB0A7EFBE4F212F57BC71FCEF21B1D69`.
