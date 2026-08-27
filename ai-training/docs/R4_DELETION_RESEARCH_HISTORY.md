# Phoenix R4 Deletion Research History

## 1. Research objective

R4 investigated whether Phoenix could detect when an expected canonical phone was omitted in a learner's word pronunciation. The research deliberately separated deletion from the already completed R3 correct/substitution work. Its final question was not merely whether deletion-related acoustic evidence existed, but whether that evidence could produce speaker-robust deletion decisions under frozen validation requirements.

Final status: **R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED**.

The final method was a self-trained CNN+BiGRU CTC acoustic phone-sequence model followed by TARGET-normalized CTC hypothesis scoring and a global threshold obtained from 12-fold TRAIN-speaker leave-one-speaker-out calibration:

```text
word audio
-> CNN + one-layer BiGRU
-> CTC phone posterior
-> compare KEEP / DELETE / 39 SUBSTITUTION hypotheses
-> TARGET-normalized sequence scores
-> D_i = DELETE_i - max(KEEP_i, BEST_SUB_i)
-> deletion when D_i >= 0.16184102947061696
```

This final method did not pass the complete frozen validation contract. R4 therefore remains a research result, not production-validated deletion behavior.

## 2. Dataset and split

R4 used V4 expected/observed metadata derived from L2-ARCTIC audio and manual annotations. MFA supplied word boundaries where needed; MFA phone labels were not treated as observed-phone truth.

| Split | Speakers | Role |
|---|---|---|
| TRAIN | BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA | Training and TRAIN-only calibration |
| VALIDATION | ABA, HKK, HQTV, LXC, MBMPS, SVBI | Iterative development validation |
| R4 TEST | ASI, ERMS, SKA, THV, TXHC, YDCK | Untouched independent holdout; never accessed |

The word-level CTC branch used 16,259 TRAIN words and 7,728 VALIDATION words. Words containing addition were excluded from the deletion experiment. The final validation population contained 26,337 expected-phone rows: 22,759 correct, 2,664 substitution, and 914 deletion.

## 3. Evaluation policy

The research used speaker-disjoint splits, frozen contracts, one-run neural experiments, deterministic sequence and tie rules, TRAIN-only calibration, locked validation thresholds, and no post-hoc validation adjustment. Duration could be used for diagnostic matched controls but could not enter the deployable deletion decision.

The final confirmation contract required all eight gates:

1. Binary Macro-F1 >= 0.70.
2. Deletion recall >= 0.45.
3. Deletion F1 >= 0.40.
4. Substitution false-deletion <= 0.25.
5. Matched-control Macro-F1 >= 0.60.
6. Matched-control deletion F1 >= 0.55.
7. Every VALIDATION speaker with at least 30 deletions had deletion recall >= 0.25.
8. Three-relation Macro-F1 >= 0.40.

R4-4D2B passed only gates 4, 7, and 8. No R4 TEST execution was authorized because the final development candidate was not confirmed.

## 4. Research timeline

The sequence below preserves the actual hypothesis-to-decision path; the final method was not assumed in advance.

| Stage | Hypothesis or question | Main evidence | Decision / next hypothesis |
|---|---|---|---|
| R4-0 | Can a deletion slot be detected from its manual interval? | Duration-only Binary MF1 0.668146; deletion F1 0.364164 | Strong annotation shortcut; retain only as comparator |
| R4-1 | Can a learned conditional classifier detect deletion? | Binary MF1 0.657336; deletion F1 0.341612 | Signal weak; inspect what evidence the model learned |
| R4-2A | Does expected-phone acoustic mismatch identify deletion? | Binary MF1 0.566997; deletion F1 0.197525 | Mostly mismatch, not deletion-specific |
| R4-2B/2C | Can temporal evidence or MFA absence localize deletion? | Some temporal evidence; about 99% of manual deletions retained expected MFA label | Close fixed-slot MFA route; formulate sequence reasoning |
| R4-3A | Are expected/observed word relations sequence-representable? | Very high oracle recovery; feasible word localization | Build deterministic sequence scorer |
| R4-3B | Can frozen R3 sliding evidence plus DP recover deletion? | Binary MF1 0.503101; deletion F1 0.025465 | Sequence formulation alone insufficient |
| R4-3C | Can global prior scaling rescue deterministic DP? | No eligible global scale in TRAIN-only audit | Close handcrafted DP/prior family |
| R4-4A | Can a self-trained CTC model represent missing phones without slots? | 40 phones + blank; about 20 ms output; zero alignability failures | Train one CNN-only CTC model |
| R4-4B | Does CNN-only CTC learn observed phone sequences? | PER 0.602840; severe under-generation | Audit acoustic failure |
| R4-4C0 | Is failure blank bias, confusion, collapse, or generalization? | Mixed under-generation and broad confusion; no major split gap | One small BiGRU justified |
| R4-4C2 | Does one BiGRU improve acoustic sequence modeling? | PER improved to 0.453330; under-generation reduced | Acoustic model improved; greedy omission still unreliable |
| R4-4D0 | Can exact CTC likelihood compare KEEP/DELETE/SUB hypotheses? | TRAIN RAW del/non-del AUC 0.905407; del/sub AUC 0.904593 | Strong continuous signal; calibrate locked threshold |
| R4-4D1 | Does one RAW TRAIN threshold transfer? | Validation AUCs remained high but deletion F1 0.321343 | Threshold transfer failed |
| R4-4D2A | Can TRAIN-speaker LOSO select robust predefined normalization? | Only TARGET passed development gates | Freeze TARGET theta 0.16184102947061696 |
| R4-4D2B | Does frozen TARGET calibration meet final validation contract? | Binary MF1 0.652102; deletion F1 0.331390; 3/8 gates | Not confirmed; close R4 |

## 5. Experiment history

### 5.1 R4-0 — duration shortcut audit

**Hypothesis.** A manually annotated deletion interval might contain detectable evidence.

**Experiment.** A duration-only diagnostic baseline used the manual deletion-slot duration.

**Evidence.** Binary Macro-F1 was 0.668146 and deletion F1 was 0.364164.

**Decision.** Duration was highly predictive in annotation data, but a system cannot observe a manual interval for a phone that was never produced. Duration therefore remained a diagnostic comparator, not a deployable Phoenix deletion feature. This motivated the first learned acoustic deletion experiment.

### 5.2 R4-1 — first learned deletion experiment

**Hypothesis.** A conditional learned classifier could distinguish deletion from non-deletion using acoustic input.

**Evidence.** Validation Binary Macro-F1 was 0.657336 and deletion F1 was 0.341612, below the duration-only baseline on both headline metrics.

**Decision.** The deletion signal was classified as weak. The next question became whether the model had learned genuine deletion evidence or merely generic mismatch cues.

### 5.3 R4-2A — expected-phone mismatch scoring

**Hypothesis.** Weak support for the expected phone might directly indicate deletion.

**Evidence.** Binary Macro-F1 was 0.566997 and deletion F1 was 0.197525. Substitutions also reduce expected-phone support.

**Decision.** Status: **EXPECTED_PHONE_MISMATCH_ONLY**. The scalar score detected pronunciation mismatch more than deletion specifically. R4 moved to temporal and localization evidence.

### 5.4 R4-2B and R4-2C — temporal localization and MFA anchors

**Hypothesis.** Local temporal evidence or absence of the expected MFA phone label could anchor deletion.

**Evidence.** R4-2B found some temporal/localization evidence. R4-2C found that approximately 99% of manually annotated deletion cases still retained the expected MFA phone label.

**Decision.** MFA absence was not a reliable runtime deletion anchor, and a fixed deletion slot remained unavailable. The fixed-slot MFA route was closed. The next hypothesis treated pronunciation as a word-level expected-versus-observed sequence problem.

### 5.5 R4-3A — sequence design feasibility

**Hypothesis.** Deletion could be inferred by aligning an expected phone sequence with an acoustically inferred observed sequence.

**Evidence.** Word-level relation reconstruction was representable, oracle relation recovery was approximately 99.979%, sequence localization was technically feasible, and R3 temporal representations could support sequence reasoning.

**Decision.** The edit-alignment contract was not the primary bottleneck. R4 proceeded to deterministic sequence scoring using frozen R3 sliding acoustic evidence.

### 5.6 R4-3B — deterministic sequence deletion

**Hypothesis.** A frozen dynamic program over MATCH, SUBSTITUTION, and DELETE could recover deletion without a manual interval.

**Evidence.** Binary Macro-F1 was 0.503101 and deletion F1 was 0.025465. Only 13 of 914 validation deletions were recovered in the locked run. The representation remained numerically functional, but empirical relation priors strongly suppressed DELETE.

**Decision.** Sequence formulation by itself did not solve deletion. A TRAIN-only audit tested whether one global prior scale could rescue the family.

### 5.7 R4-3C — global prior-rescaling audit

**Hypothesis.** A single global scale on all operation priors might reduce DELETE suppression without destroying discrimination.

**Evidence.** Exact 12-fold TRAIN-speaker LOSO calibration evaluated the preregistered scale grid. No global prior scale satisfied the frozen eligibility requirements.

**Decision.** Status: **R4_3C_NO_ELIGIBLE_PRIOR_SCALE**. No validation tuning was performed. The deterministic handcrafted prior-rescaling family was closed.

### 5.8 R4-4A — self-trained CTC feasibility

**Hypothesis.** A self-trained framewise sequence model could decode an observed phone sequence without requiring a fixed deletion interval.

**Design.** The target emitted the expected phone for a correct relation, the manually observed phone for substitution, and nothing for deletion. The vocabulary contained 40 canonical phones plus one CTC blank. Addition-containing words were excluded. Full word audio produced approximately 20 ms output steps.

**Evidence.** TRAIN and VALIDATION both had zero CTC alignability failures under the selected temporal resolution.

**Decision.** CTC could represent pronunciation sequence changes without an MFA deletion slot. One CNN-only CTC experiment was preregistered.

### 5.9 R4-4B — CNN-only CTC

**Hypothesis.** A small time-preserving CNN could learn the observed phone sequence well enough for greedy omission to indicate deletion.

**Evidence.** The selected checkpoint was epoch 35. Validation PER was 0.602840; exact-sequence accuracy was 0.124871. Binary Macro-F1 was 0.476941. Deletion precision, recall, and F1 were 0.062972, 0.483589, and 0.111433.

The decoder under-generated:

- aggregate decoded/target length ratio: 0.7757;
- words decoded shorter than target: 48.85%;
- CTC target-phone deletion edits: 6,199;
- real pronunciation deletion rows: 914.

**Decision.** Greedy missing phones were dominated by acoustic recognition omissions and could not directly be interpreted as pronunciation deletion. A failure-mode audit was required before changing the architecture.

### 5.10 R4-4C0 — CTC failure-mode audit

**Hypothesis.** The weak deletion result might be explained by blank dominance, broad phone confusion, repeat collapse, or speaker generalization.

**Evidence.** The audit classified **CTC_MIXED_ACOUSTIC_FAILURE**. It found broad phone-recognition weakness, high blank occupancy, and under-generation even on all-correct words. TRAIN and VALIDATION did not show a major generalization gap, and repeat collapse was negligible rather than primary.

**Decision.** One small BiGRU after the CNN was justified to add temporal context. No decoding or loss calibration was introduced.

### 5.11 R4-4C2 — CNN+BiGRU CTC

**Hypothesis.** One bidirectional GRU could improve phone identity and sequence completeness while preserving the small self-trained architecture.

**Evidence.** The selected checkpoint was epoch 35. PER improved from 0.602840 to 0.453330, an absolute reduction of 0.149510. The decoded/target ratio increased to 0.924911, the shorter-word rate fell to 26.51%, CTC deletion edits fell to 2,769, and clean-word false expected-phone deletion fell to 0.103964.

Relation decisions based on greedy omission remained insufficient: Binary Macro-F1 was 0.555712; deletion precision, recall, and F1 were 0.117122, 0.445295, and 0.185464.

**Decision.** The BiGRU substantially improved acoustic sequence modeling, but greedy sequence omission remained too imprecise to serve directly as deletion detection. R4 froze the acoustic model and changed only the sequence decision layer.

### 5.12 R4-4D0 — exact CTC hypothesis scoring

**Hypothesis.** Instead of requiring the greedy decoder to omit a phone, the frozen CTC posterior could directly compare the likelihoods of full-word KEEP, local DELETE, and 39 local SUBSTITUTION hypotheses.

**Evidence.** On TRAIN, the selected RAW score achieved deletion-vs-nondeletion ROC-AUC 0.905407 and deletion-vs-substitution ROC-AUC 0.904593. BEST_SUB phone accuracy was 0.657065. RAW scoring also showed an intrinsic target-length bias.

**Decision.** Strong deletion-specific continuous evidence existed. A fully frozen TRAIN-only threshold and locked validation evaluation were justified.

### 5.13 R4-4D1 — RAW threshold transfer

**Hypothesis.** A global RAW threshold selected from TRAIN could convert the strong continuous signal into robust decisions.

**Calibration.** The TRAIN-only frozen threshold was 2.197946548461914.

**Technical stop.** The first locked execution stopped before validation metrics because the matched-control file used physical CSV row numbering while the driver exported zero-based `source_index`. Exactly 233 of 1,434 identities were unmatched. Metadata-only analysis established the canonical convention `source_csv_row = source_index + 2`: `+1` converts zero-based indexing to one-based rows and another `+1` accounts for the CSV header. The correction mapped all 1,434 identities with no collision. It changed bookkeeping only; model, scores, threshold, gates, and predictions were unchanged.

**Evidence after corrected validation-only execution.** Binary Macro-F1 was 0.643750. Deletion precision, recall, and F1 were 0.253149, 0.439825, and 0.321343. Continuous validation ROC-AUC remained 0.887740 for deletion versus non-deletion and 0.887642 for deletion versus substitution.

**Decision.** Status: **R4_4D1_HYPOTHESIS_THRESHOLD_TRANSFER_FAIL**. The ranking signal transferred, but the TRAIN-calibrated RAW threshold did not meet the locked decision requirements. The technical identity stop was not a model-performance failure.

### 5.14 R4-4D2A — TRAIN-only speaker-robust calibration

**Hypothesis.** A predefined normalization might make calibration more stable across speakers without changing the frozen acoustic model.

**Experiment.** Exact 12-fold TRAIN-speaker LOSO independently audited the three score families defined before validation: RAW, TARGET, and TIME. No new formula, learned calibration, or validation selection was allowed.

| Family | OOF deletion F1 | Development result |
|---|---:|---|
| RAW | 0.289421 | Ineligible |
| TARGET | 0.326474 | Passed all frozen TRAIN development gates |
| TIME | 0.294958 | Ineligible |

**Decision.** TARGET normalization was selected. The deployment threshold was the ordinary float64 median of the 12 TARGET fold thresholds: **0.16184102947061696**. TARGET improved speaker-transfer calibration robustness relative to RAW. One final iterative development validation was preregistered, with no further R4 calibration cycle permitted if it failed.

### 5.15 R4-4D2B — final TARGET validation

**Hypothesis.** TARGET normalization plus the frozen speaker-LOSO median threshold would transfer robustly enough to pass all deletion confirmation gates.

**Evidence.** The final locked validation produced:

| Metric | Result |
|---|---:|
| Binary Macro-F1 | 0.652102 |
| Balanced Accuracy | 0.670750 |
| Deletion precision | 0.298077 |
| Deletion recall | 0.373085 |
| Deletion F1 | 0.331390 |
| Correct false-deletion | 0.031548 |
| Substitution false-deletion | 0.031907 |
| Three-relation Macro-F1 | 0.465742 |
| Matched Macro-F1 | 0.593491 |
| Matched deletion F1 | 0.485769 |
| Del/non-del ROC-AUC | 0.854124 |
| Del/sub ROC-AUC | 0.862234 |

Only 3 of 8 gates passed. Binary Macro-F1, deletion recall, deletion F1, matched Macro-F1, and matched deletion F1 failed. False-deletion control, the supported-speaker recall gate, and three-relation Macro-F1 passed.

**Decision.** Status: **R4_4D2B_DELETION_VALIDATION_NOT_CONFIRMED**. The current R4 deletion research branch was closed.

## 6. Historical comparator table

| Method | Binary Macro-F1 | Deletion F1 |
|---|---:|---:|
| Duration-only | 0.668146 | 0.364164 |
| R4-1 | 0.657336 | 0.341612 |
| R4-2A | 0.566997 | 0.197525 |
| R4-3B | 0.503101 | 0.025465 |
| R4-4C2 greedy | 0.555712 | 0.185464 |
| R4-4D1 RAW | 0.643750 | 0.321343 |
| R4-4D2B TARGET | 0.652102 | 0.331390 |

The final method substantially improved over the failed early neural and deterministic sequence approaches, but it did not surpass the duration-only diagnostic baseline on Binary Macro-F1 or deletion F1. Duration remains non-deployable because it depends on a manual deletion interval.

## 7. Why R4 failed

### Plain-language defense explanation

> The model learned meaningful acoustic evidence indicating whether an expected phone may be missing. However, the deletion score distribution changed between speakers. A threshold calibrated on the training speakers therefore missed too many real deletions on validation speakers. The system could rank likely deletion cases reasonably well, but it could not convert that signal into sufficiently stable deletion decisions.

### Giải thích ngắn bằng tiếng Việt

> Mô hình có nhận ra tín hiệu của âm bị bỏ, nhưng ngưỡng để quyết định một âm thật sự bị deletion chưa ổn định khi đổi sang người nói mới. Vì vậy hệ thống còn bỏ sót nhiều deletion thật và chưa đạt các tiêu chí validation đã đặt trước.

The final scientific interpretation is therefore:

```text
MEASURABLE DELETION-RELATED SIGNAL
but
INSUFFICIENT ROBUST DELETION DECISION PERFORMANCE
```

R4 did not demonstrate absence of deletion-related acoustic signal. It also did not validate a production deletion classifier.

## 8. Positive findings and limitations

Positive findings retained:

- self-trained CTC represented deletion without a fixed manual deletion slot;
- one BiGRU materially improved acoustic phone-sequence recognition;
- exact CTC hypothesis scoring separated deletion from substitution much better than greedy omission;
- speaker-LOSO TARGET calibration improved robustness over the original RAW threshold;
- false-deletion rates were controlled in the final run.

Final limitation:

- deletion recall and deletion F1 remained below the frozen requirements;
- full and matched-control Macro-F1 gates failed;
- the decision threshold did not provide sufficiently robust deletion recovery across unseen validation speakers;
- deletion output must be labeled **RESEARCH LIMITATION / NOT CONFIRMED**, not production behavior.

Phoenix's R3 correct/substitution research remains separate and TEST-confirmed. R4's unconfirmed deletion result does not revise that R3 conclusion.

## 9. R4 TEST status

R4 TEST was never accessed: no TEST path was resolved, no TEST audio was read, no posterior or hypothesis score was produced, and no TEST metric was calculated. No candidate passed the final frozen validation contract, so opening TEST would not have been scientifically justified.

The six-speaker R4 TEST split remains untouched for a genuinely new future deletion research generation with a new hypothesis and independent evaluation protocol.

## 10. Future research policy

The current R4 branch is closed. It must not resume through R4-4D2C, R4-4D3, a new threshold, another normalization, RAW/TARGET blending, another GRU, LSTM, Transformer, Conformer, duration correction, or phone/speaker-specific thresholds.

Any future deletion research must be a new research generation. The next planned research stage is **R5 addition feasibility**, but no R5 implementation is part of this documentation task.

## Appendix A — reproducibility and frozen identities

Important frozen identities:

| Artifact | SHA-256 |
|---|---|
| V4 metadata | `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D` |
| R4-4B selected checkpoint, epoch 35 | `A154DFAC573D69B8ED1A71CBCDC23227EA3E80929890AD87E97ED85667142106` |
| R4-4C2 selected checkpoint, epoch 35 | `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085` |
| R4-4D1 complete numerical contract | `5DC07A4B719FD6F38DBD1366CF802787FA3882CA972E5356A8F91DB435443425` |
| R4-4D2B preregistration JSON | `F0AC6874C1330DBFA2A8D99C88BE5167DCBED31122B40CE1F427FC0938DFA8AA` |
| R4-4D2B preregistration Markdown | `71BEB2A97331455863CCE203FD886F6831697D69F857B2AC1D5C35EAA5DB2D6B` |
| R4-4D2B report | `646B7F7BDFB6EEFFACEF17C038847AFDF13248D58FBD7C1EE0C5F2147A553CAB` |
| R4-4D2B final status | `DC5D7AC51DEE0079F6FE075C56E97A0321A71B7A6482C38583BD9DF063D30134` |
| R4-4D2B artifact manifest | `532308D7223B55E4BCFE5846FAE4F60A8497F34DD678A123ECB275E4755C463C` |
| R4 closure manifest | `3E21936C6175F9DEA5FAE3346E96EECD3AEF18732072C42B0B2FCAE4161D6174` |

The frozen matched-control identity contained 1,434 rows (717 deletion and 717 non-deletion) and used the canonical physical CSV convention `source_csv_row = source_index + 2`.

## Appendix B — execution discipline and technical history

- Neural experiments R4-4B and R4-4C2 each authorized one fresh run with seed 42 and 36 epochs; checkpoint selection was frozen by minimum validation PER.
- R4-3C, R4-4D0, and R4-4D2A used TRAIN-only parameter or feasibility analysis. Validation did not select their candidate scale, score family, or threshold.
- R4-4D1's first technical stop resulted from row-identity bookkeeping, not model performance. The `+2` correction was frozen before the validation-only technical re-execution.
- R4-4D2B was an iterative development validation because earlier R4 validation results had already been observed. It was not represented as an untouched final holdout.
- No post-hoc threshold, equality-rule, checkpoint, normalization, duration, phone, or speaker adjustment was allowed after locked validation results became visible.
- R4 TEST stayed closed throughout the entire branch.

## Appendix C — authoritative closure package

The authoritative final summary is `ai-training/experiments/r4_deletion_research_closure/`, with final status **R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED**. Its files preserve the scientific conclusion, experiment timeline, comparator table, limitations, TEST closure, and future research policy.
