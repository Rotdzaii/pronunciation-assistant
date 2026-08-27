# Phoenix Research Master State

Authoritative recovery and research-state register for Phoenix pronunciation research.

- Last verified: `2026-08-27 18:10:04 +07:00` (`Asia/Bangkok`)
- Verification basis: local frozen artifacts in the main repository and linked research worktree; no fetch, training, inference, VALIDATION access, or TEST access was performed for this document.
- Source-of-truth rule: frozen experiment/result/closure artifacts override roadmap prose. Unknown identities are marked `UNVERIFIED`.
- Preservation classification: `PRESERVATION_CLASSIFICATION_COMPLETE`
- Current local deletion status: `NOT_SAFE_TO_REMOVE_YET`

## 1. Repository identity and preservation state

| Field | Verified value |
|---|---|
| Main repository path | `C:/Users/Admin/Documents/KLTN/pronunciation-assistant` |
| Research worktree path | `C:/Users/Admin/Documents/KLTN/pronunciation-assistant-research` |
| Origin URL | `https://github.com/Rotdzaii/pronunciation-assistant.git` |
| Main application branch | `feature/wav2vec2-demo-score` |
| Research branch | `research/phoenix-correctness` |
| Main worktree HEAD | `4758c0dcf5ab1681af1fcf74e068d609030d11a8` |
| Research compact-preservation HEAD before documentation reconciliation | `180655ec94293b165f9ed0d3b22abd46909a9d7c` |
| Worktree relationship | Linked Git worktrees sharing the main repository's Git object database |
| Main-branch upstream/live ref | No configured upstream; `refs/heads/feature/wav2vec2-demo-score` is absent on the live origin |
| Main HEAD live backup identity | `refs/heads/backup/full-demo-state-20260716` exists live at `4758c0dcf5ab1681af1fcf74e068d609030d11a8` |
| Research upstream/live ref | `origin/research/phoenix-correctness`; live origin verified at `180655ec94293b165f9ed0d3b22abd46909a9d7c` before documentation reconciliation |
| Git LFS | Not configured for repository content; `git lfs ls-files` returned zero paths |
| External archive | `NOT_CREATED` |
| Cleanup safety | `NOT_SAFE_TO_REMOVE_YET` |

No fetch was performed. Live branch identities above were verified with read-only remote inspection. R3 through R6 compact research artifacts, including the V3/V4 compact contracts, are preserved on `origin/research/phoenix-correctness` through the pre-documentation boundary `180655ec94293b165f9ed0d3b22abd46909a9d7c`. A clean clone of the current origin still does not preserve uncommitted files, ignored or externally archived research material, local stashes, local-only branch identities, or reflog/dangling objects.

## 2. Dataset identity

### 2.1 Corpus and frozen speaker split

- Corpus: L2-ARCTIC `v5.0`
- Raw corpus directory used by the research pipelines: `ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/`
- Frozen split, verified in R4 and R5 contracts:

| Split | Speakers | Research access state |
|---|---|---|
| TRAIN | `BWC`, `EBVS`, `HJK`, `NCC`, `NJS`, `PNV`, `RRBI`, `TLV`, `TNI`, `YBAA`, `YKWK`, `ZHAA` | Used throughout development; R6-1 used TRAIN only |
| VALIDATION | `ABA`, `HKK`, `HQTV`, `LXC`, `MBMPS`, `SVBI` | Used for iterative R3/R4/R5 development as frozen artifacts specify; not accessed by R6 |
| TEST | `ASI`, `ERMS`, `SKA`, `THV`, `TXHC`, `YDCK` | R3 consumed once under locked protocol; untouched by R4/R5/R6 |

### 2.2 Derived dataset register

| Artifact | Relative path | Bytes | SHA-256 | Preservation |
|---|---|---:|---|---|
| V3 correctness dataset | `ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv` | 41,794,901 | `433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E` | Irreplaceable until clean regeneration is proven; external archive required |
| V3 audit | `ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3_audit.json` | 8,340 | `22E304F5F65D47FAAE4BC3624E8EFDACD1F6C653B5589B261D48F6F64674ACB4` | Git-sized reproducibility metadata |
| V4 expected/observed dataset | `ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv` | 48,727,961 | `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D` | Irreplaceable until clean regeneration is proven; external archive required |
| V4 audit | `ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4_audit.json` | 20,238 | `D188A6CF11A15556BD94E9F399D4D9D3506437C02FAB03E26C184C3EF8065B45` | Git-sized reproducibility metadata |

The V4 data SHA matches the frozen R3/R4/R5/R6 source identities. The V3 audit records 24 speakers, 3,599 annotation files, 135,890 intervals, and 120,236 research rows. The V4 audit exists and its bytes/hash are verified; it contains case-colliding object keys such as `observed:err` and `observed:ERR`, so recovery tooling should use a case-sensitive JSON parser.

The builders and tests that must accompany these data identities are:

- `ai-training/scripts/build_l2_arctic_all_speakers_correctness_v3.py`
- `ai-training/scripts/build_l2_arctic_expected_observed_v4.py`
- `ai-training/tests/test_build_l2_arctic_all_speakers_correctness_v3.py`
- `ai-training/tests/test_build_l2_arctic_expected_observed_v4.py`

The V3/V4 builders, tests, audits, contracts, and compact documentation are protected on `origin/research/phoenix-correctness`. The large V3/V4 CSV datasets themselves are not stored in ordinary Git and remain external-archive material; they must not be treated as safely recoverable from GitHub alone.

## 3. Canonical 40-phone vocabulary

Verified against `ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/phone_vocab.json` and the frozen R3 TEST protocol.

| Index | Phone | Index | Phone | Index | Phone | Index | Phone |
|---:|---|---:|---|---:|---|---:|---|
| 0 | AA | 10 | DH | 20 | K | 30 | SH |
| 1 | AE | 11 | EH | 21 | L | 31 | T |
| 2 | AH | 12 | ER | 22 | M | 32 | TH |
| 3 | AO | 13 | EY | 23 | N | 33 | UH |
| 4 | AW | 14 | F | 24 | NG | 34 | UW |
| 5 | AX | 15 | G | 25 | OW | 35 | V |
| 6 | AY | 16 | HH | 26 | OY | 36 | W |
| 7 | B | 17 | IH | 27 | P | 37 | Y |
| 8 | CH | 18 | IY | 28 | R | 38 | Z |
| 9 | D | 19 | JH | 29 | S | 39 | ZH |

R4/R5/R6 CTC artifacts use these phones at indices 0–39 plus CTC blank at index 40.

## 4. R3 final state — Correct/Substitution

### 4.1 Authoritative status

`TEST_TRANSFER_CONFIRMED`

R3 is complete. Its development path was R3-1A through R3-1D observed-phone modeling, R3-2A continuous expected-vs-alternative scoring, R3-2B speaker-transfer/threshold stability, frozen TEST protocol, and one R3-3 locked TEST evaluation.

### 4.2 Selected model and frozen rule

| Field | Verified value |
|---|---|
| Final checkpoint | `ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt` |
| Checkpoint bytes | 355,745 |
| Checkpoint SHA-256 | `5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E` |
| Selected epoch | 47 |
| Acoustic validation rows | 28,212 |
| Acoustic validation accuracy | 0.528285835814547 |
| Acoustic validation Macro-F1 | 0.4978231682146241 |
| Acoustic validation balanced accuracy | 0.543519960742611 |
| Acoustic validation Top-3 accuracy | 0.8234793704806466 |
| Binary score | expected-phone logit minus best alternative-phone logit |
| Frozen operational threshold | `-1.293920` |
| Full source candidate | `-1.2939202785491943` |
| Frozen decision | substitution when margin `<= -1.293920`; otherwise correct |

Frozen all-VALIDATION binary calibration:

| Metric | Value |
|---|---:|
| Accuracy | 0.7729335034736992 |
| Balanced accuracy | 0.6528404102538492 |
| Binary Macro-F1 | 0.5884736601934054 |
| Substitution precision | 0.227435697583788 |
| Substitution recall | 0.5015469233413544 |
| Substitution F1 | 0.312955812955813 |
| Confusion matrix `[correct, substitution]` | `[[20347, 4956], [1450, 1459]]` |

Locked TEST result, 28,216 rows:

| Metric | Value |
|---|---:|
| Accuracy | 0.7627941593422172 |
| Balanced accuracy | 0.6331427855188454 |
| Binary Macro-F1 | 0.5850483996852586 |
| Substitution precision | 0.23667905824039653 |
| Substitution recall | 0.46401457637412696 |
| Substitution F1 | 0.3134680480049236 |
| Acoustic Top-1 accuracy | 0.5307981287212928 |
| Acoustic Top-3 accuracy | 0.816416217748795 |
| Acoustic Macro-F1 | 0.4882340046043344 |
| Catastrophic speaker collapse | None |

### 4.3 Frozen identities and limitations

| Artifact | SHA-256 |
|---|---|
| `ai-training/experiments/r3_locked_test_protocol/r3_binary_margin_frozen_manifest.json` | `50AC3DC012191E2D401A5E7437B46F11F06356A5CC5EA931B916186B1C2B4576` |
| `ai-training/experiments/r3_locked_test_protocol/r3_binary_margin_frozen_protocol.md` | `B2750BD8438A4759DE0F40C1A9B244DE78AF8494F5CDEB139F023C0C53F470DF` |
| `ai-training/experiments/r3_locked_test_evaluation/artifact_hashes.json` | `12B15F9B8B78C1C21E507D81C13CF601E1F4F96FDB8A1972EE6A2E84253B734A` |
| `ai-training/experiments/r3_locked_test_evaluation/final_test_status.json` | `388ED5D7F221E308B10002D1C79880E89A4D7A419A64B6481C5B1406479E5344` |

Known pre-TEST limitation: `DH -> D` was a documented validation failure mode—support 300, detection recall 0.21666666666666667, 235 false negatives, median expected margin 0.051110975444316864. R3 produces a research correct/substitution decision and raw margin, not a calibrated 0–100 pronunciation score.

### 4.4 Immutable TEST policy

**R3 TEST HAS BEEN CONSUMED.** It was accessed exactly once for the frozen evaluation. There was no threshold search, retraining, checkpoint change, preprocessing change, or score remapping during that evaluation.

R3 TEST must never be used for retuning, threshold adjustment, checkpoint selection, feature selection, or post-hoc model changes. Future R3 reporting may read the already-frozen result artifacts; it must not rerun or adapt against the TEST split.

## 5. R4 final state — Deletion

### 5.1 Authoritative status and conclusion

`R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED`

Meaningful deletion-related acoustic evidence was observed, but robust deletion decision performance was not confirmed. This must not be summarized as “no signal.” The final continuous TARGET score had ROC-AUC 0.8541244506193716 for deletion versus non-deletion and 0.8622339516759647 for deletion versus substitution, but the frozen decision layer passed only 3 of 8 confirmation gates.

### 5.2 Causal progression

| Stage | Frozen result/finding |
|---|---|
| R4-0 | Duration shortcut was high and rejected as the intended acoustic solution |
| R4-1 | Direct controlled deletion signal was weak |
| R4-2A | R3 scalar evidence mainly represented mismatch, not deletion |
| R4-2B | Temporal evidence showed moderate deletion-related information |
| R4-2C | MFA cannot supply a fixed phone interval for a deleted phone |
| R4-3A | Expected/observed word-sequence relations were representable, with warnings |
| R4-3B | Deterministic sliding-logit DP suppressed DELETE; not confirmed |
| R4-3C | No eligible global prior scale rescued that DP family |
| R4-4A | Self-trained word-level CTC was technically feasible |
| R4-4B | CNN-only CTC learned sequences but under-generated; deletion decisions weak |
| R4-4C0 | Failure audit found under-generation plus broad phone confusion |
| R4-4C2 | CNN + one-layer BiGRU materially improved sequence modeling; confirmation still failed |
| R4-4D0 | TRAIN-only continuous CTC hypothesis signal was strong |
| R4-4D1 contract/driver | Complete evaluation contract frozen; driver and synthetic checks passed |
| R4-4D1 locked execution v1 | Contractual source-verification stop caused by matched-control identity failure; no scientific result |
| R4-4D1 identity correction | Identity-only correction frozen and verified |
| R4-4D1 locked execution v2 | RAW TRAIN threshold transferred poorly despite strong continuous VALIDATION signal |
| R4-4D2A | TRAIN-speaker LOSO selected TARGET normalization and robust median threshold |
| R4-4D2B | Final iterative VALIDATION passed 3/8 gates; deletion not confirmed |
| Closure | Branch closed without TEST access |

### 5.3 Final method and validation

Final method: self-trained CNN + one-layer bidirectional GRU CTC phone-sequence model, TARGET-normalized CTC hypothesis score, and frozen threshold.

| Field | Verified value |
|---|---|
| Score family | `TARGET` |
| Frozen threshold | `0.16184102947061696` |
| Threshold source | Ordinary float64 median of 12 TRAIN-speaker LOSO thresholds |
| Checkpoint | `ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt` |
| Checkpoint SHA-256 | `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085` |
| Threshold recalculated/changed on VALIDATION | No / No |
| Validation execution count | 1 |

Final R4-4D2B metrics:

| Metric | Value | Frozen outcome |
|---|---:|---|
| Binary Macro-F1 | 0.6521023094599894 | Fail (`>= 0.70`) |
| Balanced accuracy | 0.670749883524378 | Descriptive |
| Deletion precision | 0.2980769230769231 | Descriptive |
| Deletion recall | 0.3730853391684901 | Fail (`>= 0.45`) |
| Deletion F1 | 0.3313896987366375 | Fail (`>= 0.40`) |
| Three-relation Macro-F1 | 0.4657420632332678 | Pass (`>= 0.40`) |
| Matched Macro-F1 | 0.5934906451822113 | Fail (`>= 0.60`) |
| Matched deletion F1 | 0.4857685009487666 | Fail (`>= 0.55`) |
| Correct false-deletion rate | 0.031547959049167365 | Descriptive |
| Substitution false-deletion rate | 0.03190690690690691 | Pass (`<= 0.25`) |
| Supported-speaker recall gate | All supported speakers passed | Pass |

R4-4D2B did not surpass the duration-only baseline on binary Macro-F1 or deletion F1. The primary limitation is `INSUFFICIENT_ROBUST_DELETION_DECISION_PERFORMANCE`, not absence of continuous acoustic evidence.

### 5.4 Closure identities and policy

| Artifact | SHA-256 |
|---|---|
| `ai-training/experiments/r4_4d2a_train_only_robust_calibration/r4_4d2b_preregistered_validation_design.json` | `F0AC6874C1330DBFA2A8D99C88BE5167DCBED31122B40CE1F427FC0938DFA8AA` |
| `ai-training/experiments/r4_4d2b_final_target_validation/artifact_hashes.json` | `532308D7223B55E4BCFE5846FAE4F60A8497F34DD678A123ECB275E4755C463C` |
| `ai-training/experiments/r4_deletion_research_closure/artifact_hashes.json` | `3E21936C6175F9DEA5FAE3346E96EECD3AEF18732072C42B0B2FCAE4161D6174` |
| `ai-training/experiments/r4_deletion_research_closure/R4_DELETION_RESEARCH_CONCLUSION.md` | `8D6994F68E4B33C246EDF579D044A0D1FA701672D5AA27B915BEABBC89666237` |
| `ai-training/experiments/r4_deletion_research_closure/r4_final_status.json` | `DB6409A87C46CF66BFE76EEF98F8E0308C29FA5D3D40A600353F08CA8176024F` |
| `ai-training/experiments/r4_deletion_research_closure/r4_test_closure.json` | `A9707885B874DDCFDC341B5E5F182894289CF665C97F39A740003A58BC713D8A` |

**R4 TEST WAS NEVER ACCESSED.** No TEST path was resolved, audio read, posterior computed, hypothesis scored, or metric calculated. The closed R4 generation may not access R4 TEST. Any future deletion work must be a new research generation with a new hypothesis and an independently preregistered evaluation strategy; there is no automatic TEST or training authorization.

## 6. R5 current state — Addition

R5 is a sequence of completed local research stages, not unclassified scratch work. Its latest verified closure is R5-4A/R5-4B.

| Stage | Verified status | Scientific meaning |
|---|---|---|
| R5-0 | `R5_0_PASS_EXISTING_CTC_FEASIBLE` | Annotation support and frozen CTC representation were feasible for research; no detector confirmation |
| R5-1 initial execution | Technical stops for row accounting/CTC alignability | No scientific result from the stopped execution |
| R5-1A | `R5_1A_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS` (6/6 TRAIN gates) | Exact alignability-safe insertion scoring passed TRAIN development |
| R5-1B | `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED` (4/6 gates) | Continuous ranking transferred, but fixed TRAIN threshold produced excessive false additions on iterative VALIDATION |
| R5-1C/closure | `R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED` | Failure reflected mixed score shift/calibration and class overlap; not “model cannot hear addition” |
| R5-2A | `R5_2A_RELATION_COMPETITION_FEASIBLE` | Relation competition was feasible to test |
| R5-2B corrections/execution | `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED` (2/8 gates) | Valid TRAIN result after technical/path/environment-only corrections; robust threshold not authorized |
| R5-2C | `R5_2C_POSTHOC_FAILURE_AUDIT_COMPLETE` | Post-hoc failure audit only |
| R5-3A | `R5_3A_EVIDENCE_SEPARATION_GENERATION_CLOSED_NOT_CONFIRMED` (6/8 gates) | TRAIN discrimination existed, but substitution/deletion negative FAR gates failed |
| R5-4B | `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS` (10/10 materialization gates) | Complete candidate materialization/provenance passed; **not** scientific performance confirmation |
| R5-4A resumed audit | `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED` (1/4 gates) | Partial localization signal; Top-5, Top-10 and median-rank gates failed |

Latest verified R5 conclusion: R5-4B preserved a truth-blind 2,977,040-candidate TRAIN landscape across 16,582 words with exact BEST_INSERT reproduction. R5-4A then found Top-1 0.13486842105263158, Top-5 0.27960526315789475, Top-10 0.3881578947368421, median truth rank 19, and 1/4 feasibility gates. The frozen single-INSERT family did not justify a top-k or marginalization generation.

Important R5 identities:

| Artifact | SHA-256 |
|---|---|
| `ai-training/experiments/r5_1_addition_scoring_closure/artifact_hashes.json` | `C8E71EDE56902D594A60F1194ABF8A72AB0A7EFBE4F212F57BC71FCEF21B1D69` |
| `ai-training/experiments/r5_3a_evidence_separated_relation_scoring/R5_3A_CLOSURE_MANIFEST.json` | `9F57A293E0F4CA6E35E06761CFF54ABF70D967FE6ECF4F42ABFFF7C35397768C` |
| `ai-training/experiments/r5_4a_insert_candidate_recoverability/R5_4A_CLOSURE_MANIFEST.json` | `2E717A5ADEB6EEC35D645B7DF67FE24486960B9B87AAA6FFBEC4CDA3E8C3D94B` |
| `ai-training/experiments/r5_4b_insert_candidate_materialization/materialization_result/R5_4B_EXECUTION_MANIFEST.json` | `FD1E4E66168654EC54778A489E0B443BD9C691DD1392233C4284CDF6CDF07B11` |

R5 TEST remains untouched: no TEST path resolution, audio access, inference, or performance consumption is recorded. R5 TEST is not authorized for the closed generations above. R5-1B VALIDATION was consumed as iterative development evidence, not independent final confirmation.

## 7. R6 current state — Local boundary addition evidence

The research roadmap is reconciled in the same bounded documentation change set as this operational update. Frozen R6-0/R6-1 artifacts remain authoritative over roadmap prose.

### 7.1 R6-0

- Contract: `R6_0_LOCAL_BOUNDARY_ADDITION_FEASIBILITY_CONTRACT_FROZEN`
- Static result: `R6_0_LOCAL_BOUNDARY_STATIC_VERIFICATION_PASS`
- Scope: a distinct oracle-boundary TRAIN-feasibility mechanism using a fixed five-output-step window and `MEAN_UNEXPECTED_PHONE_MASS`; not whole-word INSERT rescoring and not a detector.
- R6-0 accessed no TRAIN audio/annotation content for performance, and accessed neither VALIDATION nor TEST.
- R6-0 explicitly preregistered and authorized R6-1 after static verification; that authorized step has now been executed.

### 7.2 R6-1

Final status: `R6_1_LOCAL_BOUNDARY_EVIDENCE_NOT_CONFIRMED`

| Field | Verified value |
|---|---:|
| Execution count | 1 |
| Split | TRAIN only |
| Neural training/fine-tuning | No / No |
| Checkpoint inference | Yes |
| Classifier fitting | No |
| Threshold search | No |
| VALIDATION accessed | No |
| TEST accessed | No |
| Addition event coverage | 308/342 = 0.9005847953216374 |
| Positive boundary coverage | 293/324 = 0.904320987654321 |
| Pooled boundary ROC-AUC | 0.5265224849834894 |
| Median of 12 speaker ROC-AUCs | 0.5469273349525517 |
| Speakers with ROC-AUC > 0.55 | 6/12 |
| Frozen gates | 0/4 passed |

Scientific interpretation: the frozen local boundary evidence mechanism did not achieve the required coverage or speaker-consistent ranking signal. Its scope was TRAIN boundary-level continuous feasibility only; it was not a word-level addition detector.

Important R6 identities:

| Artifact | SHA-256 |
|---|---|
| `ai-training/experiments/r6_0_local_boundary_addition_feasibility/R6_0_MANIFEST.json` | `EDB1A62CD6350AFC955C7A49B51C668BD0ED4217F7BBC9AD916525769DF8718A` |
| `ai-training/experiments/r6_0_local_boundary_addition_feasibility/R6_0_STATIC_MANIFEST.json` | `9EBF82AE11FA081E839837D9FFE5FDDD2D81BDD8E588EA5A3B840D722B1DD982` |
| `ai-training/experiments/r6_1_local_boundary_evidence_train_feasibility/R6_1_EXECUTION_MANIFEST.json` | `04CFCF2D6F882E53B9A025FF462A443EAB108E65543233FC29262B545F167BBF` |
| `ai-training/experiments/r6_1_local_boundary_evidence_train_feasibility/r6_1_boundary_scores.jsonl` | `A45B9560DDA67C3E0CD90E88E1100EB73058E281AC0FD2E5E001B592DBDA0DB4` |
| `ai-training/experiments/r6_1_local_boundary_evidence_train_feasibility/r6_1_final_status.json` | `AF8EBDBB61D7CE17ECECBFF480FCB2CCDF6A507702CF60390A29A5B20A75E314` |

The frozen R6-1 failure rule says to stop this mechanism. A later R6 experiment is not authorized by the inspected artifacts. Next authorized research step: `NONE`; a future step would require a separately named and preregistered generation.

## 8. TEST consumption register

| Research phase | TEST split | Access status | Allowed future use |
|---|---|---|---|
| R2 | `UNVERIFIED` | `UNVERIFIED` — R2 artifacts do not provide a phase-wide TEST-consumption register | Review frozen R2 artifacts before any TEST-related action; do not assume untouched |
| R3 | `ASI ERMS SKA THV TXHC YDCK` | **CONSUMED once under locked protocol** | Frozen reporting only; no retuning, threshold adjustment, checkpoint/feature selection, or post-hoc model changes |
| R4 | Same six speakers | **UNTOUCHED**; no paths/audio/posteriors/scores/metrics accessed | Closed R4 generation may not access TEST; only a new preregistered generation may define an independent strategy |
| R5 | Same six speakers | **UNTOUCHED**; no paths/audio/inference/performance access | Closed R5 generations may not access TEST; no automatic authorization |
| R6 | Same six speakers | **UNTOUCHED**; R6-1 was TRAIN-only and VALIDATION was also untouched | Current mechanism is stopped; no TEST or VALIDATION use is authorized |

## 9. Checkpoint register

Every checkpoint below is local-only/ignored and requires external preservation unless explicitly identified as a duplicate. Public downloaded Wav2Vec2 base-model files are excluded.

### 9.1 Selected R2/R3/R4 checkpoints

| Phase | Purpose | Relative path | Bytes | SHA-256 | Priority | External archive? |
|---|---|---|---:|---|---|---|
| R2A | Binary correctness selected validation checkpoint | `ai-training/experiments/r2a_correctness_seed42/R2A_binary_correctness_seed42_best_validation_macro_f1.pt` | 335,997 | `4E31A37471EC4F02ED5814A60FF5BD8F79FFB5BE5BC4915AE36AB360BBE78D44` | High | Yes |
| R2B | Audio+phone correctness selected checkpoint | `ai-training/experiments/r2b_audio_phone_seed42/R2B_audio_phone_binary_correctness_seed42_best_validation_macro_f1.pt` | 341,757 | `C80130A9BB533344521610752F8699FC1FBE684AE7FAC5258F05F5F8019ABE6C` | High | Yes |
| R3-1A | Development selected checkpoint | `ai-training/experiments/r3_1a_observed_phone_seed42/R3_1A_observed_phone_40class_seed42_best_validation_macro_f1.pt` | 353,953 | `9CF8382CE563AEB3B25AAC8383D61010F742CAC0297125E5F2E8FB409E312CC6` | High | Yes |
| R3-1B | Development selected checkpoint | `ai-training/experiments/r3_1b_observed_phone_seed42_24epochs/R3_1B_observed_phone_40class_seed42_best_validation_macro_f1.pt` | 354,721 | `107D538448FC25A2DCA1F3EB21F5B2CDEFD636D0D9596B3782F3BD3B143E86E8` | High | Yes |
| R3-1C | Development selected checkpoint | `ai-training/experiments/r3_1c_observed_phone_seed42_36epochs/R3_1C_observed_phone_40class_seed42_best_validation_macro_f1.pt` | 355,425 | `B3751D492572447B88565ADAFDA5BA6FC3748847C0A6C5E3F9899C74219F0174` | High | Yes |
| R3-1D | Final R3 acoustic checkpoint used for locked TEST | `ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt` | 355,745 | `5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E` | Critical | Yes |
| R4-1 | Controlled deletion selected checkpoint | `ai-training/experiments/r4_1_controlled_deletion_seed42/R4_1_controlled_deletion_audio_phone_seed42_best_validation_macro_f1.pt` | 342,091 | `4773F1ACFF2A20BAC81B68743FE052394DF4F7F844C924964D1CF2C5FC75B6EF` | High | Yes |
| R4-4B | CNN-only CTC selected checkpoint | `ai-training/experiments/r4_4b_ctc_sequence_seed42/R4_4B_ctc_phone_sequence_seed42_best_validation_per.pt` | 348,025 | `A154DFAC573D69B8ED1A71CBCDC23227EA3E80929890AD87E97ED85667142106` | High | Yes |
| R4-4C2 | Frozen CNN+BiGRU CTC used by R4/R5/R6 | `ai-training/experiments/r4_4c2_bigru_ctc_seed42/R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt` | 813,093 | `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085` | Critical | Yes |

### 9.2 Main-worktree Phoenix checkpoints

These paths are relative to the main worktree, not the research worktree.

| Purpose | Main-worktree relative path | Bytes | SHA-256 | Priority | External archive? |
|---|---|---:|---|---|---|
| Current default CNN-attention runtime model | `ai-training/models/l2_arctic_error_type_cnn_attention.pt` | 333,189 | `5547AAFE79DD02494A12D4103B1A28827BE7B06B27DD5FFC86995062E105595C` | Critical | Yes |
| Context stability seed-42 HQTV used by runtime-validation docs | `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt` | 337,027 | `51A49B27269D302BB8E05E8F50687A3E2A819753E2F0C51EDAB446C0D06F36C3` | High | Yes |
| Context-0.10 HQTV research checkpoint | `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_HQTV.pt` | 335,557 | `76B828B3B89FB97250B7A09D5A9422696E50AE1BF5E6B94AE61BF294A9D30FD9` | High | Yes |
| Wav2Vec2 attention fine-tune | `ai-training/models/l2_arctic_error_type_wav2vec2_attention.pt` | 377,596,209 | `A431429C5ED7BF311905DFB615F8E428B35A94CE09D3355C1CD4B866B3EF9C22` | High | Yes |
| Wav2Vec2 original-segment context fine-tune | `ai-training/models/l2_arctic_error_type_wav2vec2_context_original_segment.pt` | 377,593,522 | `709919903258A45C310CAB60ADFD148B0490DBD714FB5480E74BCCA88F1CDBDC` | High | Yes |
| Wav2Vec2 context-0.10 fine-tune | `ai-training/models/l2_arctic_error_type_wav2vec2_context_0_10.pt` | 377,592,646 | `4700A0CCFFF58780E29C5FB805D074CBDCD8B3EA9966B571F3862F3C89F5A8FE` | High | Yes |
| Wav2Vec2 context-0.15 fine-tune | `ai-training/models/l2_arctic_error_type_wav2vec2_context_0_15.pt` | 377,592,646 | `0617695725BEE579B402D129F91B5EF893443BA17D9070676B22CB34F87DA1D7` | High | Yes |
| Wav2Vec2 encoder fine-tune | `ai-training/models/l2_arctic_error_type_wav2vec2_encoder.pt` | 377,591,487 | `85B4B03FF1DEAC55858DB7349C725AD2C60389C64A53AA8B5549FF9B25A11EF9` | High | Yes |

Phoenix v2 selected scorer family is `cnn_attention_context`, but the exact deploy checkpoint is `UNVERIFIED`: the local-only selection commit requires a compatible `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` without pinning a file/hash, and the code default `ai-training/models/l2_arctic_cnn_attention_context_0_10.pt` does not exist. Do not infer that the seed-42 HQTV file is the final deploy checkpoint without a later explicit selection artifact.

Canonical duplicate policy:

- Canonical context-0.10 HQTV: `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_context_0_10_HQTV.pt`; `..._baseline_backup.pt` is byte-identical (`76B828...`).
- Canonical seed-42 HQTV: main-worktree `ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt`; `ai-worker/checkpoints/...` is byte-identical (`51A49B...`).
- Do not archive both copies unless deployment packaging explicitly requires it.
- Other local CNN/stability/ablation checkpoints under `ai-training/models/` remain scientifically mixed and need a future archive manifest; their final retention subset is `UNVERIFIED`.

## 10. Compact and large artifact register

### 10.1 Compact authoritative artifacts

Paths without an `ai-training/` prefix in this subsection are relative to `ai-training/experiments/`.

| Phase | Authoritative compact groups |
|---|---|
| R3 | `r3_1d_observed_phone_seed42_48epochs/{final_status.json,selected_validation_report.json,phone_vocab.json,model_metadata.json}`; `r3_2b_speaker_transfer_margin/{final_status.json,all_validation_threshold_candidate.json,oof_report.json,threshold_stability.json}`; all of `r3_locked_test_protocol/`; `r3_locked_test_evaluation/{artifact_hashes.json,freeze_verification.json,final_test_status.json,test_binary_metrics.json,test_acoustic_metrics.json,r3_locked_test_report.md}`; R3 docs under `ai-training/docs/` |
| R4 | Frozen preregistrations/contracts in `r4_3b_frozen_numerical_contract/`, `r4_4c1_bigru_ctc_preregistration/`, `r4_4d1_numerical_contract/`, and `r4_4d2a_train_only_robust_calibration/`; `r4_4d2b_final_target_validation/{artifact_hashes.json,final_status.json,validation_binary_metrics.json,validation_3class_metrics.json,continuous_auc_metrics.json,threshold_identity.json,r4_4d2b_report.md}`; all nine `r4_deletion_research_closure/` files; R4 history/thesis/docs manifest |
| R5 | R5-0 preregistration/source/status/hash files; R5-1A/B contracts/manifests/statuses and R5-1 closure; R5-2A decision; R5-2B correction/protocol/source/status manifests; R5-3A contract/execution/closure manifests; R5-4A contract/resume/closure manifests; R5-4B contract and execution manifests; R5 history/thesis/docs manifest |
| R6 | All compact files in `r6_0_local_boundary_addition_feasibility/`; `r6_1_local_boundary_evidence_train_feasibility/{r6_1_train_execution.py,r6_1_source_identity.json,r6_1_coverage.json,r6_1_primary_auc.json,r6_1_speaker_auc.json,r6_1_gate_results.json,r6_1_protocol_audit.json,r6_1_final_status.json,R6_1_EXECUTION_MANIFEST.json,R6_1_TRAIN_FEASIBILITY_RESULT.md}` |

### 10.2 Large scientific evidence

| Phase/group | Directory | Approximate size | Requirement | Checksum/manifest anchor |
|---|---|---:|---|---|
| R2 | `ai-training/experiments/r2*` | 8.875 MB total | Preserve selected checkpoints and canonical large CSV evidence | No phase-wide hash manifest found: `UNVERIFIED`; direct checkpoint hashes are recorded above |
| R3 | `ai-training/experiments/r3*` | 61.990 MB total | Preserve locked TEST predictions and large validation evidence externally | `r3_locked_test_evaluation/artifact_hashes.json` plus per-experiment compact identities |
| R4 | `ai-training/experiments/r4*` | 362.199 MB total | Preserve final and causal-chain CSV/JSONL/NPZ evidence externally | Per-experiment `artifact_hashes.json`; final anchor `r4_4d2b_final_target_validation/artifact_hashes.json` |
| R5 | `ai-training/experiments/r5*` | 1.233 GB total | Preserve large scoring/materialization evidence externally | R5 per-stage manifests; R5-4B execution manifest hashes all 12 candidate shards |
| R5-4B candidate landscape | `ai-training/experiments/r5_4b_insert_candidate_materialization/materialization_result/` | ~1.125 GB | Irreplaceable provenance/scientific materialization; must archive | `R5_4B_EXECUTION_MANIFEST.json` |
| R6 | `ai-training/experiments/r6*` | 82.607 MB total | Preserve boundary-level TRAIN evidence externally | `r6_1_local_boundary_evidence_train_feasibility/R6_1_EXECUTION_MANIFEST.json` |
| R6 boundary scores | `ai-training/experiments/r6_1_local_boundary_evidence_train_feasibility/r6_1_boundary_scores.jsonl` | 82,465,617 bytes | Must archive | SHA `A45B9560...DBDA0DB4` in R6-1 manifest |

Do not list or upload giant prediction/materialization files to ordinary Git. Git should retain their identities, purposes, statuses, and manifest hashes.

## 11. External archive register

`EXTERNAL_ARCHIVE_STATUS: NOT_CREATED`

Archive location: `UNVERIFIED — no independent archive exists`  
Archive ID: `UNVERIFIED — no archive has been created`  
Archive-wide checksum manifest: `UNVERIFIED — not created`

### Irreplaceable / must archive

- Selected R2/R3/R4 checkpoints and key Phoenix-created main-worktree checkpoints.
- V3/V4 derived CSVs.
- Large R2–R6 scientific evidence.
- R3 locked TEST outputs.
- R4 final/causal evidence and closure-linked large artifacts.
- R5-4B candidate materialization and R5-4A recoverability evidence.
- R6-1 boundary-score evidence.
- Reviewed Git-history export containing meaningful local-only commits/refs after secret isolation.

### Public / redownloadable or optional

- Downloaded public Wav2Vec2 base model: redownloadable; not a Phoenix-created checkpoint.
- Extracted L2-ARCTIC raw tree: recreatable from a retained archive or source access.
- L2-ARCTIC ZIP: optional retention for convenience/input identity; not unique scientific output.
- CMU ARCTIC raw data: public/redownloadable.
- Environments, dependencies, caches, build output and logs: recreate.

## 12. Raw-data policy

| Asset | Verified local path/state | Size | Policy |
|---|---|---:|---|
| L2-ARCTIC v5 extracted raw data | Main worktree `ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/` | 16,245,997,574 bytes | Recreate from source/archive after preservation verification |
| L2-ARCTIC v5 ZIP | Main worktree `ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0.zip` | 7,515,454,814 bytes | Optional keep for convenience/input identity; ZIP SHA-256 is `UNVERIFIED` because it was not computed in this task |
| CMU ARCTIC raw | Main worktree `ai-training/datasets/cmu-arctic/raw/` | 533,631,306 bytes | Public/redownloadable |

Raw public data is not a substitute for the unique V3/V4 derived datasets, their audits, speaker split, or frozen scientific evidence.

## 13. Secret exclusion and handling

The following local paths must never be committed or placed in an ordinary research archive:

- `.env`
- `ai-worker/.env`
- `fastapi-backend/.env`
- `frontend/.env`
- `fastapi-backend/.tokens.local.env`
- `.claude/settings.local.json`

The token environment path currently exists inside a local stash rather than as a live working-tree file. No secret values were read for this document. If still needed, secrets require a secure, separate backup. Sanitized `.env.example` templates are not substitutes for secret backup.

## 14. Local Git-history preservation risk

GitHub alone does not yet preserve all current local Git history.

- 123 local branches exist.
- Nine local branch names point to cached-origin-absent commit history: `feature/ai-error-model-improvement`, `feature/ai-train-error-context-v3`, `feature/ai-worker-forced-alignment`, `feature/ai-worker-mfa-backend-post-verification`, `feature/class-management-api`, `feature/frontend-vocabulary-read-ui`, `feature/pending-local-changes`, `feature/phoenix-v2-model-selection`, and `feature/seed-demo-data`.
- `feature/class-management-api` and `feature/seed-demo-data` currently share the same tip.
- Ten stashes exist; one includes a sensitive local token environment path.
- One `refs/codex/...` ref exists.
- Without reflogs, Git reports 31 unreachable commits, 122 unreachable trees, and 432 unreachable blobs.

Before cleanup, meaningful local-only history must be reviewed and either pushed as normal branches/commits or exported to a verified archive. Sensitive stash material must be isolated. Do not upload all stashes or dangling objects blindly.

## 15. Recovery procedure

1. Clone the verified GitHub origin on a clean machine.
2. Verify that the intended main and `research/phoenix-correctness` branch tips match the preservation record; restore a reviewed Git-history export if required.
3. Read this file before running any research code.
4. Restore the external archive into paths documented by its future recovery map.
5. Verify every restored file against archive SHA-256 manifests and the dataset/checkpoint identities in this document.
6. Restore V3/V4 derived datasets and their audits.
7. Restore selected R2/R3/R4 and Phoenix runtime/research checkpoints, retaining the canonical copies identified above.
8. Restore R2–R6 large evidence, especially R3 locked TEST outputs, R4 final evidence, R5-4B candidate shards, and R6-1 boundary scores.
9. Recreate Python/Node environments from repository configuration; do not restore caches or virtual environments as authoritative state.
10. Redownload public Wav2Vec2/L2-ARCTIC/CMU ARCTIC inputs if they were intentionally not retained, observing source licenses and version identities.
11. Restore still-needed secrets separately through secure local configuration; never copy values into Git or research manifests.
12. Run only non-evaluation integrity checks: file existence, JSON parsing, source-identity checks, SHA-256 verification, imports, and static tests that do not access VALIDATION/TEST data.
13. Do **not** rerun R3 locked TEST. Do **not** access R4/R5/R6 TEST under the closed generations.

## 16. Stale documents and unresolved identities

- `ai-training/docs/R5_DOCUMENTATION_UPDATE_MANIFEST.json` is preserved byte-for-byte as a historical R5-1 documentation snapshot and is superseded for current closure documentation by `ai-training/docs/R5_RESEARCH_CLOSURE_DOCUMENTATION_MANIFEST.json`.
- Live GitHub state was verified without fetch: `research/phoenix-correctness` exists at the compact-research preservation boundary recorded above; `feature/wav2vec2-demo-score` is absent as a named live branch, while its HEAD is preserved by `backup/full-demo-state-20260716`.
- External archive location, ID and archive-wide checksums: `UNVERIFIED` because the archive does not exist.
- L2-ARCTIC v5 ZIP SHA-256: `UNVERIFIED`; not computed in this task.
- Exact Phoenix v2 deploy checkpoint: `UNVERIFIED`; scorer family is selected, but no frozen selection artifact pins a compatible checkpoint path/hash.
- Final retention subset for the remaining main-worktree ablation/intermediate checkpoints: `UNVERIFIED`; preserve until a reviewed archive manifest resolves it.
- R2 TEST consumption: `UNVERIFIED`; no phase-wide authoritative register was found.
- R2 phase-wide artifact manifest: `UNVERIFIED`; no single hash manifest was found, although selected checkpoint hashes were verified directly.

## 17. Cleanup safety state

`CURRENT_LOCAL_DELETION_STATUS: NOT_SAFE_TO_REMOVE_YET`

Completed compact Git preservation:

- The initial MASTER_STATE was preserved.
- `research/phoenix-correctness` was created and pushed.
- V3/V4 compact builders, tests, contracts, audits, and documentation were preserved.
- R2 compact records were preserved.
- R3 compact development plus the locked TEST protocol/result record were preserved.
- R4 compact research history and deletion closure were preserved.
- R5 compact research history and closure were preserved.
- R6 compact local-boundary research history was preserved.

Still incomplete:

- The external archive is not created or verified.
- V3/V4 CSV datasets, required checkpoints, large R2–R6 evidence, R5 materialization, and R6 boundary scores remain local-only external-archive material.
- Still-required main-worktree local-only state has not been fully preserved.
- Local-only branches, stashes, `refs/codex`, and reflog/dangling history are not normalized or preserved where required.
- Sensitive local files and the sensitive stash require separate handling.
- A complete external-archive inventory and clean-machine safe-removal verification remain pending.

Local deletion may be reconsidered only after Git preservation, external archive creation, checksum verification, sensitive backup handling, and a clean-machine recovery check are complete.
