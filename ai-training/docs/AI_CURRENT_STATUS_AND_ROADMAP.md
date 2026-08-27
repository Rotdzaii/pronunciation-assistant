# AI Current Status and Roadmap

## 1. Purpose

This document clarifies what the current AI can do, what it cannot do yet, and what future AI phases should improve. It is intended for the capstone report and for future development planning after the current context AI flow validation.

## 2. Current AI Position

The current AI is a working research candidate integrated into the application pipeline, not a fully final production-grade pronunciation assessment model.

The selected current candidate is `CNN Attention with context_0_10`.

## 3. What Has Been Completed

- Dataset preparation using L2-ARCTIC
- Vietnamese-only baseline
- All-speaker expansion
- CNN baseline
- CNN Attention
- Speaker-disjoint evaluation
- Addition-focused experiment
- `context_0_10` experiment
- Multi-seed stability check
- AI Worker integration
- Backend webhook validation
- PGMQ validation
- `worker.py` once validation
- Demo validation checklist/result

## 4. Selected Candidate

The selected candidate is:

`CNN Attention with context_0_10`

Vietnamese speaker-disjoint multi-seed stability:

| Metric | Result |
|---|---:|
| Macro F1 | `0.5170 ± 0.0338` |
| Addition F1 | `0.1251 ± 0.0473` |
| Accuracy | `0.6618 ± 0.0324` |

It was selected because it improved speaker-disjoint macro F1 and addition F1 compared with the baseline while keeping the architecture lightweight enough for current system integration.

## 5. What The Current AI Can Do

- Classify broad phone error types: addition, deletion, and substitution
- Run as AI Worker scorer mode `cnn_attention_context`
- Process queued practice jobs
- Return structured AI results to the backend
- Preserve context metadata
- Support the teacher/student feedback pipeline

## 6. What The Current AI Cannot Fully Do Yet

- Cannot claim full Vietnamese-accent pronunciation diagnosis
- Cannot determine pronunciation correctness or a user-facing 0-100 score
- Cannot rely on fallback alignment as precise phone timing
- Cannot treat classifier confidence as pronunciation correctness
- Cannot fully explain all pronunciation errors
- Cannot guarantee robust performance for all unseen Vietnamese learners

## 7. Why Training Looked Lightweight

Training looked lightweight because the current model is CNN Attention, not a large speech foundation model. The input segments are short, the dataset size is limited, and the project did not train Wav2Vec2 or Whisper from scratch.

The current goal was to establish a baseline research candidate and integrate it into the system pipeline. Heavier AI work remains future work.

## 8. Runtime Status

The current AI Worker runs with CPU torch. CPU inference is fast based on the local benchmark, and the current bottleneck appears to be audio decoding/preprocessing, especially WebM frontend audio, rather than model inference.

Benchmark values:

- `total_runtime_mean ≈ 1.013766s`
- `audio_prepare_time_mean ≈ 0.930055s`
- `inference_time_mean ≈ 0.014268s`

GPU runtime is future work if inference becomes the bottleneck later.

## 9. Missing Pieces For A More Complete Pronunciation Model

1. Real forced alignment
2. Audited correct-phone samples and a learned correctness head
3. Quality labels suitable for a learned quality/scoring head
4. Larger and more diverse datasets
5. Speaker-independent and sentence-safe evaluation with more speakers
6. Stronger acoustic models or fine-tuning
7. Real user or teacher-reviewed evaluation

## 10. Recommended Phase 4 Roadmap

### Phase 4A — Real forced alignment

- MFA or equivalent
- TextGrid parsing
- Phone-level timing reliability

### Phase 4B — Learned correctness modeling

- Add correct examples alongside addition, deletion, and substitution labels.
- Train and validate a correctness head while keeping the CNN + Attention +
  Context architecture family as the baseline.
- Do not publish heuristic, GOP/CaGOP, or classifier confidence as a score.

### Phase 4C — Learned quality/scoring research

- Collect phone- or word-level quality labels with a defined rubric.
- Train and validate an ordinal or regression quality head only when those
  labels and a leakage-safe evaluation protocol exist.
- Until then, publish `score: null` and `score_type: "unavailable"`.

### Phase 4D — Dataset expansion

- Add more public pronunciation datasets if available
- Normalize schema
- Avoid simply mixing incompatible labels

### Phase 4E — Stronger model experiment

- Wav2Vec2/HuBERT/Whisper encoder fine-tuning or feature extraction
- Compare against current CNN Attention context candidate

### Phase 4F — Runtime optimization

- Audio decode/preprocessing improvement
- Local audio vs signed URL benchmark
- GPU only if inference becomes bottleneck

## 11. Capstone-Friendly Summary

Architecture update: the legacy paragraph below predates the accepted Deep
Learning First decision. GOP/CaGOP is not the current roadmap. MFA is forced
alignment only; current public output uses `score: null` and
`score_type: "unavailable"` until a learned quality scorer is trained and
validated.

Mô hình AI hiện tại của hệ thống là một ứng viên nghiên cứu đã được kiểm chứng và tích hợp end-to-end vào quy trình ứng dụng, từ hàng đợi xử lý, AI Worker, webhook backend đến kết quả hiển thị cho người dùng. Ứng viên hiện tại sử dụng CNN Attention với ngữ cảnh `context_0_10` và cho kết quả ổn định hơn baseline trong đánh giá speaker-disjoint trên nhóm người học tiếng Việt. Tuy nhiên, mô hình này chưa được tuyên bố là mô hình chẩn đoán phát âm hoàn chỉnh ở mức production. Các hướng phát triển tiếp theo cần tập trung vào forced alignment thật, GOP/CaGOP thật, mở rộng dữ liệu, thử nghiệm mô hình âm học mạnh hơn và tối ưu hóa runtime, đặc biệt là bước giải mã và tiền xử lý âm thanh.

## 12. Current Research-Generation Roadmap

This section records the newer Phoenix pronunciation-relation research status. It does not erase the legacy runtime and deployment history above.

| Generation | Research scope | Current status | Next decision |
|---|---|---|---|
| R3 | Correct and substitution | **TEST CONFIRMED** (`TEST_TRANSFER_CONFIRMED`) | Frozen reporting only; TEST was consumed once under the locked protocol and may not be rerun or used for retuning |
| R4 | Deletion | **CLOSED — NOT CONFIRMED** (`R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED`) | The closed generation may not access TEST; future work requires a separately named, preregistered generation with an independent evaluation strategy |
| R5 | Addition | **R5-0 FEASIBILITY PASS; R5-1/R5-3A/R5-4A CLOSED — NOT CONFIRMED; R5-2 DEVELOPMENT NOT CONFIRMED; R5-4B MATERIALIZATION CLOSED — TECHNICAL/PROVENANCE PASS ONLY** | No top-k/marginalization generation is justified; review the complete evidence before considering a fundamentally different mechanism |
| R6 | Oracle-boundary local posterior evidence for addition detection | **R6-0 CONTRACT FROZEN / STATIC PASS; R6-1 NOT CONFIRMED — TRAIN only, 0/4 gates** | Current mechanism stopped; no successor was automatically authorized; continuation requires a separately named, preregistered generation |
| R7 | Runtime integration | Planned | Integrate only validated, explicitly scoped capabilities |

R3 TEST was consumed exactly once under the frozen locked protocol. The preserved record includes the development chain, frozen TEST protocol, and locked result. No threshold search, retraining, checkpoint change, preprocessing change, or score remapping occurred after TEST access. R3 TEST must not be rerun or used for retuning.

R4's final method used a CNN+BiGRU CTC sequence model, TARGET-normalized CTC hypothesis scoring, and TRAIN-speaker-LOSO threshold `0.16184102947061696`. Meaningful deletion-related signal existed and the continuous TARGET score transferred above chance, but robust fixed-threshold deletion decision performance was not confirmed. Final R4-4D2B validation produced Binary Macro-F1 0.652102 and deletion F1 0.331390, passing exactly 3 of 8 frozen gates.

R4 TEST was never accessed and remains untouched. The closed R4 generation may not access TEST. Future deletion research requires a separately named, preregistered research generation with an independent evaluation strategy. See [R4_DELETION_RESEARCH_HISTORY.md](R4_DELETION_RESEARCH_HISTORY.md) and [R4_DELETION_THESIS_SUMMARY.md](R4_DELETION_THESIS_SUMMARY.md).

## 13. R5 Addition Research Status

R5-0 established sufficient addition data and runtime sequence feasibility. R5-1 developed exact INSERT-vs-KEEP evidence but failed fixed-threshold VALIDATION transfer. R5-2 tested hard competition with SUB/DELETE explanations; it reduced relation-specific false additions but over-suppressed true Additions and was not confirmed. R5-3A separated `[A,S,D]` evidence in a fixed linear fusion; it recovered aggregate TRAIN discrimination but failed the substitution- and deletion-negative FAR gates. R5-4B then materialized the complete frozen INSERT candidate landscape with exact BEST_INSERT reproduction. The resumed R5-4A feasibility audit found partial localization signal but only 38.8% Top-10 recovery and median truth rank 19, failing three of four gates.

| Stage | Status | Meaning |
|---|---|---|
| R5-0 | **FEASIBILITY PASS** | Addition annotations and speaker-disjoint support were sufficient; greedy insertion was directional but noisy. |
| R5-1 | **TECHNICAL STOP BEFORE METRICS** | Impossible CTC INSERT targets exposed an unsafe `zero_infinity=True` interpretation; no scientific performance result. |
| R5-1A | **TRAIN DEVELOPMENT PASS (6/6)** | Alignability-safe exact CTC scoring passed the frozen TRAIN gates; theta `0.7485884030659993` was frozen. |
| R5-1B | **VALIDATION TRANSFER NOT CONFIRMED (4/6)** | Ranking evidence transferred, but Binary Macro-F1 and correct-only false-addition control failed. |
| R5-1C | **POST-HOC AUDIT COMPLETE** | Exploratory result: mixed speaker calibration/location shift and class overlap. |
| R5-1D | **CURRENT GENERATION CLOSED — NOT CONFIRMED** | Do not open TEST or continue this scoring generation. |
| R5-2B / R5-2C | **DEVELOPMENT NOT CONFIRMED / FAILURE AUDIT COMPLETE** | Hard relation competition passed only G6/G7; useful confounder suppression was offset by true-Addition over-suppression and threshold compensation. |
| R5-3A | **CLOSED — NOT CONFIRMED (6/8)** | Aggregate AUC/decision quality recovered, but SUB FAR 0.052579 and DELETE FAR 0.099073 failed G6/G7; no robust threshold. |
| R5-4B | **MATERIALIZATION CLOSED — PASS (10/10)** | Complete 2,977,040-candidate TRAIN landscape preserved; this is provenance success, not scientific performance confirmation. |
| R5-4A | **RECOVERABILITY CLOSED — NOT CONFIRMED (1/4)** | Exact mapping passed, but Top-5, Top-10, and median-rank gates failed. |
| Future R5 generation | **UNDEFINED / NOT AUTHORIZED** | The current evidence does not justify top-k or marginalization of the same INSERT family. Any future work requires a fundamentally different mechanism, new name, preregistration, consumed-VALIDATION disclosure, and preserved TEST. |

Latest diagnostic conclusion: `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED`. Technical materialization closure: `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS`.

R5 VALIDATION addition performance was consumed in R5-1B and is no longer untouched independent confirmation. R5 TEST speakers ASI, ERMS, SKA, THV, TXHC, and YDCK remain untouched; no TEST audio, inference, or performance was consumed. See [R5_ADDITION_RESEARCH_HISTORY.md](R5_ADDITION_RESEARCH_HISTORY.md) and [R5_ADDITION_THESIS_SUMMARY.md](R5_ADDITION_THESIS_SUMMARY.md).

## 14. R6 Local-Boundary Addition Evidence Status

R6 tested oracle-boundary local posterior evidence for addition detection. Its frozen primary score was mean unexpected-phone mass over a five-output-step window around the mapped gold/manual expected boundary. This was a TRAIN boundary-level continuous-feasibility mechanism, not a word-level addition detector and not a generic overall scoring or reliability project.

| Stage | Status | Meaning |
|---|---|---|
| R6-0 contract | `R6_0_LOCAL_BOUNDARY_ADDITION_FEASIBILITY_CONTRACT_FROZEN` | The local-boundary hypothesis, population, features, gates, and evaluation policy were frozen. |
| R6-0 static verification | `R6_0_LOCAL_BOUNDARY_STATIC_VERIFICATION_PASS` | Two byte-identical static runs passed 36/36 tests each and authorized exactly one R6-1 TRAIN-only execution. |
| R6-1 | `R6_1_LOCAL_BOUNDARY_EVIDENCE_NOT_CONFIRMED` | The frozen TRAIN-only mechanism passed 0/4 feasibility gates and stopped. |

R6-1 executed once on TRAIN over 16,582 words and 74,426 boundaries: 324 positive boundaries, 74,102 negative boundaries, and 342 addition events, of which 308 were covered. VALIDATION and TEST remained untouched. The current mechanism stopped; no successor experiment was automatically authorized. Any continuation requires a separately named, preregistered research generation.

The roadmap remains:

- R3 correct/substitution: **TEST CONFIRMED**; TEST was consumed once under the locked protocol and is frozen for reporting only, with no rerun or retuning.
- R4 deletion: **CLOSED — NOT CONFIRMED**; meaningful continuous signal existed, robust fixed-threshold performance was not confirmed, and the closed generation may not access untouched TEST.
- R5 addition: R5-0 **FEASIBILITY PASS**; R5-1, R5-3A, and R5-4A **CLOSED — NOT CONFIRMED**; R5-2 **DEVELOPMENT NOT CONFIRMED** with post-hoc failure audit complete; R5-4B materialization **CLOSED — TECHNICAL/PROVENANCE PASS ONLY**; TEST remains untouched and unauthorized.
- R6 local-boundary addition evidence: R6-0 contract frozen/static verification pass; R6-1 TRAIN-only **NOT CONFIRMED (0/4 gates)**; VALIDATION and TEST remain untouched; the mechanism stopped and no successor was automatically authorized.
- R7 runtime integration: planned.
