# R6-0 Local Boundary Addition Feasibility Contract

Status: `R6_0_LOCAL_BOUNDARY_ADDITION_FEASIBILITY_CONTRACT_FROZEN`

Date frozen: 2026-08-19

Generation: `R6-0_LOCAL_BOUNDARY_ADDITION_FEASIBILITY`

## 1. Upstream identity verification

The frozen R5 conclusion is preserved. Direct SHA-256 checks passed for the R5-4A contract, resumed-audit manifest, closure manifest, R5-4B execution manifest, V4 metadata, and frozen checkpoint. Direct checks also passed for the R5-1A execution manifest, R5-2B final execution manifest, and R5-3A closure manifest. All 46 entries referenced by the R5-4A closure, R5-4A resumed-audit, and R5-4B execution manifests were independently re-read and matched their recorded byte sizes and hashes.

- R5-4A: `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED`
- R5-4B: `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS`
- Frozen checkpoint SHA-256: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`
- V4 SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Frozen scorer SHA-256: `2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3`

No frozen R5 artifact was modified.

## 2. Why R6 is scientifically distinct from R5

R5 enumerated complete single-INSERT phone sequences and compared whole-word CTC likelihoods. R6-0 instead defines one boundary-level observation at an already mapped expected boundary and summarizes posterior behavior only at nearby output indices. It performs no INSERT candidate enumeration, no CTC path likelihood for an inserted sequence, no BEST_INSERT ranking, and no KEEP/SUB/DELETE fusion. If a future implementation reduces to whole-word INSERT rescoring, it must stop as `R6_0_MECHANISM_NOT_DISTINCT_FROM_R5`.

This is an oracle-boundary feasibility mechanism, not a final detector and not a claim that the checkpoint has a strictly local receptive field.

## 3. Acoustic checkpoint output

The frozen R4-4C2 model accepts full MFA word spans and computes centered 64-bin log-mel features at 16 kHz using a 25 ms Hann window and 10 ms hop. Its CNN downsamples time by two. A packed single-layer bidirectional GRU and linear head return raw pre-softmax logits shaped `[B,T_out,41]`, where classes 0-39 are the frozen canonical phones and class 40 is CTC blank. `log_softmax(logits, dim=-1)` and posterior probabilities are directly obtainable before any CTC reduction.

For a word crop with `L` samples:

- `T_feature = floor(L / 160) + 1`
- `T_out = floor(T_feature / 2)`
- nominal output spacing = 20 ms

The CNN time-index receptive field is 16 mel frames (nominally about 160 ms), but the bidirectional GRU makes each output logit dependent on the entire unpadded word sequence. Accordingly, R6 localizes which output indices are aggregated; it does not claim the indexed logits are context-free local acoustics.

Historical R5 drivers already receive these logits and derive log-probabilities in memory, while the frozen R5 result artifacts preserve word/candidate scores rather than complete frame tensors. R6-0 required no checkpoint load or inference.

## 4. Boundary provenance

The recommended research reference is the established GOLD/manual L2-ARCTIC annotation, represented by the frozen V4 row provenance and the manual `phones` tier:

`MANUAL_TEXTGRID_ROOT/<speaker>/annotation/<utterance_id>.TextGrid`

`MANUAL_TEXTGRID_ROOT` is `C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0`.

TRAIN existence audit: 1,799 expected manual TextGrids, 1,799 present. Times are Praat seconds. V4 preserves stable row order, start/end times, clean relation labels, and canonical expected/observed phones. Frozen R5 mapping already establishes 342/342 runtime Addition events across 323 positive words. Full future boundary/window coverage remains a preregistered gate rather than an assumed result.

MFA provenance is:

`AUDIO_ROOT/<speaker>/textgrid/<utterance_id>.TextGrid`

The same raw root contains 13,448 MFA TextGrids across the 12 TRAIN speaker directories, and all 1,799 manually annotated TRAIN utterance stems have a matching MFA file. MFA word and phone tiers use seconds and are produced by the established forced-alignment path. MFA can map canonical expected phones through deterministic sequence alignment, but prior audits recorded `spn` labels and ambiguous mappings; it is not manual ground truth. MFA remains the historical source of the word audio crop and a later production/runtime consideration. It must not silently replace GOLD in the first feasibility experiment.

## 5. Exact acoustic-frame / boundary mapping

Let the expected phone sequence be `q[0:N]`, with GOLD/V4 intervals for its ordered expected rows. The oracle boundary time `tau_b` is:

- `b = 0`: start of `q[0]`
- `0 < b < N`: midpoint of the end of `q[b-1]` and start of `q[b]`
- `b = N`: end of `q[N-1]`

An unavailable or non-unique expected-row mapping is not imputed. With historical MFA word start `w0`, `t_rel = tau_b - w0`. The nominal center of output index `k` is `c_k = 0.020*k + 0.005` seconds relative to the word crop. Choose the index minimizing `abs(c_k - t_rel)`; an exact distance tie chooses the lower index. An anchor outside the MFA word crop is invalid rather than clipped.

The one frozen local window recommendation is output indices `k-2` through `k+2`, intersected with `[0,T_out-1]`. This is a five-step, nominal 100 ms bin window at the established 20 ms output spacing. Edge windows remain included with their available frames and their actual frame count recorded. No window search is authorized. Centered-STFT zero padding, the roughly 160 ms CNN receptive field, and whole-word BiGRU context must be reported as limitations.

## 6. Smallest local evidence feature set recommended

From `P = exp(log_softmax(logits, dim=-1))`, define the adjacent expected-phone set at boundary `b` from the available left and right expected phones.

Primary continuous score:

`MEAN_UNEXPECTED_PHONE_MASS = mean_t sum_{p in phones excluding adjacent expected phones} P[t,p]`

Two descriptive controls are frozen:

- `PEAK_UNEXPECTED_PHONE_POSTERIOR`: maximum posterior of an unexpected canonical phone over the window.
- `MEAN_NONBLANK_MASS`: mean `1 - P[t,blank]` over the window.

Posterior values are generated in the checkpoint's float32 path and cast to float64 only for deterministic summary reductions. No clipping, learned fusion, duration threshold, run-length threshold, entropy feature, candidate phone search, or feature selection is authorized in the first experiment.

## 7. Treatment of single, multiple, and mixed Addition cases

The frozen TRAIN word population is 16,582. It induces 74,426 expected-boundary instances. Frozen truth identifies 342 Addition events in 323 words and 324 unique positive boundary identities; 74,102 boundaries have no Addition event.

A boundary is positive exactly when at least one frozen Addition event maps to its `(source_identity, boundary_index)`. Event multiplicity is retained separately. All other boundaries are negative, including non-Addition boundaries inside Addition-positive words. No word, speaker, mixed relation, or difficult case is silently removed.

- Single-Addition words: 304; analyzed as one event and one positive boundary each.
- Multiple-Addition words: 19; 38 events; retained with boundary-level labels and event counts.
- Mixed substitution/addition words: 117; retained and tagged.
- Mixed deletion/addition words: 26; retained and tagged.
- Correct, substitution, deletion, and mixed non-Addition boundaries remain in the negative population with frozen cohort labels.

## 8. Exact next-stage feasibility experiment

The next separately authorized stage is `R6-1_LOCAL_BOUNDARY_EVIDENCE_TRAIN_FEASIBILITY`.

It will perform exactly one TRAIN-only inference pass with the frozen checkpoint and historical MFA word-crop preprocessing, map GOLD oracle boundaries using this contract, calculate only the three frozen local summaries, and persist boundary-level records. It will not train or fit a classifier, enumerate INSERT candidates, select a threshold, or construct a word-level detector.

The primary analysis is the fixed `MEAN_UNEXPECTED_PHONE_MASS` score. Report pooled boundary ROC-AUC, 12 per-speaker ROC-AUC values as speaker-held-out partitions with no fitting, median speaker AUC, count of speakers above 0.55, and positive-event plus positive-boundary mapping/window coverage. Single/multiple and mixed cohorts are descriptive only. No production threshold is authorized.

Before that inference, a synthetic/static stage must verify the mapping equations, window indexing, endpoint behavior, adjacent-phone exclusion, feature formulas, label independence, speaker partitioning, and absence of R5 whole-word scoring.

## 9. Preregistered next-stage success/failure gates

All four gates are required:

- F1: frozen Addition-event valid anchor/window coverage `>= 0.99` (denominator 342 events); also report coverage over 324 positive boundary identities.
- F2: pooled boundary ROC-AUC for `MEAN_UNEXPECTED_PHONE_MASS >= 0.65`.
- F3: median of the 12 TRAIN-speaker ROC-AUC values `>= 0.60`.
- F4: at least 9 of 12 TRAIN speakers have ROC-AUC `> 0.55`.

If 4/4 pass: `R6_1_LOCAL_BOUNDARY_EVIDENCE_FEASIBLE`, permitting only a separately named and preregistered development stage. If any valid gate fails: `R6_1_LOCAL_BOUNDARY_EVIDENCE_NOT_CONFIRMED`, and this mechanism stops. Undefined speaker AUC, missing coverage, identity failure, or mapping failure is a technical stop, not a scientific failure. These gates do not authorize a final model or threshold.

## 10. Protocol audit

- neural training: false
- fine-tuning: false
- classifier fitting: false
- performance metrics: false
- threshold search: false
- checkpoint loaded: false
- checkpoint inference: false
- TRAIN audio accessed: false
- TRAIN annotation content read: false
- frozen TRAIN score artifact read: structural population metadata only
- R5 artifacts modified: false
- scientific mechanism executed: false

## 11. VALIDATION / TEST status

VALIDATION dataset paths, rows, audio, boundaries, logits, and scores were not resolved or accessed. TEST remained untouched. Frozen source/configuration documentation was inspected only to establish historical semantics.

## 12. Artifact/hash audit

The self-excluding `R6_0_MANIFEST.json` records path, byte size, and SHA-256 for all eight R6-0 payload artifacts. Every entry was independently re-read after placement. Result: `HASH_AUDIT_PASS`.

## 13. Final status

`R6_0_LOCAL_BOUNDARY_ADDITION_FEASIBILITY_CONTRACT_FROZEN`

## 14. Next action

perform R6-0 synthetic/static verification of the frozen boundary mapping and local evidence formulas before any TRAIN inference.

