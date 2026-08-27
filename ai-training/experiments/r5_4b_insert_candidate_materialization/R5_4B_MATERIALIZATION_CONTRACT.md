# R5-4B Full INSERT Candidate Score Materialization Contract

## Purpose

R5-4B has one purpose: materialize every frozen exact-CTC single-INSERT candidate identity and authoritative TARGET score for every word in the frozen 16,582-word R5 TRAIN population. It is a diagnostic provenance stage, not a new scorer, performance rerun, recoverability audit, classifier experiment, or threshold experiment.

This contract stage performs no checkpoint inference, audio access, score materialization, truth-rank inspection, recoverability calculation, VALIDATION access, or TEST access.

## Frozen sources

- R5-4A contract SHA-256: `111DDB77EC09B177505AEEF7B260476D8C718F745AD6170E6969926841338A27`.
- R5-4A manifest SHA-256: `012322265AACD90009001AFCBF675C9228ADC55417B0A31029CD2A3EF32393B8`.
- R5-2B scorer: `ai-training/experiments/r5_2b_relation_competition/r5_2b_scorer.py`, SHA-256 `2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3`.
- R5-2B historical execution driver: `r5_2b_train_execution_driver_tc1_pa1_env.py`, SHA-256 `BC023CFD259406989CAEADABE3EB08AA945B7A612729ECA2C05711E0C387A5D0`.
- Frozen checkpoint SHA-256: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`.
- Frozen V4 SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`.

No frozen artifact may be modified.

## Authorized future execution

Exactly one later R5-4B execution may use the frozen research interpreter, load the frozen checkpoint, read TRAIN audio, reproduce the frozen R5-2B preprocessing/inference path, enumerate every INSERT candidate, score it with the historical batch CTC adapter, and write the contracted candidate shards. It may not train, fine-tune, change the scorer/checkpoint/runtime, use truth to alter materialization, calculate recoverability or word-level performance, fit a classifier, search a threshold, access VALIDATION, or access TEST.

## Population

Future materialization is limited to the exact frozen 16,582-word TRAIN population and speakers, in order: BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA. Stable source identities and row order must match frozen R5-2B TRAIN scores exactly. No exclusions are authorized.

## Candidate enumeration

For expected sequence `E` of length `N`, enumerate boundary `b = 0..N` in ascending order. Within each boundary enumerate canonical phone index `p = 0..39` in ascending order. Candidate target is `E[:b] + [p] + E[b:]`. Zero-based `candidate_index = 40*b + p`; total count is `40*(N+1)`. This is the exact historical list-comprehension order in the frozen scorer.

BEST_INSERT is the finite alignable candidate minimizing `(-TARGET_SCORE, boundary, phone_index)`: highest score, then lower boundary, then lower phone index.

## Score semantics

INSERT targets are nonempty. Alignability is `T >= len(H) + adjacent_repeat_count(H)`. Impossible candidates are not passed to CTCLoss and have mathematical TARGET score `-infinity`.

For alignable candidates, reuse the historical R5-2B path exactly: model logits and CPU `log_softmax` evidence are float32; batched acoustic tensors are float32; CTCLoss runs on the authorized CUDA device with blank 40, reduction `none`, and `zero_infinity=True`; loss is copied to CPU and converted to NumPy float64; `RAW_SCORE = -loss`; and authoritative `TARGET_SCORE = RAW_SCORE / float64(len(H))`. Hypothesis batch size remains 4096 and historical word/candidate batching order remains unchanged. No dtype, device, normalization, batching, or log-softmax change is authorized.

## Storage

Write 12 uncompressed UTF-8 JSONL shards named `r5_4b_insert_candidates_<SPEAKER>.jsonl`, one per TRAIN speaker. Use LF line endings, no BOM, one candidate per line, and Python standard-library `json.dumps` with `ensure_ascii=False`, compact separators, and `allow_nan=False`. Shards occur in frozen TRAIN-speaker order; within a shard, words retain frozen R5-2B row order and candidates retain ascending `candidate_index`.

Finite TARGET scores are persisted as both a JSON number and Python `float.hex` string; they must reconstruct to the identical binary64 Python float. Impossible scores use `null`, `null`, and `authoritative_insert_score_is_neg_inf=true`. No bare Infinity/NaN is permitted. The manifest and a materialization index independently hash every shard.

Candidate rows contain no truth labels, truth-candidate flags, rank, Top-k membership, recoverability field, or binary prediction.

## Reproduction guards

For every word, require `40*(N+1)` unique `(phone_index,boundary)` identities, with every candidate classified alignable or impossible. Global accounting must reproduce 2,977,040 total, 2,976,844 alignable, 196 impossible, 65 affected words, and zero words without a finite INSERT.

Derive BEST_INSERT from materialized rows using the frozen rule. Identity must match frozen R5-2B for 16,582/16,582 words. The reconstructed winning binary64 score must equal the frozen persisted R5-2B `best_insert_score_value` using exact `==`. R5-4B must reuse the exact R5-2B runtime and batch-scoring path; no tolerance is authorized. TC1's forward-error rule applies only to the independent empty-target diagnostic and is not applicable to nonempty INSERT scores. Any score mismatch stops the stage before PASS; the comparison policy must not be relaxed afterward.

## Materialization gates

M1-M10 are frozen in `r5_4b_materialization_gates.json`. All ten must pass for `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_PASS`. These are provenance/completeness gates, not scientific recoverability gates.

## Stop boundary

R5-4B execution must freeze and hash candidate artifacts before any R5-4A truth join. It must not calculate Top-1/3/5/10, truth rank, MRR, score gaps, phone/speaker/position recoverability, AUC, F1, FAR, event metrics, classifier outputs, or thresholds.
