# Phoenix v2 Model Inventory

## A. Purpose

This inventory supports selecting a stable Deep Learning scorer for Phoenix v2 Stable. It records the current scorer modes, checkpoint references, training/evaluation scripts, and AI Worker runtime integration found in the repository.

This document is based on repository inspection only. It does not train models, download datasets, modify checkpoints, or change backend/frontend/AI Worker behavior.

Phoenix v2 remains Deep Learning-first, production-safe: the selected model should provide the main pronunciation score and pronunciation error diagnosis, while runtime safety layers protect deployment stability.

## B. Scorer Inventory

| Scorer mode | File path | Deep Learning role | Status | Phoenix v2 suitability | Notes |
|---|---|---|---|---|---|
| `mock` | `ai-worker/scorers/mock_scorer.py` | None. Deterministic placeholder. | Implemented and selected by default in `ai-worker/.env.example`. | Not suitable as Phoenix v2 Stable scorer. | Input is a practice job dict with `job_id` and `target_word`. Output is a completed mock result with score, `problem_phonemes`, feedback, and confidence metadata. Useful for plumbing only. |
| `wav2vec2` | Referenced by `ai-worker/worker.py` as `scorers.wav2vec2_scorer`; implementation file was not found in current `ai-worker/scorers/`. | Intended ASR baseline, not final pronunciation correctness scoring. | Referenced but not present in the inspected worker tree. | Not suitable without restoring implementation and validating runtime. | Existing docs describe Wav2Vec2 as transcript-driven. It may validate audio and ASR flow, but ASR confidence/text similarity is not phoneme-level pronunciation correctness. |
| `cnn_attention` | `ai-worker/app/scorers/cnn_attention_scorer.py` | Deep Learning phone error classifier using CNN Attention. | Implemented. Requires torch, librosa/numpy, audio input, and compatible local checkpoint. | Possible fallback candidate, but not the preferred Phoenix v2 Stable candidate if context scorer validates. | Input can be `audio_path` or `audio_url`; prompt text enables alignment and segment-level inference. Output follows normalized AI result shape with diagnosis, scorer metadata, and optional segment data. Current scoring metadata still marks demo/heuristic score paths. |
| `cnn_attention_context` | `ai-worker/app/scorers/cnn_attention_scorer.py` | Deep Learning phone error classifier using CNN Attention on `context_0_10` crops. | Implemented. Requires prompt text, alignment, context config, and compatible local context checkpoint. | Recommended if compatible checkpoint and runtime validation pass. | Existing docs identify it as the leading integrated research candidate. It uses aligned phone segments, expands each segment by 0.10s left/right, and preserves original segment boundaries for user-facing location. Confidence is diagnosis confidence, not pronunciation correctness. |

Additional scoring/alignment support found:

| Component | File path | Role | Phoenix v2 note |
|---|---|---|---|
| Heuristic GOP scorer | `ai-worker/app/scoring/heuristic_gop_scorer.py` | Produces placeholder GOP-like scores from diagnosis/alignment signals. | Safety/demo support only. It must not replace Deep Learning scoring as the Phoenix v2 Stable core. |
| Scoring service | `ai-worker/app/scoring/scoring_service.py` | Selects scoring behavior through `SCORING_MODE`. | Current supported behavior is scaffold-level; not a Deep Learning scorer. |
| Hybrid diagnosis | `ai-worker/app/hybrid/hybrid_diagnosis.py` | Combines diagnosis and scoring output into issue candidates. | Output formatting/support layer, not the model scoring core. |
| Alignment service | `ai-worker/app/alignment/alignment_service.py` | Selects `fallback`, `mfa`, or `none` alignment mode. | Timing support only. Alignment timing is not pronunciation correctness. |
| MFA aligner | `ai-worker/app/alignment/mfa_aligner.py` | Runs local MFA and parses TextGrid output. | Useful for phone timing when configured; not a scorer. |
| Fallback aligner | `ai-worker/app/alignment/fallback_aligner.py` | Approximate even-split alignment. | Demo/safety fallback only; limited location reliability. |

## C. Checkpoint Inventory

| Env var / config | Expected path | Compatible scorer | Tracked or local-only | Notes |
|---|---|---|---|---|
| `CNN_ATTENTION_CHECKPOINT_PATH` | Defaults to `ai-training/models/l2_arctic_error_type_cnn_attention.pt` when unset. | `cnn_attention` | Local-only. | Loader expects a checkpoint dict with `model_state_dict` and label metadata through `index_to_label`, `label_to_index`, or `error_to_label`. |
| `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` | Defaults in code to `ai-training/models/l2_arctic_cnn_attention_context_0_10.pt`; docs also reference compatible context families under `ai-training/models/`. | `cnn_attention_context` | Local-only. | Must match `SmallPronunciationCNNAttention`. Compatibility notes require attention keys such as `attention.score.*` and classifier keys such as `classifier.1.*`. |
| `CNN_ATTENTION_CONTEXT_MODE` | `context_0_10` | `cnn_attention_context` | Config only. | Current context scorer rejects unsupported context modes. |
| `CNN_ATTENTION_CONTEXT_LEFT_SECONDS` | `0.10` | `cnn_attention_context` | Config only. | Controls left crop expansion around aligned segment. |
| `CNN_ATTENTION_CONTEXT_RIGHT_SECONDS` | `0.10` | `cnn_attention_context` | Config only. | Controls right crop expansion around aligned segment. |
| Training script `MODEL_OUTPUT` values | Examples include `ai-training/models/l2_arctic_error_type_cnn_attention.pt`, `l2_arctic_all_speakers_cnn_attention.pt`, `l2_arctic_error_type_wav2vec2_context_0_10.pt`, and speaker-disjoint context checkpoint families. | Training/evaluation artifacts | Local-only. | `ai-training/.gitignore` ignores `models/*.pt`, `models/*.pth`, and `models/*.ckpt`. `git ls-files ai-training/models` returned no tracked model files during inspection. |
| `MFA_DICTIONARY_PATH` | Local dictionary path or model name | Alignment support | Local-only/config only. | MFA resource, not a scorer checkpoint. Do not commit dictionaries if they are local artifacts. |
| `MFA_ACOUSTIC_MODEL_PATH` | Local acoustic model path or model name | Alignment support | Local-only/config only. | MFA resource, not a scorer checkpoint. |
| `MFA_TEMP_DIR` | Local temp directory | Alignment support | Local-only/config only. | Must not leak local paths into webhook payloads. |

Local checkpoint files are present under `ai-training/models/` in the working directory, including CNN, CNN Attention, context, speaker-disjoint, and Wav2Vec2 artifacts. They are ignored local artifacts and should not be committed.

Checkpoint compatibility must be verified before Phoenix v2 model selection. The repository provides `ai-worker/scripts/inspect_context_checkpoints.py` for local context checkpoint inspection.

## D. Training and Evaluation Scripts

| Script | Purpose | Dataset | Output/metrics | Relevance to Phoenix v2 |
|---|---|---|---|---|
| `ai-training/scripts/train_l2_arctic_error_type_cnn_attention.py` | Train CNN Attention phone error classifier. | L2-ARCTIC phone error metadata. | Local `.pt` checkpoint; accuracy, macro F1, per-class metrics. | Relevant baseline model family for `cnn_attention`. |
| `ai-training/scripts/run_l2_arctic_cnn_attention_stability.py` | Run multi-seed CNN Attention stability evaluation. | L2-ARCTIC phone error metadata. | `cnn_attention_stability_metrics.json`, CSV summaries, seed checkpoints. | Relevant for model reliability evidence. |
| `ai-training/scripts/train_l2_arctic_all_speakers_cnn_attention.py` | Train all-speaker CNN Attention model. | All-speaker L2-ARCTIC metadata. | `l2_arctic_all_speakers_cnn_attention.pt`, evaluation metrics. | Relevant comparison, but existing docs warn non-disjoint Vietnamese subset may be optimistic. |
| `ai-training/scripts/evaluate_l2_arctic_all_speakers_cnn_attention.py` | Evaluate all-speaker CNN Attention checkpoint. | All-speaker L2-ARCTIC metadata. | Confusion matrix, accuracy, macro F1, per-class/per-speaker/per-L1 metrics. | Relevant evidence, not sufficient alone for stable selection. |
| `ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_cnn_attention.py` | Leave-one-Vietnamese-speaker-out CNN Attention evaluation. | L2-ARCTIC with held-out Vietnamese speakers. | Per-fold JSON/CSV, accuracy, macro F1, per-class F1, checkpoints. | Relevant generalization baseline. |
| `ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_cnn_attention_context.py` | Evaluate context-window CNN Attention variants. | L2-ARCTIC with held-out Vietnamese speakers. | Context result JSON/CSV, per-fold and aggregate metrics. | Highly relevant to `cnn_attention_context`. |
| `ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py` | Multi-seed stability check for context CNN Attention. | L2-ARCTIC with held-out Vietnamese speakers. | Stability summaries, per-seed/per-fold metrics, context checkpoints. | Most relevant existing training evidence for `cnn_attention_context`. |
| `ai-training/scripts/compare_vietnamese_speaker_disjoint_context_stability.py` | Compare context stability against baseline runs. | Existing evaluation JSON/CSV files. | `vietnamese_speaker_disjoint_context_stability_comparison.csv`. | Relevant for candidate comparison. |
| `ai-training/scripts/compare_vietnamese_speaker_disjoint_results.py` | Compare Vietnamese speaker-disjoint results. | Existing evaluation outputs. | `vietnamese_speaker_disjoint_comparison.csv`. | Relevant for reporting and model selection context. |
| `ai-training/scripts/compare_vietnamese_speaker_disjoint_context.py` | Compare context-window results against baselines. | Existing evaluation outputs. | `vietnamese_speaker_disjoint_context_comparison.csv`. | Relevant for context candidate evidence. |
| `ai-training/scripts/train_l2_arctic_error_type_wav2vec2_attention.py` | Train/fine-tune Wav2Vec2-based error type classifier head. | L2-ARCTIC phone error metadata. | Large local `.pt` checkpoint, accuracy, macro F1, per-class metrics. | Research comparison only for current Phoenix v2 selection because worker runtime implementation was not found. |
| `ai-training/scripts/train_l2_arctic_error_type_wav2vec2_context.py` | Train Wav2Vec2 context crop classifiers. | L2-ARCTIC phone error metadata. | Context checkpoints and metrics. | Research comparison only unless worker support is restored and validated. |
| `ai-training/scripts/evaluate_l2_arctic_error_type_wav2vec2_attention.py` | Evaluate Wav2Vec2 attention classifier. | L2-ARCTIC phone error metadata. | Confusion matrix, accuracy, macro F1, per-class metrics. | Relevant as baseline evidence, not current stable runtime candidate. |
| `ai-training/scripts/evaluate_l2_arctic_error_type_wav2vec2_context.py` | Evaluate Wav2Vec2 context classifiers. | L2-ARCTIC phone error metadata. | Context comparison metrics and confusion matrices. | Relevant as baseline evidence, not current stable runtime candidate. |
| `ai-training/scripts/train_l2_arctic_error_type_cnn.py` | Train earlier CNN error type classifier. | L2-ARCTIC phone error metadata. | Local `.pt`, accuracy, macro F1, per-class metrics. | Historical baseline, not preferred. |
| `ai-training/scripts/train_l2_arctic_error_type_cnn_v2.py` | Train CNN v2-style model. | L2-ARCTIC phone error metadata. | Local `.pt`, v2 metrics. | Historical baseline, not preferred. |
| `ai-training/scripts/train_l2_arctic_error_type_cnn_sampler.py` | Train sampler-based CNN variant. | L2-ARCTIC phone error metadata. | Local `.pt`, sampler metrics. | Not selected in existing docs due macro F1 tradeoff. |
| `ai-training/scripts/train_l2_arctic_addition_binary_cnn.py` | Train addition-vs-other binary CNN. | L2-ARCTIC phone error metadata. | Binary checkpoint and metrics. | Research support only; not a complete Phoenix v2 scorer. |
| `ai-training/scripts/train_l2_arctic_del_sub_binary_cnn.py` | Train deletion/substitution binary CNN. | L2-ARCTIC phone error metadata. | Binary checkpoint and metrics. | Research support only; not a complete Phoenix v2 scorer. |
| `ai-training/scripts/evaluate_l2_arctic_binary_stage_pipeline.py` | Evaluate two-stage binary pipeline. | L2-ARCTIC phone error metadata. | Pipeline confusion matrix and metrics. | Comparison only; not integrated as worker scorer mode. |
| `ai-training/scripts/run_l2_arctic_balancing_ablation.py` | Test balancing strategies. | L2-ARCTIC phone error metadata. | Ablation JSON/CSV and local checkpoints. | Research evidence; not stable scorer by itself. |
| `ai-training/scripts/run_l2_arctic_error_ablation.py` | Test error crop/context ablations. | L2-ARCTIC phone error metadata. | Ablation JSON/CSV and local checkpoints. | Research evidence for crop/context decisions. |
| `ai-training/scripts/infer_l2_arctic_error_type_cnn_attention.py` | Run inference demo for CNN Attention checkpoint. | L2-ARCTIC/demo audio metadata. | Prediction examples. | Useful for local validation, not training. |
| `ai-training/scripts/infer_l2_arctic_all_speakers_cnn_attention.py` | Run inference for all-speaker CNN Attention. | L2-ARCTIC/demo metadata. | Prediction examples. | Useful for inspection only. |
| `ai-training/scripts/demo_l2_arctic_error_type_predictions.py` | Demo prediction output. | L2-ARCTIC metadata. | Demo predictions CSV. | Useful for output inspection. |
| Metadata/build/review scripts under `ai-training/scripts/build_*`, `inspect_*`, `review_*`, `prepare_*`, `validate_*` | Build, inspect, prepare, or validate metadata/audio. | L2-ARCTIC, CMU ARCTIC, combined metadata. | CSV/JSON metadata and validation reports. | Supporting data pipeline only; not scorer candidates. |

## E. Worker Runtime Integration

`SCORER_MODE`:

- Defined in `ai-worker/worker.py`.
- Supported values found in code: `mock`, `wav2vec2`, `cnn_attention`, `cnn_attention_context`.
- Default in `ai-worker/.env.example`: `mock`.
- `worker.py` prints supported modes and selected mode at startup.

Scorer selection:

```text
worker.py
-> _load_env()
-> validate SCORER_MODE
-> _score(job, scorer_mode, confidence_threshold)
-> import selected scorer lazily where possible
```

Current scorer dispatch:

- `mock` calls `scorers.mock_scorer.score_pronunciation`.
- `wav2vec2` attempts to import `scorers.wav2vec2_scorer.score_pronunciation`; no matching file was found during inspection.
- `cnn_attention` imports `app.scorers.cnn_attention_scorer.score_pronunciation`.
- `cnn_attention_context` imports `app.scorers.cnn_attention_scorer.score_pronunciation_context`.

`ALIGNMENT_MODE`:

- Defined in `ai-worker/app/alignment/alignment_service.py`.
- Supported values: `fallback`, `mfa`, `none`.
- Default: `fallback`.
- `ALLOW_ALIGNMENT_FALLBACK` defaults to true.
- `mfa` uses `MFA_DICTIONARY_PATH` and `MFA_ACOUSTIC_MODEL_PATH` when configured.

Queue flow:

```text
Supabase PGMQ practice_jobs
-> read RPC: read_practice_job or pgmq_read
-> parse job fields: job_id, student_id, target_word, audio_url
-> selected scorer
-> webhook payload validation
-> backend webhook POST
-> archive RPC: archive_practice_job or pgmq_archive
```

Webhook flow:

- Required env vars: `NODE_WEBHOOK_URL`, `AI_WEBHOOK_SECRET`.
- POST target is usually `http://localhost:8000/practice/webhook/ai-result`.
- Header: `x-ai-webhook-secret`.
- Success payloads are built with `build_success_webhook_payload`.
- Failure payloads are built with `build_failed_webhook_payload`.
- Payload validation failure prevents POST and archive.
- Non-2xx webhook response prevents archive.

Worker mode:

- `WORKER_MODE=once` processes at most one queue message and exits.
- `WORKER_MODE=loop` keeps polling with backoff.
- Default in `.env.example`: `loop`.

Known runtime risks from current docs/code:

- `wav2vec2` mode is referenced but implementation file was not found in the inspected worker tree.
- `cnn_attention` and `cnn_attention_context` require local torch/audio dependencies and compatible local checkpoints.
- `cnn_attention_context` requires prompt text so alignment can create segment boundaries.
- Fallback alignment is approximate and has limited location reliability.
- MFA requires local installation/configuration and may fall back if allowed.
- Current scoring metadata can still include heuristic/demo scoring paths; Phoenix v2 selection must ensure the model remains the scoring core before stable deploy claims.
- WebM/frontend audio may make audio decoding/preprocessing the runtime bottleneck in local docs.
- Real backend/JWT/webhook runtime validation may still be needed for deployment.

## F. Recommended Candidate For Phoenix v2 Stable

Based only on repository inspection, `cnn_attention_context` is the recommended Phoenix v2 Stable candidate if compatible checkpoint and runtime validation pass.

Reasons:

- It is an implemented worker scorer mode.
- It is Deep Learning-based.
- Existing model-selection docs identify CNN Attention with `context_0_10` as the leading integrated research candidate.
- It has a dedicated checkpoint env var and context configuration.
- It has documented worker validation paths and runtime benchmark docs.
- It better matches the Phoenix v2 direction than `mock`, missing `wav2vec2` runtime support, or heuristic scoring support.

This recommendation is conditional. The next branch should verify checkpoint compatibility, dependency availability, runtime output shape, failure behavior, and backend webhook compatibility before selecting it as Phoenix v2 Stable.

## G. Risks / Open Questions

- Checkpoint compatibility must be verified for the exact deploy checkpoint.
- `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH` must point to a compatible local checkpoint.
- Local checkpoints should not be committed.
- Model confidence is not pronunciation correctness.
- MFA alignment supports timing but is not scoring.
- Fallback alignment is approximate and should not be presented as precise phone timing.
- Real JWT/backend runtime validation may still be needed.
- The current `wav2vec2` scorer mode may fail at runtime because the referenced implementation file was not found.
- Current scoring output may still include heuristic/demo score metadata; Phoenix v2 Stable must avoid presenting heuristic replacement scores as model scores.
- Dataset size and speaker coverage remain limited.
- Human-level pronunciation diagnosis should not be claimed.
