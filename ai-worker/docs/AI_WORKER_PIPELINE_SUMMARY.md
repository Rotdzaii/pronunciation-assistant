# AI Worker Pipeline Summary

## 1. Purpose

This document summarizes the AI Worker pipeline after integrating the selected CNN Attention model, normalized output contracts, alignment/scoring scaffolds, hybrid diagnosis, final output validation, and backend webhook payload construction.

The goal is to keep the AI Worker output stable for backend/frontend integration while clearly separating diagnosis confidence from pronunciation scoring.

## 2. Selected Model

Selected integrated model: CNN Attention phone error classifier.

Phase 3 context candidate: CNN Attention with `context_0_10`.

Task: phone-level pronunciation error classification.

Classes:

- `addition`
- `deletion`
- `substitution`

Key metrics:

- mean test macro F1 = 0.5124 +/- 0.0214
- mean test addition F1 = 0.1938 +/- 0.0415

Default local checkpoint path:

```text
ai-training/models/l2_arctic_error_type_cnn_attention.pt
```

The checkpoint is a local artifact and is not committed to Git.

Context candidate metrics from Phase 2 Vietnamese speaker-disjoint multi-seed stability:

- mean macro F1 = 0.5170 +/- 0.0338
- mean addition F1 = 0.1251 +/- 0.0473
- mean accuracy = 0.6618 +/- 0.0324

The context candidate uses a separate worker scorer mode so the existing `cnn_attention` behavior remains compatible.

## 3. Pipeline Overview

Flow:

```text
job payload
  -> audio loading/preprocessing
  -> alignment service
  -> CNN Attention diagnosis
  -> scoring service
  -> hybrid diagnosis
  -> final AI result validation
  -> backend webhook payload
  -> backend practice_history update
```

Simple diagram:

```text
PGMQ job / demo job
      |
      v
AI Worker scorer mode
      |
      +-- mock / wav2vec2 legacy paths
      |
      +-- cnn_attention
      |
      +-- cnn_attention_context
            |
            v
      alignment_service
      fallback | mfa scaffold | none
            |
            v
      CNN Attention segment diagnosis
            |
            v
      scoring_service
      heuristic_gop | none
            |
            v
      hybrid diagnosis
            |
            v
      normalized AI result + validator
            |
            v
      webhook payload builder
            |
            v
      POST /practice/webhook/ai-result
```

## 4. Components

AI result contract

- Path: `ai-worker/app/contracts/ai_result_contract.py`
- Role: builds normalized completed/failed AI results.
- Status: implemented.
- Limitation: score can still be demo/heuristic until real GOP/CaGOP exists.

CNN Attention scorer

- Path: `ai-worker/app/scorers/cnn_attention_scorer.py`
- Role: loads the selected CNN Attention checkpoint, performs clip or aligned segment inference, and maps outputs into the normalized result.
- Status: implemented.
- Limitation: real inference requires local `torch`, audio dependencies, and checkpoint.

CNN Attention context scorer

- Path: `ai-worker/app/scorers/cnn_attention_scorer.py`
- Mode: `SCORER_MODE=cnn_attention_context`
- Role: runs the Phase 2 `context_0_10` candidate on context-expanded aligned segment crops while preserving original segment boundaries for user-facing location.
- Status: implemented.
- Limitation: requires local context checkpoint and alignment boundaries; fallback alignment remains approximate when MFA is unavailable.

Alignment contract

- Path: `ai-worker/app/contracts/alignment_contract.py`
- Role: normalizes alignment segment/result shape.
- Status: implemented.
- Limitation: alignment quality depends on provider.

Fallback aligner

- Path: `ai-worker/app/alignment/fallback_aligner.py`
- Role: approximate even-split alignment for demos/scaffolding.
- Status: implemented.
- Limitation: not real forced alignment and not precise.

MFA scaffold

- Path: `ai-worker/app/alignment/mfa_aligner.py`
- Role: wrapper for locally configured MFA execution.
- Status: implemented for local validation flow.
- Limitation: does not install MFA or provide models/dictionaries.

TextGrid parser

- Path: `ai-worker/app/alignment/textgrid_parser.py`
- Role: parses common MFA/Praat TextGrid word and phone tiers.
- Status: scaffolded/minimal parser.
- Limitation: targets common interval tiers only.

Scoring contract

- Path: `ai-worker/app/contracts/scoring_contract.py`
- Role: defines phone/word/utterance segmental scoring output.
- Status: implemented.
- Limitation: contract supports real GOP later, but current scorer is heuristic.

Heuristic GOP scorer

- Path: `ai-worker/app/scoring/heuristic_gop_scorer.py`
- Role: produces placeholder GOP-like scores from segment diagnosis, confidence, duration, and alignment status.
- Status: implemented scaffold.
- Limitation: not real GOP/CaGOP.

Hybrid diagnosis

- Path: `ai-worker/app/hybrid/hybrid_diagnosis.py`
- Role: selects top issues using diagnosis predictions and scoring output.
- Status: implemented logic layer.
- Limitation: severity thresholds are not calibrated.

Final output validator

- Path: `ai-worker/app/contracts/ai_result_validator.py`
- Role: validates normalized AI result shape and safety constraints.
- Status: implemented.
- Limitation: structural/safety validation only, not clinical or model-quality validation.

Webhook payload builder

- Path: `ai-worker/app/contracts/webhook_payload.py`
- Role: builds legacy-compatible backend webhook payloads and includes the full normalized result.
- Status: implemented.
- Limitation: current backend stores rich output through `feedback.ai_result`; a dedicated backend JSONB column can be added later.

## 5. Environment Variables

```dotenv
SCORER_MODE=mock|wav2vec2|cnn_attention|cnn_attention_context
CNN_ATTENTION_CHECKPOINT_PATH=C:\path\to\l2_arctic_error_type_cnn_attention.pt
CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH=C:\path\to\context-checkpoint.pt
CNN_ATTENTION_CONTEXT_MODE=context_0_10
CNN_ATTENTION_CONTEXT_LEFT_SECONDS=0.10
CNN_ATTENTION_CONTEXT_RIGHT_SECONDS=0.10

ALIGNMENT_MODE=fallback|mfa|none
ALLOW_ALIGNMENT_FALLBACK=true
MFA_COMMAND=mfa
MFA_DICTIONARY_PATH=C:\path\to\dictionary.dict
MFA_ACOUSTIC_MODEL_PATH=C:\path\to\acoustic-model.zip
MFA_TEMP_DIR=C:\path\to\temp

SCORING_MODE=heuristic_gop|none

NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_SECRET=replace-with-local-secret
```

Do not commit `.env`, service-role keys, webhook secrets, checkpoints, MFA models, dictionaries, or raw audio.

## 6. Output Contract

Final AI output fields:

- `status`
- `score`
- `score_note`
- `problem_phonemes`
- `predicted_error_type`
- `diagnosis`
- `feedback`
- `scorer`
- `metadata`

Webhook payloads also include:

- `job_id`
- `ai_result`
- rich top-level copies of `diagnosis`, `scorer`, and `metadata`

The full normalized AI result is preserved under `feedback.ai_result` for current backend compatibility.

Important: `diagnosis.diagnosis_confidence` is classifier diagnosis confidence, not pronunciation score.

For `cnn_attention_context`, metadata also includes context crop fields such as `context_mode`, `context_used`, `segment_start_time`, `segment_end_time`, `crop_start_time`, and `crop_end_time`.

When `ALIGNMENT_MODE=mfa` succeeds, final metadata should also preserve:

- `alignment_method=mfa`
- `alignment_status=success`
- `is_forced_alignment=true`
- `mfa_used=true`
- `textgrid_parse_success=true`
- `word_segments_count`
- `phone_segments_count`
- `fallback_alignment=false`

If MFA fails and fallback is allowed, final metadata should preserve the fallback reason without exposing local TextGrid or temporary paths.

Recorded safe local validation for this path passed with:

- `alignment_status=success`
- `alignment_method=mfa`
- `is_forced_alignment=true`
- `mfa_used=true`
- `mfa_exit_code=0`
- `textgrid_parse_success=true`
- `fallback_alignment=false`
- `word_segments_count=1`
- `phone_segments_count=9`
- `score=67.1`
- `predicted_error_type=deletion`
- `context_mode=context_0_10`
- `location_reliability=forced_alignment`
- `ai_result_valid=true`

## 7. Backend Webhook

Route:

```text
POST /practice/webhook/ai-result
```

Header:

```text
x-ai-webhook-secret
```

Legacy backend fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The current backend updates `practice_history` with those fields. The full normalized AI result is preserved under `feedback.ai_result`.

## 8. How to Run Demos

```powershell
python ai-worker/scripts/demo_ai_result_contract.py
python ai-worker/scripts/demo_alignment_contract.py
python ai-worker/scripts/demo_scoring_contract.py
python ai-worker/scripts/demo_hybrid_diagnosis.py
python ai-worker/scripts/demo_final_ai_output.py
python ai-worker/scripts/demo_backend_webhook_payload.py
python ai-worker/scripts/demo_cnn_attention_context_scorer.py
python ai-worker/scripts/demo_context_mfa_aligned_inference.py --dry-run
python ai-worker/scripts/demo_worker_end_to_end.py --dry-run
python ai-worker/scripts/demo_backend_integration.py --job-id demo-job-id --dry-run
```

Optional backend POST:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
python ai-worker/scripts/demo_backend_integration.py --job-id <existing-practice-history-job-id> --post
```

## 9. Current Limitations

- Fallback alignment is approximate.
- MFA must be installed and configured locally for real forced alignment.
- MFA-aligned context inference depends on local dictionary/acoustic model availability or named MFA models.
- `heuristic_gop` is not real GOP/CaGOP.
- Severity thresholds are not calibrated.
- CNN Attention confidence is diagnosis confidence, not pronunciation score.
- CNN Attention context confidence is also diagnosis confidence, not pronunciation score.
- Real CNN Attention inference requires local `torch` dependencies and checkpoint.
- Context scorer inference requires a local context checkpoint and aligned segment boundaries.
- Current backend stores rich AI result data inside `feedback.ai_result`.

## 10. Recommended Next Work

- Run a real backend POST test with an existing `practice_history` job.
- Configure `torch` and the CNN Attention checkpoint in the AI Worker virtual environment.
- Configure MFA locally with dictionary and acoustic model.
- Implement real GOP/CaGOP scoring.
- Calibrate hybrid severity and score thresholds.
- Run speaker-independent evaluation.
- Add frontend display mapping for `diagnosis.top_issues`, `problem_phonemes`, and `feedback`.
