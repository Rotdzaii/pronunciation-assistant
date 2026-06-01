# Context CNN Attention Backend Dry Run

## Purpose

This document describes the safe dry-run flow for validating `SCORER_MODE=cnn_attention_context` through the AI Worker backend payload path.

The dry run:

1. Generates temporary WAV audio when no audio file is provided.
2. Runs the context CNN Attention scorer.
3. Builds the normalized AI result.
4. Validates the AI result contract.
5. Builds the backend webhook payload.
6. Validates the webhook payload.
7. Prints summaries without POSTing by default.

No model training is performed.

## Environment Variables

The script sets the context defaults internally, but these are the relevant runtime values:

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\path\to\context-checkpoint.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
```

The checkpoint is local only and must not be committed.

## Dry-Run Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py --job-id demo-job-id
```

Optional checkpoint override:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py `
  --job-id demo-job-id `
  --checkpoint-path C:\path\to\context-checkpoint.pt
```

Optional audio input:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py `
  --job-id demo-job-id `
  --audio-path C:\path\to\audio.wav
```

## Optional Real POST

POST is disabled by default. To POST explicitly:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py `
  --job-id <existing-practice-history-job-id> `
  --post `
  --webhook-url http://localhost:8000/practice/webhook/ai-result `
  --secret <local-ai-webhook-secret>
```

Use `--post` only with a valid backend webhook target and a real existing job id.

## Validation Result

Validated command:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_context_backend_payload.py --job-id demo-job-id
```

Result:

- Inference ran: yes.
- POST attempted: no.
- `ai_result_valid=True`
- `payload_valid=True`
- `predicted_error_type=deletion`
- `diagnosis_confidence=0.45438268780708313`
- `score_note=Heuristic/demo score, not production GOP.`

Class probabilities:

| Class | Probability |
|---|---:|
| addition | 0.2529054582118988 |
| deletion | 0.45438268780708313 |
| substitution | 0.2927118241786957 |

The result is a runtime validation signal using generated audio. It is not a model-quality measurement.

## Context Metadata Location

The backend payload preserves the normalized AI result under:

```text
feedback.ai_result
```

Context metadata appears at:

```text
feedback.ai_result.metadata.context_mode
feedback.ai_result.metadata.context_used
feedback.ai_result.metadata.context_left_seconds
feedback.ai_result.metadata.context_right_seconds
feedback.ai_result.metadata.segment_start_time
feedback.ai_result.metadata.segment_end_time
feedback.ai_result.metadata.crop_start_time
feedback.ai_result.metadata.crop_end_time
```

Validated metadata:

```json
{
  "context_mode": "context_0_10",
  "context_used": true,
  "context_left_seconds": 0.1,
  "context_right_seconds": 0.1,
  "segment_start_time": 0.0,
  "segment_end_time": 0.15,
  "crop_start_time": 0.0,
  "crop_end_time": 0.25
}
```

## Limitations

- The checkpoint is local only and is not committed.
- CPU torch works but is slower than CUDA inference.
- Generated audio is only for plumbing validation.
- Fallback alignment is approximate and not real forced alignment.
- Heuristic score is not real GOP/CaGOP.
- Classifier confidence is not pronunciation correctness.
