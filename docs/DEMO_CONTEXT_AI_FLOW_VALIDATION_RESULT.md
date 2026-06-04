# Demo Context AI Flow Validation Result

## 1. Purpose

This document records the safe validation result of the context AI flow after the demo context AI flow checklist was executed.

The recorded facts are limited to validation-safe identifiers, scorer mode, status values, and non-secret runtime results. This document does not include service-role keys, webhook secrets, signed audio URLs, audio files, or model checkpoints.

## 2. Validation Scope

The validated flow was:

```text
Frontend / backend-created practice job
-> PGMQ practice_jobs message
-> AI Worker using SCORER_MODE=cnn_attention_context
-> context CNN Attention inference
-> backend webhook update
-> PGMQ archive
-> practice_history result available for frontend/history
```

## 3. Environment

- `worker_mode=once`
- `scorer_mode=cnn_attention_context`
- `context_mode=context_0_10`
- Backend webhook returned `200`
- No secrets recorded
- No signed audio URLs recorded

## 4. Validation Evidence

### PGMQ script validation

- `msg_id=25`
- `job_id=ab74fb36-9408-4386-b1fe-3e89d3e2801f`
- `inference_ran=true`
- `ai_result_valid=True`
- `payload_valid=True`
- `post_success=True`
- `archive_success=True`
- `context_mode=context_0_10`
- `context_used=true`

### Real worker.py once validation

- `msg_id=26`
- `job_id=aeaf098f-64cf-4cbd-8469-1d245fd5d93a`
- `target_word=Architecture`
- `scorer_mode=cnn_attention_context`
- `model_confidence=0.5814686417579651`
- `webhook_status_code=200`
- Message archived successfully

## 5. What Was Verified

- Queue message could be read
- Context scorer was used
- AI result was built
- Backend webhook accepted result
- `practice_history` was updated
- Queue message was archived after processing

## 6. Known Warnings

- PySoundFile may fail on frontend WebM and librosa may fall back to audioread.
- This is not fatal because processing still completed.
- Audio preprocessing optimization can be future work.

## 7. Safety Notes

- No secrets committed
- No signed URLs committed
- No audio files committed
- No checkpoints committed
- Classifier confidence is not pronunciation correctness
- Heuristic score is not real GOP
- Fallback alignment is approximate

## 8. Demo Conclusion

The context AI flow is validated end-to-end for demo purposes using the current research candidate, but future AI improvement should still address real forced alignment, real GOP/CaGOP, dataset expansion, and stronger model evaluation.
