# AI Worker Integration Status

## Completed

- Normalized AI result contract.
- CNN Attention scorer integration.
- Clip-level and aligned CNN Attention inference paths.
- Alignment contract.
- Approximate fallback aligner.
- MFA wrapper scaffold.
- TextGrid parser scaffold.
- Scoring contract.
- Heuristic GOP-like scoring scaffold.
- Hybrid diagnosis pipeline.
- Final AI output validator.
- Backend webhook payload builder.
- End-to-end worker dry-run demo.
- Backend integration dry-run demo.
- Documentation for output, webhook, scoring, MFA, hybrid diagnosis, and demos.

## Scaffolded

- MFA execution through local `mfa` command.
- TextGrid parsing for common word/phone tiers.
- `heuristic_gop` scoring as a placeholder segmental scoring layer.
- Hybrid severity selection based on diagnosis and heuristic scoring.

These pieces are intentionally marked as scaffold or heuristic where appropriate.

## Future Work

- Real MFA setup and validation on safe local audio.
- Real GOP/CaGOP acoustic posterior or likelihood scoring.
- Calibration of score and severity thresholds.
- Backend migration for a dedicated `ai_result` JSONB column if needed.
- Frontend display mapping for top issues and segment feedback.
- Speaker-independent evaluation and error analysis.

## Selected Model

Current selected model:

```text
CNN Attention phone error classifier
```

Classes:

- addition
- deletion
- substitution

Metrics:

- mean test macro F1 = 0.5124 +/- 0.0214
- mean test addition F1 = 0.1938 +/- 0.0415

Default checkpoint path:

```text
ai-training/models/l2_arctic_error_type_cnn_attention.pt
```

The checkpoint is local and not committed to Git.

## Current Output Path To Backend

```text
AI Worker normalized result
  -> validate_ai_result(...)
  -> build_success_webhook_payload(...) or build_failed_webhook_payload(...)
  -> validate_webhook_payload(...)
  -> POST /practice/webhook/ai-result
  -> practice_history update
```

Current backend compatibility relies on legacy fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The full normalized result is preserved under `feedback.ai_result`.

## Recommended Next Practical Test

Run the backend locally, create a safe test practice job, then execute:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
python ai-worker/scripts/demo_backend_integration.py --job-id <existing-practice-history-job-id> --post
```

Then verify:

- backend returns a 2xx response
- `practice_history.status` becomes `completed`
- `practice_history.score` is populated
- `practice_history.feedback.ai_result` contains the normalized AI result

Do not use production data or commit secrets.
