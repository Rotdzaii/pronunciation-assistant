# Backend Webhook Contract

## Purpose

The AI Worker posts pronunciation job results to the backend webhook:

```text
POST /practice/webhook/ai-result
```

The current worker uses:

- `NODE_WEBHOOK_URL`
- `AI_WEBHOOK_SECRET`
- HTTP header `x-ai-webhook-secret`

## Legacy-Compatible Fields

Every payload must include fields the current FastAPI backend already accepts:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The existing backend writes these fields into `practice_history`.

## Rich AI Result Fields

The AI Worker payload also includes richer fields for future backend support:

- `predicted_error_type`
- `diagnosis`
- `scorer`
- `metadata`
- `ai_result`

`ai_result` contains the full normalized AI result object after local path sanitization. It is intended for debugging, analytics, and future calibration.

Because the current FastAPI request model only requires legacy fields, existing backend code can keep reading the legacy shape and ignore rich fields initially. The worker also embeds `ai_result` inside `feedback.ai_result` so the current JSONB `feedback` column can preserve the full result without a backend schema change.

## Success Payload Example

```json
{
  "job_id": "11111111-1111-1111-1111-111111111111",
  "status": "completed",
  "score": 60.0,
  "score_note": "Heuristic/demo score, not production GOP.",
  "problem_phonemes": ["EH", "G"],
  "feedback": {
    "summary": "He thong phat hien kha nang co loi bo am tai EH.",
    "tips": ["Luyen lai am hoac tu duoc danh dau voi toc do cham."],
    "ai_result": {}
  },
  "predicted_error_type": "deletion",
  "diagnosis": {
    "diagnosis_confidence": 0.84,
    "confidence_note": "Classifier confidence, not pronunciation correctness."
  },
  "scorer": {
    "name": "cnn_attention"
  },
  "metadata": {
    "alignment_used": true,
    "alignment_method": "fallback_even_split",
    "gop_used": false,
    "scoring_method": "heuristic_gop",
    "scoring_is_heuristic": true,
    "hybrid_used": true
  },
  "ai_result": {}
}
```

## Failed Payload Example

```json
{
  "job_id": "22222222-2222-2222-2222-222222222222",
  "status": "failed",
  "score": null,
  "problem_phonemes": [],
  "feedback": {
    "summary": "AI worker khong tao duoc ket qua chan doan phat am.",
    "tips": [],
    "ai_result": {}
  },
  "error_message": "demo failure",
  "ai_result": {}
}
```

## Safety Rules

Classifier confidence is diagnosis confidence, not pronunciation score. Frontend and backend code must not map `diagnosis.diagnosis_confidence` into `score`.

`heuristic_gop` is not real GOP. With heuristic scoring, `metadata.gop_used` remains `false` and `metadata.scoring_is_heuristic` is `true`.

Fallback alignment is approximate. Payload metadata must preserve `alignment_method`, `alignment_note`, or `location_reliability` so the UI does not overstate location precision.

The worker strips sensitive local artifact paths such as checkpoint paths, TextGrid paths, MFA temp/output directories, and local audio paths before constructing webhook payloads.

## Backend Storage Recommendation

For UI compatibility, keep storing:

- `score`
- `problem_phonemes`
- `feedback.summary`
- `feedback.tips`

For debugging and research, store the full normalized result JSON:

- `ai_result`
- `metadata`
- `diagnosis`
- `scoring`
- `segments` when available

The current backend can preserve the full result through `feedback.ai_result`. A future backend migration may add a dedicated JSONB column such as `ai_result`.

## Current Limitations

- `heuristic_gop` is scaffold scoring, not production GOP.
- Fallback alignment is approximate.
- MFA execution is scaffolded and requires local installation/configuration.
- Real GOP/CaGOP is not implemented.
