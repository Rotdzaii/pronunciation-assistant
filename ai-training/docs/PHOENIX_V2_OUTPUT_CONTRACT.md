# Phoenix v2 Output Contract

## A. Purpose

This document defines the official Phoenix v2 Stable output contract for the selected deploy/demo candidate:

```dotenv
SCORER_MODE=cnn_attention_context
ALIGNMENT_MODE=mfa
MODEL_VERSION=phoenix_v2_stable
```

The contract ensures Phoenix v2 returns stable, predictable data to the AI Worker, backend, and frontend. It separates successful Deep Learning scorer output from production safety failure output so deploy/demo behavior is explicit and testable.

This document is documentation/schema guidance only. It does not train models, download datasets, modify checkpoints, or change worker runtime behavior.

## B. Success Output

A successful Phoenix v2 result means the Deep Learning scorer completed and produced a valid pronunciation result.

Required top-level fields:

- `status`: must be `completed`.
- `score`: number from `0` to `100`.
- `problem_phonemes`: array.
- `feedback`: object.

Required `feedback` fields:

- `model_version`: expected `phoenix_v2_stable`.
- `scorer_mode`: expected `cnn_attention_context`.
- `alignment_method`: expected requested/used alignment method, usually `mfa`.
- `is_forced_alignment`: boolean.
- `model_confidence`: number or object if available; omit or set `null` only when unavailable.
- `summary`: user-safe summary string.
- `details`: array of model-derived issue details.
- `warnings`: array if warnings are available.

Success constraints:

- `score` must be numeric and bounded from `0` to `100`.
- `problem_phonemes` must be an array, even when empty.
- `feedback.details` must be an array, even when empty.
- `model_confidence` is model diagnosis confidence, not pronunciation correctness.
- Local paths, checkpoint paths, signed URLs, TextGrid paths, and secrets must not appear in the payload.

## C. Failed Output

A failed Phoenix v2 result means the worker could not produce a valid model score. Failed fallback must not pretend to be a valid model score.

Required top-level fields:

- `status`: must be `failed`.
- `score`: must be `null`.
- `problem_phonemes`: must be `[]`.
- `feedback`: object.

Required `feedback` fields:

- `error_type`: one of the standard error types in this contract.
- `summary`: user-safe failure summary string.
- `model_version`: expected `phoenix_v2_stable`.
- `scorer_mode`: expected attempted scorer mode, usually `cnn_attention_context`.

Current backend behavior:

- `POST /practice/webhook/ai-result` accepts `score = null` when `status = failed`.
- The backend requires a numeric score only when `status = completed`.

Failure constraints:

- Do not emit a heuristic, default, confidence-derived, or placeholder score for failed model inference.
- Do not populate `problem_phonemes` from a failed inference path.
- Preserve enough failure metadata for debugging, but sanitize local paths, signed URLs, checkpoints, and secrets before webhook submission.

## D. Error Types

Standard Phoenix v2 `feedback.error_type` values:

| Error type | Meaning |
|---|---|
| `audio_decode_failed` | The worker could not decode the submitted audio into an inference-ready format. |
| `audio_preprocess_failed` | Audio decoded but preprocessing, normalization, trimming, or feature extraction failed. |
| `alignment_failed` | Requested alignment failed and no acceptable alignment/fallback path allowed scoring. |
| `checkpoint_missing` | The configured checkpoint path is missing or unavailable. |
| `checkpoint_incompatible` | The checkpoint exists but does not match the selected scorer architecture or expected `state_dict`. |
| `scorer_timeout` | Scorer execution exceeded the allowed runtime. |
| `scorer_failed` | The scorer crashed or returned invalid output. |
| `webhook_failed` | The backend webhook rejected or could not receive the result. |
| `unknown_error` | Unclassified failure after sanitization. |

## E. Deep Learning Boundary

Phoenix v2 Stable is Deep Learning-first.

On successful runs:

- `score` should come from the Deep Learning scoring path for the selected model.
- `problem_phonemes` should come from Deep Learning diagnosis and available segment/alignment metadata.
- `feedback.details` should identify model-derived issue candidates and their source.

Safety fallback only handles failure states. It can validate inputs and outputs, catch exceptions, enforce timeouts, sanitize payloads, and return safe failed results.

Heuristic or rule fallback must not replace model scoring in Phoenix v2 Stable. Classifier confidence must not be converted into pronunciation correctness.

If an intermediate demo path still produces a heuristic score, it must be explicitly marked as non-production and must not be claimed as Phoenix v2 Stable model scoring.

## F. Webhook Compatibility

Phoenix v2 output must remain compatible with:

```text
POST /practice/webhook/ai-result
```

Expected backend fields:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

Current FastAPI backend compatibility:

- `status` accepts `completed` or `failed`.
- `score` accepts `null` generally, but completed results are rejected if `score` is `null`.
- `problem_phonemes` must be a list.
- `feedback` must be an object.
- The backend stores `status`, `score`, `problem_phonemes`, and `feedback` on `practice_history`.

The worker may preserve richer debug/model data inside `feedback.ai_result` or related metadata, but the legacy-compatible top-level fields above must remain present.

## G. Example Payloads

Successful payload:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000001",
  "status": "completed",
  "score": 82.4,
  "problem_phonemes": ["EH", "K"],
  "feedback": {
    "model_version": "phoenix_v2_stable",
    "scorer_mode": "cnn_attention_context",
    "alignment_method": "mfa",
    "is_forced_alignment": true,
    "model_confidence": 0.81,
    "summary": "Phoenix v2 detected likely pronunciation issues in selected phonemes.",
    "details": [
      {
        "phoneme": "EH",
        "error_type": "substitution",
        "confidence": 0.81,
        "source": "deep_learning_scorer"
      }
    ],
    "warnings": []
  }
}
```

Failed scorer payload:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000002",
  "status": "failed",
  "score": null,
  "problem_phonemes": [],
  "feedback": {
    "model_version": "phoenix_v2_stable",
    "scorer_mode": "cnn_attention_context",
    "error_type": "scorer_failed",
    "summary": "Phoenix v2 could not produce a model score for this attempt.",
    "details": [],
    "warnings": ["No fallback score was generated."]
  }
}
```

Alignment unavailable but scorer succeeded payload:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000003",
  "status": "completed",
  "score": 76.0,
  "problem_phonemes": ["R"],
  "feedback": {
    "model_version": "phoenix_v2_stable",
    "scorer_mode": "cnn_attention_context",
    "alignment_method": "fallback",
    "is_forced_alignment": false,
    "model_confidence": 0.74,
    "summary": "Phoenix v2 completed scoring, but precise MFA timing was unavailable.",
    "details": [
      {
        "phoneme": "R",
        "error_type": "deletion",
        "confidence": 0.74,
        "source": "deep_learning_scorer"
      }
    ],
    "warnings": [
      "MFA alignment was unavailable; location reliability is limited."
    ]
  }
}
```

## H. Validation Checklist

Before deploy:

- Output JSON always includes required fields.
- `completed` status includes numeric `score`.
- `failed` status does not produce a fake model score.
- `feedback` contains `model_version` and `scorer_mode`.
- `problem_phonemes` is always an array.
- `feedback.details` is always an array when present.
- `feedback.error_type` is present for failed results.
- Webhook accepts payload.
- Frontend can render `feedback.summary` and `problem_phonemes` safely.
- Sensitive local/runtime values are absent from webhook payloads.
- Classifier confidence is not used as pronunciation correctness score.

## I. Next Step

Recommended next branch:

```text
feature/phoenix-v2-worker-hardening
```

Worker hardening should implement or verify timeout handling, exception mapping to the standard error types, payload sanitization, output validation, safe failed results, and queue archive behavior after webhook success only.

