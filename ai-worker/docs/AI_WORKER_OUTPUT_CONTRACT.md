# AI Worker Output Contract

## Purpose

The AI Worker returns a normalized result object before posting to the backend
webhook. This contract separates phone error diagnosis from a future learned
pronunciation score so the app-facing shape remains stable.

For the backend/frontend-ready final shape and examples, see `ai-worker/docs/FINAL_AI_OUTPUT_CONTRACT.md`.

Current selected model candidate: CNN Attention phone error classifier.

Selected model metrics:

- mean test macro F1: 0.5124 +/- 0.0214
- mean test addition F1: 0.1938 +/- 0.0415

## Confidence Is Not Score

The CNN Attention classifier predicts an error type and class probabilities. Its confidence is the probability or confidence of a diagnostic class such as `deletion`, `substitution`, or `addition`.

Classifier confidence is not pronunciation correctness. It must not be displayed as a pronunciation score.

The `score` field is reserved for a future learned pronunciation score. The
current CNN has only addition/deletion/substitution classes, so public output
uses `score: null` and `score_type: "unavailable"`. MFA supplies timing only;
heuristic/GOP/CaGOP values and classifier confidence must not be published as
pronunciation scores.

## Completed Result Example

```json
{
  "status": "completed",
  "score": null,
  "score_type": "unavailable",
  "score_note": "A learned pronunciation score is not available for the current model.",
  "pronunciation_score_source": null,
  "problem_phonemes": ["/t/"],
  "predicted_error_type": "deletion",
  "diagnosis": {
    "primary_error_type": "deletion",
    "class_probabilities": {
      "addition": 0.12,
      "deletion": 0.73,
      "substitution": 0.15
    },
    "diagnosis_confidence": 0.73,
    "confidence_note": "Classifier confidence, not pronunciation correctness."
  },
  "feedback": {
    "summary": "He thong phat hien kha nang co loi bo am trong phat am.",
    "tips": [
      "Luyen phat am ro am bi bo hoac am cuoi cua tu.",
      "Doc cham hon va chu y ket thuc am.",
      "Nghe lai phat am mau roi thu am lai."
    ]
  },
  "scorer": {
    "name": "cnn_attention_phone_error_classifier",
    "type": "phone_error_classifier",
    "version": "demo-contract-v1"
  },
  "metadata": {
    "model_output_is_scoring": false,
    "alignment_used": false,
    "gop_used": false,
    "hybrid_used": false,
    "score_type": "unavailable"
  }
}
```

Aligned inference can include internal scoring diagnostics, but
`scoring_method=heuristic_gop` is not a public pronunciation score. GOP/CaGOP
is not the Phoenix v2 roadmap.

Hybrid aligned inference can add `diagnosis.top_issues`, `diagnosis.severity`, `metadata.hybrid_method`, and `metadata.location_reliability`. These fields are advisory and must preserve the distinction between diagnosis confidence, pronunciation score, and location reliability.

## Failed Result Example

```json
{
  "status": "failed",
  "score": null,
  "problem_phonemes": [],
  "predicted_error_type": null,
  "diagnosis": {
    "primary_error_type": null,
    "class_probabilities": {
      "addition": 0.0,
      "deletion": 0.0,
      "substitution": 0.0
    },
    "diagnosis_confidence": null,
    "confidence_note": "Classifier confidence, not pronunciation correctness."
  },
  "feedback": {
    "summary": "AI worker khong tao duoc ket qua chan doan phat am.",
    "tips": [
      "Hay thu thu am lai trong moi truong yen tinh hon.",
      "Neu loi tiep tuc xay ra, vui long thu lai sau."
    ]
  },
  "scorer": {
    "name": "cnn_attention_phone_error_classifier",
    "type": "phone_error_classifier",
    "version": "demo-contract-v1"
  },
  "metadata": {
    "model_output_is_scoring": false,
    "alignment_used": false,
    "gop_used": false,
    "hybrid_used": false,
    "error": "demo failure"
  }
}
```

## Future Extensions

Forced alignment should populate `problem_phonemes` from aligned phone spans and set `metadata.alignment_used = true`.

The selected future path is a learned correctness head and, only after quality
labels exist, a learned quality/scoring head. The current `heuristic_gop`
scaffold stays internal and keeps `metadata.gop_used = false`.

Hybrid diagnosis combines alignment and classifier diagnosis into clearer issue
selection and feedback. It may retain heuristic diagnostics internally, but
must not publish them as a score or claim GOP/CaGOP.

## Backend Webhook Compatibility

The current FastAPI webhook accepts:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The full normalized result can be stored inside `feedback` or posted after the backend schema is widened. Until then, the worker should preserve the existing webhook fields and avoid changing frontend assumptions.

Warning: `diagnosis.diagnosis_confidence` and class probabilities must not be displayed as pronunciation score.
