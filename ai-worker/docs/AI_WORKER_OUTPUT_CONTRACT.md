# AI Worker Output Contract

## Purpose

The AI Worker returns a normalized result object before posting to the backend webhook. This contract separates phone error diagnosis from pronunciation scoring so CNN Attention, forced alignment, GOP/CaGOP, and hybrid scoring can be integrated without changing the app-facing shape each time.

For the backend/frontend-ready final shape and examples, see `ai-worker/docs/FINAL_AI_OUTPUT_CONTRACT.md`.

Current selected model candidate: CNN Attention phone error classifier.

Selected model metrics:

- mean test macro F1: 0.5124 +/- 0.0214
- mean test addition F1: 0.1938 +/- 0.0415

## Confidence Is Not Score

The CNN Attention classifier predicts an error type and class probabilities. Its confidence is the probability or confidence of a diagnostic class such as `deletion`, `substitution`, or `addition`.

Classifier confidence is not pronunciation correctness. It must not be displayed as a pronunciation score.

The `score` field is reserved for pronunciation scoring. Until forced alignment, GOP/CaGOP, or hybrid scoring is available, any demo score must be clearly marked as heuristic/demo metadata.

## Completed Result Example

```json
{
  "status": "completed",
  "score": 72.0,
  "score_note": "Demo heuristic only. This is not real pronunciation scoring and must not be derived from classifier confidence in production.",
  "pronunciation_score_source": "demo_error_type_heuristic",
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
    "is_demo_score": true,
    "score_note": "Demo heuristic only. This is not real pronunciation scoring and must not be derived from classifier confidence in production."
  }
}
```

Aligned inference can also include a `scoring` block. In the current scaffold, `scoring_method=heuristic_gop` is a demo placeholder and not production GOP.

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

GOP/CaGOP should provide pronunciation scoring evidence and set `metadata.gop_used = true` only when real acoustic GOP/CaGOP is used. The current `heuristic_gop` scaffold keeps `metadata.gop_used = false` and marks `metadata.scoring_is_heuristic = true`.

Hybrid diagnosis combines alignment, scoring, and classifier diagnosis into clearer issue selection and feedback. With heuristic scoring, it may set `metadata.hybrid_used = true`, but it must still mark `metadata.scoring_is_heuristic = true` and avoid claiming real GOP/CaGOP.

## Backend Webhook Compatibility

The current FastAPI webhook accepts:

- `job_id`
- `status`
- `score`
- `problem_phonemes`
- `feedback`

The full normalized result can be stored inside `feedback` or posted after the backend schema is widened. Until then, the worker should preserve the existing webhook fields and avoid changing frontend assumptions.

Warning: `diagnosis.diagnosis_confidence` and class probabilities must not be displayed as pronunciation score.
