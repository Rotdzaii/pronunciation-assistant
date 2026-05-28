# Final AI Output Contract

## Purpose

This document defines the stable AI Worker output shape for backend and frontend integration. The contract separates pronunciation score, diagnostic confidence, alignment metadata, scoring metadata, and user-facing feedback.

## Completed Result Structure

Core fields:

- `status`: `completed`
- `score`: pronunciation score or demo/heuristic score
- `score_note`: explanation of score source and limitations
- `pronunciation_score_source`: source such as `heuristic_gop`
- `problem_phonemes`: list of affected phones
- `predicted_error_type`: primary diagnosis label
- `diagnosis`: classifier and hybrid diagnosis details
- `feedback`: user-facing summary and tips
- `scorer`: model/scorer metadata
- `metadata`: alignment, scoring, and hybrid flags
- `scoring`: optional segmental scoring contract

Example:

```json
{
  "status": "completed",
  "score": 61.2,
  "score_note": "Heuristic/demo score, not production GOP.",
  "pronunciation_score_source": "heuristic_gop",
  "problem_phonemes": ["EH"],
  "predicted_error_type": "deletion",
  "diagnosis": {
    "primary_error_type": "deletion",
    "class_probabilities": {
      "addition": 0.05,
      "deletion": 0.84,
      "substitution": 0.11
    },
    "diagnosis_confidence": 0.84,
    "confidence_note": "Classifier confidence, not pronunciation correctness.",
    "severity": "high",
    "top_issues": [
      {
        "phone": "EH",
        "word": "example",
        "predicted_error_type": "deletion",
        "diagnosis_confidence": 0.84,
        "phone_score": 55.1,
        "severity": "high"
      }
    ]
  },
  "feedback": {
    "summary": "He thong phat hien kha nang co loi bo am tai EH.",
    "tips": ["Luyen lai am hoac tu duoc danh dau voi toc do cham."]
  },
  "scorer": {
    "name": "cnn_attention",
    "type": "phone_error_classifier",
    "version": "cnn_attention_selected_baseline"
  },
  "metadata": {
    "model_output_is_scoring": false,
    "alignment_used": true,
    "alignment_method": "fallback_even_split",
    "gop_used": false,
    "hybrid_used": true,
    "scoring_is_heuristic": true,
    "location_reliability": "limited_fallback_alignment"
  }
}
```

## Failed Result Structure

Example:

```json
{
  "status": "failed",
  "score": null,
  "score_note": null,
  "pronunciation_score_source": null,
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
    "tips": ["Hay thu thu am lai trong moi truong yen tinh hon."]
  },
  "scorer": {
    "name": "cnn_attention",
    "type": "phone_error_classifier"
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

## Field Meanings

`score` is the pronunciation score field. In the current scaffold it can come from `heuristic_gop`, which is not production GOP.

`diagnosis_confidence` is classifier confidence for diagnosis. It is not pronunciation correctness and must not be displayed as the pronunciation score.

`problem_phonemes` is the list of phones selected for feedback.

`diagnosis.top_issues` contains localized hybrid issue candidates.

`feedback.summary` and `feedback.tips` are safe app-facing feedback fields.

`metadata.alignment_used` indicates whether segment alignment was used.

`metadata.gop_used` should be true only for real acoustic GOP/CaGOP. With `heuristic_gop`, it remains false.

`metadata.hybrid_used` indicates that the hybrid issue-selection layer was applied.

`metadata.scoring_is_heuristic` marks demo scoring that must not be presented as real GOP.

## Current Status

- CNN Attention is the selected real classifier.
- Fallback alignment is approximate and not real forced alignment.
- MFA execution is scaffolded and requires local configuration.
- `heuristic_gop` is a scaffold and not real GOP/CaGOP.
- Hybrid diagnosis is a logic layer that combines available signals.

## Backend And Frontend Notes

The frontend can display:

- `score`
- `score_note`
- `feedback.summary`
- `feedback.tips`
- `problem_phonemes`
- `diagnosis.top_issues`

The frontend must not display `diagnosis_confidence` as pronunciation score.

The backend should persist raw `metadata`, `diagnosis`, `scoring`, and `segments` when available for debugging and future calibration.

Run `ai-worker/scripts/demo_final_ai_output.py` to generate and validate representative completed and failed outputs.
