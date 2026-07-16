# Final AI Output Contract

## Purpose

This document defines the stable AI Worker output shape for backend and frontend integration. The contract separates pronunciation score, diagnostic confidence, alignment metadata, scoring metadata, and user-facing feedback.

## Completed Result Structure

Core fields:

- `status`: `completed`
- `score`: nullable; unavailable until a learned quality scorer is trained and
  validated
- `score_type`: `unavailable` for the current three-class model
- `score_note`: explanation of score source and limitations
- `pronunciation_score_source`: `null` until a learned quality scorer exists
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
  "score": null,
  "score_type": "unavailable",
  "score_note": "A learned pronunciation score is not available for the current model.",
  "pronunciation_score_source": null,
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
    "score_type": "unavailable",
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

`score` is reserved for a learned pronunciation quality scorer. The current CNN
only classifies `addition`, `deletion`, and `substitution`; it has no correct
class or learned quality head. Therefore public output uses `score: null` and
`score_type: "unavailable"`.

`diagnosis_confidence` is classifier confidence for diagnosis. It is not pronunciation correctness and must not be displayed as the pronunciation score.

`problem_phonemes` is the list of phones selected for feedback.

`diagnosis.top_issues` contains localized hybrid issue candidates.

`feedback.summary` and `feedback.tips` are safe app-facing feedback fields.

`metadata.alignment_used` indicates whether segment alignment was used.

`metadata.gop_used` remains false. GOP/CaGOP was considered and rejected as
the Phoenix v2 scoring roadmap.

`metadata.hybrid_used` indicates that the hybrid issue-selection layer was applied.

Internal heuristic data, when retained for diagnostics, must not be published
as a score or presented as pronunciation correctness.

## Current Status

- CNN Attention is the selected real classifier.
- Fallback alignment is approximate and not real forced alignment.
- MFA execution is scaffolded and requires local configuration.
- `heuristic_gop` is internal diagnostic scaffolding, not a public score.
- MFA supplies forced-alignment timing only.
- The selected future path is a learned correctness head followed by a learned
  quality/scoring head when appropriate supervised labels exist.
- Hybrid diagnosis is a logic layer that combines available signals.

## Backend And Frontend Notes

The frontend can display:

- `score` only when it is not `null`; otherwise show that a score is unavailable
- `score_note`
- `feedback.summary`
- `feedback.tips`
- `problem_phonemes`
- `diagnosis.top_issues`

The frontend must not display `diagnosis_confidence` as pronunciation score.

The backend should persist raw `metadata`, `diagnosis`, `scoring`, and `segments` when available for debugging and future calibration.

Run `ai-worker/scripts/demo_final_ai_output.py` to generate and validate representative completed and failed outputs.
