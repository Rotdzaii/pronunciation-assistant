# AI Phase 4 Research Plan

## Purpose

Phase 4 should move the project from a validated research candidate toward a more complete pronunciation assessment model. The current `CNN Attention with context_0_10` candidate is integrated and demo-ready, but it still depends on approximate alignment, heuristic scoring, limited Vietnamese speaker coverage, and broad error-type classification.

## 1. Phase 4A: Real Forced Alignment

- Treat real forced alignment as the first recommended technical step in Phase 4.
- Evaluate MFA or an equivalent forced-alignment tool.
- Produce reliable phone-level timing from prompt text and learner audio.
- Parse TextGrid or equivalent alignment output into the project schema.
- Compare fallback alignment against forced-alignment timing to quantify timing error.
- Keep fallback alignment as a clearly labeled degraded/demo path, not as phone-level evidence.
- Complete this step before training learned phone-level heads, because the
  CNN + Attention + Context pipeline needs reliable phone boundaries to select
  the intended acoustic segment.

## 2. Phase 4B: Correct Samples And A Learned Correctness Head

- Add audited `correct` phone examples to the current addition/deletion/
  substitution dataset without leakage between speaker or prompt splits.
- Train a correctness head in the CNN + Attention + Context model family.
- Keep diagnosis confidence separate from correctness probability.
- Do not publish any numerical pronunciation score until a learned quality
  head has appropriate supervised labels.

## 3. Phase 4C: Learned Quality/Scoring Research

- Define and collect phone- or word-level human quality labels with an
  explicit scoring rubric.
- Train a regression or ordinal quality head only after establishing a
  leakage-safe evaluation protocol.
- Before then, public output remains `score: null` and
  `score_type: "unavailable"`.

## 4. Phase 4D: Dataset Expansion

- Search for additional public pronunciation or mispronunciation datasets suitable for English L2 learners.
- Normalize speaker metadata, prompt text, phone labels, and error labels before combining datasets.
- Avoid directly mixing incompatible label definitions without mapping rules.
- Prioritize Vietnamese or similar L1 learner coverage if available.

## 5. Phase 4E: Stronger Model Experiments

- Test Wav2Vec2, HuBERT, Whisper encoder features, or fine-tuning.
- Compare stronger acoustic representations against the current CNN Attention context candidate.
- Keep speaker-independent evaluation as the primary selection protocol.
- Track macro F1, addition F1, deletion F1, substitution F1, calibration, and runtime.

## 6. Evaluation Plan

- Expand speaker-disjoint evaluation beyond the current four Vietnamese speakers when data permits.
- Evaluate correctness and quality heads against their own supervised labels;
  do not convert classifier confidence into a pronunciation score.
- Separate pronunciation correctness scoring from error-type classification.
- Include real user or teacher-reviewed evaluation if feasible.

## 7. Phase 4F: Runtime And Audio Preprocessing Optimization

- Benchmark local audio files separately from signed URL downloads.
- Optimize WebM decoding and conversion from frontend recordings.
- Reduce audio preparation time before considering GPU deployment.
- Treat GPU as useful only if model inference, not preprocessing, becomes the bottleneck.
