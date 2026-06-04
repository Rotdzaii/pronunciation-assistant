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
- Complete this step before implementing real GOP/CaGOP, because GOP/CaGOP needs reliable phone boundaries.

## 2. Phase 4B: GOP/CaGOP

- Start only after real forced-alignment validation is available.
- Implement real GOP or CaGOP-style scoring based on acoustic likelihoods or posterior probabilities.
- Calibrate phone-level scores so they are interpretable and not just classifier confidence.
- Replace or clearly separate the current `heuristic_gop` value.
- Validate whether GOP/CaGOP scores correlate with actual pronunciation quality labels.

## 3. Phase 4C: Dataset Expansion

- Search for additional public pronunciation or mispronunciation datasets suitable for English L2 learners.
- Normalize speaker metadata, prompt text, phone labels, and error labels before combining datasets.
- Avoid directly mixing incompatible label definitions without mapping rules.
- Prioritize Vietnamese or similar L1 learner coverage if available.

## 4. Phase 4D: Stronger Model Experiments

- Test Wav2Vec2, HuBERT, Whisper encoder features, or fine-tuning.
- Compare stronger acoustic representations against the current CNN Attention context candidate.
- Keep speaker-independent evaluation as the primary selection protocol.
- Track macro F1, addition F1, deletion F1, substitution F1, calibration, and runtime.

## 5. Evaluation Plan

- Expand speaker-disjoint evaluation beyond the current four Vietnamese speakers when data permits.
- Add confidence calibration metrics.
- Separate pronunciation correctness scoring from error-type classification.
- Include real user or teacher-reviewed evaluation if feasible.

## 6. Phase 4E: Runtime And Audio Preprocessing Optimization

- Benchmark local audio files separately from signed URL downloads.
- Optimize WebM decoding and conversion from frontend recordings.
- Reduce audio preparation time before considering GPU deployment.
- Treat GPU as useful only if model inference, not preprocessing, becomes the bottleneck.
