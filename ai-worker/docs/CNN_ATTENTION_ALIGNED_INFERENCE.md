# CNN Attention Aligned Inference

## Purpose

Aligned inference runs the selected CNN Attention phone error classifier over approximate prompt segments instead of only the first/whole audio clip. It is designed as scaffolding for phone or word-localized diagnosis while the project waits for real forced alignment.

This feature does not train models and does not add heavy dependencies.

## Clip-Level vs Aligned Inference

Clip-level inference:

- loads the submitted audio
- pads or truncates it to the 1.0 second training length
- predicts one error type for the whole clip
- does not localize `problem_phonemes`

Aligned inference:

- receives or creates an `alignment_result`
- crops each segment by `start` and `end`
- pads or truncates every segment to the same 1.0 second training length
- predicts an error type per phone or word segment
- aggregates the most confident predicted issue into the normalized AI result

## Fallback Alignment

When prompt text is available and no external aligner result is provided, the worker can use fallback alignment:

- with `canonical_phones`, audio duration is split evenly across phones
- without `canonical_phones`, audio duration is split evenly across words
- if only a target word is present, that word becomes the single segment

Fallback alignment is approximate scaffolding only. It is not real forced alignment, does not inspect the speech signal for boundaries, and must not be described as precise.

## Segment Prediction Fields

Each segment prediction includes:

- `phone`
- `word`
- `start`
- `end`
- `predicted_error_type`
- `class_probabilities`
- `diagnosis_confidence`
- `confidence_note`

`diagnosis_confidence` is classifier confidence for the predicted diagnosis. It is not pronunciation correctness and must not be displayed as a pronunciation score.

## Aggregated Output

The normalized AI result sets:

- `predicted_error_type` from the most confident segment-level predicted issue
- `problem_phonemes` from high-confidence phone segments
- `diagnosis.primary_error_type`
- `diagnosis.class_probabilities`
- `diagnosis.diagnosis_confidence`

Metadata includes:

- `alignment_used: true`
- `alignment_status`
- `alignment_method`
- `alignment_note`
- `gop_used: false`
- `hybrid_used: false`
- `model_output_is_scoring: false`
- `segment_level_inference: true`

If fallback alignment is used, `alignment_note` clearly states that boundaries are approximate and not real forced alignment.

## Future MFA Integration

Future MFA integration should produce the same alignment contract shape with real phone or word boundaries. The CNN Attention scorer can then consume that `alignment_result` through `score_aligned_audio(...)` without changing the classifier interface.

Expected future path:

- MFA generates phone-level boundaries from prompt text and audio
- worker passes MFA `alignment_result` to CNN Attention scorer
- fallback alignment remains only for demos or degraded operation
- GOP/CaGOP or hybrid scoring supplies pronunciation correctness scores

## Current Limitations

- fallback alignment boundaries are approximate
- model classes are error-type diagnoses, not correctness scores
- demo scores remain heuristic metadata until real scoring is integrated
- phone localization is only as reliable as the provided alignment
- no MFA alignment is implemented in this feature
