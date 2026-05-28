# GOP Scoring Plan

## Purpose

Goodness of Pronunciation (GOP) is intended to provide pronunciation correctness evidence at phone, word, and utterance levels. It should be separate from CNN Attention classifier confidence.

CNN Attention predicts an error type such as deletion, substitution, or addition. Its confidence is diagnosis confidence, not a pronunciation score.

## Current Implementation

This feature adds a scaffold only:

- normalized scoring contract
- `SCORING_MODE=heuristic_gop|none`
- heuristic GOP-like scoring service
- aligned CNN Attention integration

The current method is `heuristic_gop`. It is not production GOP and does not use acoustic posterior probabilities or phone likelihoods.

The heuristic uses:

- segment-level predicted error type
- diagnosis confidence as a diagnostic signal, not as the score itself
- segment duration mismatch
- fallback alignment status

The result is marked:

- `scoring_method: heuristic_gop`
- `metadata.is_real_gop: false`
- `metadata.is_heuristic: true`
- `metadata.note: Heuristic score is not a production GOP score.`

## Contract

`ai-worker/app/contracts/scoring_contract.py` defines:

- `scoring_status`
- `scoring_method`
- `utterance_segmental_score`
- `words`
- `phones`
- `metadata`

Phone output includes:

- `phone_score`
- `gop_score_raw`
- `gop_score_calibrated`
- `duration_mismatch`
- `severity`
- `source`

## Real GOP/CaGOP Requirements

A production GOP or CaGOP implementation would require:

- reliable phone alignment, preferably MFA phone boundaries
- acoustic posterior or phone likelihood model
- phone likelihood normalization against competing phones
- calibration from raw likelihood values to user-facing scores
- validation on held-out pronunciation data

No acoustic model or external GOP dependency is added by this scaffold.

## Future Fusion With CNN Attention

CNN Attention should continue to provide error-type diagnosis. GOP/CaGOP should provide pronunciation correctness evidence. A later hybrid diagnosis layer can combine:

- aligned CNN Attention error type
- GOP/CaGOP phone scores
- alignment confidence and boundary source
- duration and consistency signals

The fused output should keep diagnosis confidence and pronunciation score separate.

## Current Limitations

- heuristic scores are demo placeholders
- fallback alignment limits reliability
- no acoustic posterior scoring exists yet
- no raw GOP likelihood is computed
- no CaGOP calibration is implemented

## Replacement Plan

Replace `ai-worker/app/scoring/heuristic_gop_scorer.py` with real GOP/CaGOP scoring behind `score_pronunciation_segments(...)`. Preserve the scoring contract so downstream AI result formatting and future hybrid diagnosis do not need a shape change.
