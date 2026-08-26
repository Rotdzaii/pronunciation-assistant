# Historical GOP/CaGOP Proposal (Not Current Roadmap)

## Purpose

This document is retained as a historical/hypothetical GOP/CaGOP proposal. It
is not the selected Phoenix v2 roadmap. The accepted direction is recorded in
[Deep Learning First Pronunciation Scoring](../../ai-training/docs/ADR_DEEP_LEARNING_FIRST_PRONUNCIATION_SCORING.md).

GOP/CaGOP was considered and rejected for Phoenix v2. It must not replace the
CNN + Attention + Context research path or provide a public score.

CNN Attention predicts an error type such as deletion, substitution, or addition. Its confidence is diagnosis confidence, not a pronunciation score.

## Historical Scaffold

This feature adds a scaffold only:

- normalized scoring contract
- `SCORING_MODE=heuristic_gop|none`
- heuristic GOP-like scoring service
- aligned CNN Attention integration

The historical method named `heuristic_gop` is not production GOP and does not
use acoustic posterior probabilities or phone likelihoods. It may remain as
internal diagnostic information, but its public score is unavailable.

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

## Historical GOP/CaGOP Requirements

A production GOP or CaGOP implementation would require:

- reliable phone alignment, preferably MFA phone boundaries
- acoustic posterior or phone likelihood model
- phone likelihood normalization against competing phones
- calibration from raw likelihood values to user-facing scores
- validation on held-out pronunciation data

No acoustic model or external GOP dependency is added by this scaffold.

## Rejected Fusion Direction

CNN Attention continues to provide error-type diagnosis. The selected future
direction is a learned correctness head followed, only when supervised quality
labels exist, by a learned quality head. GOP/CaGOP is not the selected fusion
component.

- aligned CNN Attention error type
- learned correctness and quality outputs when trained and validated
- alignment confidence and boundary source
- duration and consistency signals

The fused output should keep diagnosis confidence and pronunciation score separate.

## Current Limitations

- heuristic scores are demo placeholders
- fallback alignment limits reliability
- no acoustic posterior scoring exists yet
- no raw GOP likelihood is computed
- no CaGOP calibration is implemented

## Superseded Replacement Plan

Do not replace `ai-worker/app/scoring/heuristic_gop_scorer.py` with GOP/CaGOP
as a Phoenix v2 roadmap item. Until a learned quality scorer is trained and
validated, public output must use `score: null` and
`score_type: "unavailable"`.
