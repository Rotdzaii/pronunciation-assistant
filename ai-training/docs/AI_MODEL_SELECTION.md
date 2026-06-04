# AI Model Selection

## Current Decision

The Phase 2 leading research candidate is:

`CNN Attention with context_0_10, validated with Vietnamese speaker-disjoint multi-seed evaluation`

This candidate is not yet a final production model. It is the best current research model for the phone-level error classification task because it has the strongest evidence under unseen Vietnamese speaker evaluation.

## Previous Candidate

Previous candidate:

`Vietnamese-only CNN Attention`

Key result:

| Model | Evaluation | Macro F1 | Addition F1 |
|---|---|---:|---:|
| Vietnamese-only CNN Attention | Vietnamese-only test split | 0.5105 | 0.1754 |

This model remained important because it was trained and evaluated directly on Vietnamese L2-ARCTIC speakers. However, the split was not the final Phase 2 generalization test.

## Phase 2 Candidates

| Candidate | Evaluation | Macro F1 | Addition F1 | Decision |
|---|---|---:|---:|---|
| Vietnamese-only CNN Attention | Vietnamese-only test split | 0.5105 | 0.1754 | Previous candidate. |
| All-speaker CNN Attention | Non-disjoint Vietnamese subset | 0.5420 | 0.2769 | Not selected alone; speaker overlap may be optimistic. |
| Speaker-disjoint baseline | Vietnamese leave-one-speaker-out | 0.5022 +/- 0.0210 | 0.0881 +/- 0.0391 | Baseline for stricter evaluation. |
| Addition-focused sampler | Vietnamese leave-one-speaker-out | 0.4715 +/- 0.0299 | 0.0958 +/- 0.0348 | Not selected. |
| Context `0.10s` CNN Attention | Vietnamese leave-one-speaker-out, single seed | 0.5178 +/- 0.0252 | 0.1246 +/- 0.0271 | Promising candidate. |
| Context `0.10s` CNN Attention | Vietnamese leave-one-speaker-out, 3 seeds | 0.5170 +/- 0.0338 | 0.1251 +/- 0.0473 | Selected Phase 2 research candidate. |

## Why Not All-Speaker Non-Disjoint Alone

The all-speaker CNN Attention model improved Vietnamese subset metrics:

- Macro F1: `0.5420`
- Addition F1: `0.2769`

Those results are useful, but they are not enough for final selection because the evaluation is not speaker-disjoint. Speaker overlap can make the model look better than it will be on unseen Vietnamese learners.

## Why Not Addition-Focused Sampler

The addition-focused sampler only improved addition F1 slightly:

- Speaker-disjoint baseline addition F1: `0.0881`
- Addition-focused sampler addition F1: `0.0958`

But it reduced macro F1:

- Speaker-disjoint baseline macro F1: `0.5022`
- Addition-focused sampler macro F1: `0.4715`

This tradeoff is not acceptable because the classifier must still handle deletion and substitution reliably.

## Why Context_0_10 Is The Best Current Candidate

`context_0_10` uses a 0.10 second audio window around the annotated phone-error segment. It keeps the same CNN Attention architecture and the same label schema while giving the model local acoustic context.

The multi-seed stability result confirms that the single-seed improvement was not just one lucky run:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.6618 | 0.0324 |
| Macro F1 | 0.5170 | 0.0338 |
| Addition F1 | 0.1251 | 0.0473 |
| Deletion F1 | 0.6819 | 0.0382 |
| Substitution F1 | 0.7439 | 0.0316 |

Compared with the speaker-disjoint baseline, it improves macro F1 and addition F1 while keeping deletion and substitution in a usable range.

## Cautions

- This is still a research candidate, not a final production pronunciation model.
- Only four Vietnamese speakers are available for speaker-disjoint evaluation.
- Addition remains difficult and high variance.
- The model operates on known phone-error segments.
- Confidence is classifier confidence, not pronunciation correctness.
- Checkpoints are local artifacts and are not committed.

## Phase 4 Direction

Phase 2 selected `context_0_10` CNN Attention as the current candidate, and the candidate has since been integrated into the AI Worker demo flow.

Phase 4 should focus on real forced alignment, real GOP/CaGOP, larger and better-normalized datasets, stronger acoustic models or fine-tuning, calibration, and runtime/audio preprocessing optimization.
