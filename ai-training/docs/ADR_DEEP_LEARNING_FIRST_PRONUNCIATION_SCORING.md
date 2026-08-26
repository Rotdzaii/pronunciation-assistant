# ADR: Deep Learning First Pronunciation Scoring

## Status

Accepted for Phoenix v2.

## Decision

Phoenix v2 uses Deep Learning First for pronunciation modeling. The roadmap
continues the CNN + Attention + Context model family, first with audited
correct-phone samples and a learned correctness head, then with a learned
quality/scoring head only when suitable supervised quality labels exist.

GOP/CaGOP was considered and rejected as the selected Phoenix v2 scoring
roadmap. Heuristic rules, GOP/CaGOP calculations, and classifier confidence
must not be represented as a public pronunciation score.

Until a learned quality scorer has been trained and validated, public results
must set `score: null` and `score_type: "unavailable"`.

## Rationale

The current CNN Attention Context checkpoint was trained with three error
classes: `addition`, `deletion`, and `substitution`. It has no `correct` class,
no correctness head, and no supervised 0-100, ordinal, or regression quality
label. A classifier confidence is evidence for an error-type prediction, not
pronunciation correctness or quality.

Using a numerical score derived from heuristics, duration, GOP/CaGOP, or class
confidence would create a score without the required supervision and validation.

## MFA Role

MFA is the forced-alignment component. It normalizes known transcript audio
into word and phone timing so the model can select and localize segments. MFA
quality determines timing reliability only; it does not determine pronunciation
correctness or a public score.

## Safety Rules Role

Rule-based checks remain limited to input validation, audio quality gates,
alignment integrity, runtime timeouts, reliability states, output validation,
and safe failure handling. They may prevent an unreliable diagnosis from being
published, but may not replace the learned model with a numerical score.

## Conditions For A Deep Learning 0-100 Score

A public learned score requires all of the following:

- Audited correct and error samples with a leakage-safe speaker and prompt
  split.
- Phone- or word-level quality labels with an explicit human scoring rubric.
- A learned quality head trained against those labels, such as ordinal or
  regression supervision.
- Held-out evaluation demonstrating calibration, reliability, and useful
  correlation with the target quality labels.
- A contract that keeps quality score, correctness probability, diagnosis
  confidence, and alignment reliability separate.

## Consequences

- The current system may publish reliable addition/deletion/substitution
  diagnosis and feedback.
- The current system may not publish a pronunciation score.
- Future checkpoints can preserve the CNN + Attention + Context backbone while
  adding separate correctness, error-type, and learned quality heads.
