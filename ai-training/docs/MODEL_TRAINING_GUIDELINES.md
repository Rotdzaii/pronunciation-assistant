# Model Training Guidelines

## X. Pronunciation Modeling Boundaries

These guidelines follow the Phoenix v2 decision in
[ADR: Deep Learning First Pronunciation Scoring](ADR_DEEP_LEARNING_FIRST_PRONUNCIATION_SCORING.md).

### X.5 Historical GOP/CaGOP Note

GOP/CaGOP is retained in repository documentation only as a historical or
hypothetical alternative for acoustic-likelihood research. It is not the
selected Phoenix v2 roadmap and must not be implemented as a replacement for
the Deep Learning First approach without a new architecture decision.

Do not derive a public pronunciation score from heuristic GOP, GOP/CaGOP, or
classifier confidence. Before a learned quality head exists and is validated,
publish `score: null` and `score_type: "unavailable"`.
