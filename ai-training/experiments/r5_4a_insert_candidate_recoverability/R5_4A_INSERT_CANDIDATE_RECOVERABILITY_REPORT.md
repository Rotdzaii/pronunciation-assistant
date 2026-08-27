# R5-4A INSERT Candidate Recoverability Audit

## Result

`R5_4A_BLOCKED_CANDIDATE_SCORE_PROVENANCE`

All frozen source identities passed. The R5-4A audit contract, including Top-k values and four feasibility gates, was frozen before candidate-score provenance inspection.

## Candidate-score provenance

The authoritative R5-1A and R5-2B TRAIN score artifacts contain stable source identity, expected sequence, frozen truth events, INSERT candidate counts, and the identity and score of `BEST_INSERT`. They do not contain the identities or scores of non-winning INSERT candidates. Their frozen execution drivers confirm that complete candidate families were scored transiently in memory and reduced to family winners during serialization.

R5-2B's candidate audit contains aggregate counts only. R5-3A contains word-level `[A,S,D]` features and OOF classifier outputs, not INSERT rankings. A narrow inventory of the authoritative R5-1A, R5-2B, and R5-3A experiment directories found no separate persisted candidate-ranking, posterior, logit, or per-candidate score artifact.

Therefore the exact truth-candidate rank, Top-1/3/5/10 recoverability, score gaps, speaker/phone/position breakdowns, and multiple-addition ranks cannot be calculated from frozen artifacts. `BEST_INSERT` alone cannot reconstruct the ordering below rank 1.

## Required stop

No truth mapping or feasibility metric was computed, and none of G1-G4 was evaluated. This is a provenance blockage, not a failed recoverability result. No checkpoint was loaded, no inference or acoustic scoring was run, and no audio, VALIDATION, or TEST data was accessed.

## Interpretation questions

1. Whether the true event is usually near the top: not answerable from the persisted frozen artifacts.
2. Whether Top-5/Top-10 substantially improve over Top-1: not answerable.
3. Whether poor exact-event F1 is dominated by top-1 selection or low family recoverability: not distinguishable.
4. Whether any failure is broad or concentrated: not answerable without ranks.
5. Whether top-k/marginalized/local-boundary evidence is justified: not established by R5-4A; complete candidate rankings would first need to be materialized under a separately preregistered TRAIN-only stage.
