# R5-4A INSERT Candidate Recoverability Audit Contract

## Status and purpose

R5-4A is a feasibility audit of already-frozen TRAIN evidence. It asks whether the exact true Addition phone/boundary candidate is ranked near the top of the unchanged exact single-INSERT CTC candidate family. It does not authorize a new scorer, classifier, decision threshold, checkpoint inference, VALIDATION access, or TEST access.

This contract was frozen before inspection of any complete INSERT candidate-ranking values.

## Frozen population

The primary population is every exactly mappable frozen TRAIN single-addition word. Multiple-addition words and their individual events are analyzed separately and do not enter the four primary feasibility gates. No difficult word, phone, speaker, or position may be excluded.

## Frozen candidate identity and ranking

For expected sequence length `N`, the candidate family contains exactly `40 * (N + 1)` candidates. Candidate identity is `(inserted canonical phone, expected-sequence boundary)` using the frozen 40-phone vocabulary.

If complete authoritative frozen candidate scores exist, finite candidates are ranked within each word by:

1. higher frozen INSERT TARGET score;
2. the existing frozen tie order: lower insertion boundary, then lower canonical-phone index.

Rank 1 is `BEST_INSERT`. KEEP, SUB, and DELETE do not enter this audit.

## Frozen recoverability metrics

For exactly mappable single-addition words, report Top-1, Top-3, Top-5, and Top-10 exact-event recall; mean, median, Q25, Q75, Q90, and maximum truth rank; and mean reciprocal rank. Report incremental gain Top-1 to Top-3, Top-3 to Top-5, and Top-5 to Top-10.

For each word, preserve truth score, BEST_INSERT score, truth rank, and `BEST_INSERT_SCORE - TRUTH_INSERT_SCORE`. Summarize the gap without deriving a new decision rule. Report descriptive speaker, added-phone, and position diagnostics. Analyze multiple-addition words separately.

## Frozen feasibility gates

All four gates must pass:

1. G1: exact truth-candidate mapping coverage for single-addition words `>= 0.99`.
2. G2: Top-5 exact-event recoverability `>= 0.40`.
3. G3: Top-10 exact-event recoverability `>= 0.55`.
4. G4: median exact truth-event rank `<= 10`.

If all pass, status is `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_FEASIBLE`. If a completed audit fails any gate, status is `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_NOT_CONFIRMED`.

## Provenance stop rule

Before any recoverability calculation, frozen artifacts must provide, for every eligible word, the complete INSERT candidate identities and corresponding authoritative frozen scores, stable source identity, and frozen truth event identity. `BEST_INSERT` alone is insufficient.

If complete frozen INSERT rankings are unavailable, stop without checkpoint inference or score reconstruction using status `R5_4A_BLOCKED_CANDIDATE_SCORE_PROVENANCE`. Do not compute recoverability metrics or evaluate the four gates.

## Protocol

- Neural training: no.
- Checkpoint inference: no.
- New acoustic scores: no.
- Classifier fitting: no.
- Threshold search: no.
- TRAIN audio access: no.
- VALIDATION access: no.
- TEST access: no.
- Modification or rerun of R5-1A, R5-2B, or R5-3A: no.
