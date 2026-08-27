# R5-4A / R5-4B Addition Candidate Closure

R5-4B completed the technical materialization of the complete frozen TRAIN INSERT landscape. It persisted 2,977,040 candidates for 16,582 words and reproduced frozen BEST_INSERT identity and exact winning score for every word. All 10 provenance gates passed. This is a technical artifact-completeness result, not scientific Addition confirmation.

The resumed R5-4A feasibility audit mapped all 304 single-addition words exactly. Top-1, Top-3, Top-5, and Top-10 exact-event recoverability were 0.13486842105263158, 0.24013157894736842, 0.27960526315789475, and 0.3881578947368421. Median truth rank was 19.0 and mean rank was 57.10197368421053. Only the mapping gate passed; Top-5, Top-10, and median-rank gates failed.

The frozen exact single-INSERT family therefore contains partial localization information, but the correct event is not generally ranked near enough to the top to meet the preregistered criteria. BETWEEN additions were more recoverable than BEFORE_FIRST and AFTER_FINAL additions, and phone and speaker variation was substantial. These are descriptive TRAIN findings, not production rules.

Low historical exact-event F1 cannot be explained mainly by BEST_INSERT top-1 selection alone. This closure does not authorize a new generation based only on top-k, marginalization, or alternative pooling of the same candidate family.

Final statuses:

- `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_CLOSED_NOT_CONFIRMED`
- `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_CLOSED_PASS`

VALIDATION was not accessed. TEST remains untouched. The approximately 1.1 GB R5-4B candidate package remains preserved.
