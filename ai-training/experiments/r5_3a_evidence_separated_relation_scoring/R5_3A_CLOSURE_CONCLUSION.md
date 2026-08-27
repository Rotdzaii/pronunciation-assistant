# R5-3A Evidence-Separation Generation Closure

## Frozen result

R5-3A evaluated one preregistered evidence-separated linear fusion on the exact 16,582-word TRAIN population using 12-fold speaker LOSO. All 12 folds converged in seven iterations. The final development result was:

`R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED`

Six of eight frozen gates passed. G1, G2, G3, G4, G5, and G8 passed; G6 and G7 failed. No robust threshold was authorized.

## Scientific interpretation

Relative to frozen R5-1A, R5-3A improved Addition/all-negative AUC, Addition/correct-only AUC, Binary Macro-F1, Addition F1, correct-only FAR, and exact-event F1. It also reversed much of the overall discrimination damage observed under R5-2B hard-max relation competition.

The intended relation-confounder behavior was not retained. Substitution-negative FAR increased from 0.05016116035455278 to 0.05257856567284448, and deletion-negative FAR increased from 0.03349964362081254 to 0.09907341411261582. Standardized coefficient medians were positive for A, S, and D, with positive signs in all 12 folds. This is consistent with failure to retain relation-specific suppression; it does not establish that substitution or deletion evidence causally creates Addition.

The defensible conclusion is that fixed three-feature linear fusion recovered useful aggregate Addition discrimination but failed the preregistered substitution- and deletion-negative FAR requirements. R5-3A is not globally superior, confirmed, production-ready, VALIDATION-confirmed, or TEST-confirmed.

## Closure

Final closure status:

`R5_3A_EVIDENCE_SEPARATION_GENERATION_CLOSED_NOT_CONFIRMED`

R5-3A must not be rerun, retuned, or modified. VALIDATION was not accessed by R5-3A. TEST remains untouched. Any further Addition research requires a separately named mechanistic hypothesis and preregistration before performance is observed.
