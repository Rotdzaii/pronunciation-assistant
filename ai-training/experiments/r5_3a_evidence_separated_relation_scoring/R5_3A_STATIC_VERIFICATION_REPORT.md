# R5-3A Synthetic Static Verification

## Result

`R5_3A_STATIC_VERIFICATION_PASS`

All 27 frozen static criteria passed. Two complete runs each executed 31 synthetic tests with zero failures and zero errors. Their deterministic summary artifacts were byte-identical with SHA-256 `DC5E43D8C7C692A0F748012772EDA157EDAAD082885651460C0D8DDD6351954C`.

## Identities

The R5-3A contract, preregistration, and 9/9 contract-manifest entries matched. R5-1A 14/14, R5-2B 21/21, and R5-2C 15/15 upstream manifested artifacts matched. V4 and checkpoint hashes matched; the checkpoint was not loaded.

## Feature semantics

The implementation constructs exactly three finite float64 columns in deterministic order `[A,S,D]`:

- A uses frozen R5-1A `addition_score_A_value`.
- S is frozen R5-2B BEST_SUB minus KEEP.
- D is frozen R5-2B BEST_DELETE minus KEEP.

Asymmetric fixtures detected swaps. Four-column, forbidden-feature, nonfinite, duplicate, missing, and identity-mismatch fixtures were rejected.

## Classifier and probability output

StandardScaler parameters matched `copy=True`, `with_mean=True`, and `with_std=True`. LogisticRegression matched the frozen L2, C=1.0, lbfgs, balanced, max_iter=1000 configuration. The continuous helper calls `predict_proba` and resolves the column whose class identity is exactly 1; `predict` and `decision_function` were not used.

scikit-learn 1.8.0 emits a future deprecation warning for the explicitly frozen `penalty="l2"` argument. The static implementation intentionally preserves the preregistered parameter instead of changing the contract.

## Leakage and LOSO

A four-speaker, 24-row synthetic fixture verified calibration-only scaler fitting, model fitting, probability calibration, and threshold selection. Extreme held-out values did not affect scaler mean, variance, or scale. Mutating held-out labels did not change the fold model or threshold. The same calibration scaler transformed held-out rows. Each synthetic row appeared in exactly one deterministic held-out output.

## Threshold and events

Candidate construction included exactly finite unique scores and the two np.nextafter edges. Equality used `score >= theta`. All four tie priorities passed independently. Predicted-positive events preserved the input BEST_INSERT phone and boundary exactly; predicted-negative rows emitted no Addition event.

## Real-artifact provenance

Only the first JSONL row from each frozen TRAIN score artifact was read to verify required field names and types. Frozen population metadata remained 16,582 words, 323 positive, and 16,259 negative. No real feature matrix was built, and no real classifier, threshold, or performance metric was calculated.

## Protocol

No checkpoint inference, audio access, real TRAIN fitting, real TRAIN threshold selection, performance calculation, VALIDATION access, or TEST access occurred. R5-1A, R5-2B, R5-2C, and the frozen R5-3A contract were not modified.
