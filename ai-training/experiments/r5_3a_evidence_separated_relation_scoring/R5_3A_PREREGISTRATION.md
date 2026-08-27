# R5-3A Evidence-Separated Relation Scoring Preregistration

## Stage

R5-3A is a new TRAIN-only development generation. R5-2B remains frozen as `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED` and will not be modified or rerun.

This document freezes design only. It authorizes no classifier fitting, performance calculation, VALIDATION access, or TEST access.

## Hypothesis

Keeping Addition, Substitution, and Deletion evidence as separate features will preserve useful Addition evidence while allowing a simple linear classifier to use substitution/deletion evidence as confounder information.

R5-2C showed that hard-max competition reduced substitution/deletion false additions but also suppressed genuine additions and caused compensating LOSO thresholds. R5-3A therefore changes only evidence fusion. It does not change the checkpoint, hypothesis scores, population, or BEST_INSERT localization.

## Frozen population and sources

Use only the exact stable-identity join of existing frozen TRAIN score artifacts:

- R5-1A: `r5_1a_train_scores.jsonl`
- R5-2B: `r5_2b_tc1_pa1_env_train_scores.jsonl`

Required accounting:

- 16,582 matched words
- 323 Addition-positive
- 16,259 negative
- zero missing or duplicate identities
- no fuzzy matching or new exclusions

TRAIN speakers, in order: BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA.

## Frozen feature vector

Feature order is exactly `[A, S, D]`:

1. `A = BEST_INSERT_SCORE - KEEP_SCORE`, using the persisted frozen R5-1A `addition_score_A_value`.
2. `S = BEST_SUB_SCORE - KEEP_SCORE`, using frozen R5-2B source values.
3. `D = BEST_DELETE_SCORE - KEEP_SCORE`, using frozen R5-2B source values.

Parse source values as finite float64 values. Compute S and D by float64 subtraction. Do not clip, impute, round, normalize globally, or add features.

R5-2B `C`, `max(S,D)`, pre-subtracted relation features, speaker identity, phones, words, boundaries, duration, annotations, and relation labels are forbidden classifier inputs.

## Frozen classifier

For each fold:

```python
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
)
```

`random_state` is not an experimental degree of freedom because it is not applicable to the selected deterministic lbfgs solver.

The authoritative continuous score is frozen as:

```python
model.predict_proba(X_scaled)[:, 1]
```

No alternate classifier, hyperparameter search, polynomial feature, or decision-function output is authorized.

## Exact speaker LOSO

Run exactly 12 folds. In each fold:

1. Hold out one TRAIN speaker.
2. Fit StandardScaler only on the other 11 speakers.
3. Transform the 11-speaker calibration data.
4. Fit LogisticRegression only on those calibration rows and labels.
5. Calculate calibration probabilities.
6. Select a threshold from calibration probabilities only.
7. Transform the held-out speaker using the calibration scaler.
8. Apply the fitted classifier and selected threshold once.

Held-out features may be transformed but must never influence scaler statistics. Held-out labels must not influence scaler fitting, model fitting, or threshold selection.

Threshold candidates are every unique finite float64 calibration probability, `np.nextafter(min_score, -np.inf)`, and `np.nextafter(max_score, np.inf)`. The decision is `score >= theta`.

Threshold tie priority is frozen:

1. higher Binary Macro-F1
2. higher Addition F1
3. lower correct-only FAR
4. higher threshold

## Event output

For a threshold-positive word, retain the existing frozen BEST_INSERT phone and expected-sequence boundary. R5-3A changes binary evidence fusion only; it does not change event localization.

## Frozen gates

All eight must pass at full precision:

1. Addition/all-negative ROC-AUC `>= 0.70`
2. Addition/correct-only ROC-AUC `>= 0.70`
3. OOF Binary Macro-F1 `> 0.5551978767901391`
4. OOF Addition F1 `> 0.1379980563654033`
5. Correct-only FAR `<= 0.03491295938104449`
6. Substitution-negative FAR `< 0.05016116035455278`
7. Deletion-negative FAR `< 0.03349964362081254`
8. Exact-event F1 `>= 0.04389312977099236`

Only 8/8 PASS authorizes `R5_3A_ROBUST_THETA`, defined as the ordinary float64 median of the 12 fold thresholds. Otherwise record `R5_3A_ROBUST_THETA_NOT_AUTHORIZED`.

Future scientific status:

- 8/8: `R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_PASS`
- any valid gate failure: `R5_3A_EVIDENCE_SEPARATION_DEVELOPMENT_NOT_CONFIRMED`

## Required static verification

Before real fitting or performance, a separate synthetic/static stage must verify exact feature construction and ordering, row identity, calibration-only scaling/fitting, fixed classifier parameters, predict-probability output, threshold helper/ties, event metadata preservation, determinism, and absence of forbidden-feature leakage.

## Split policy

VALIDATION was consumed by prior R5 work and is prohibited here. TEST remains untouched and prohibited. No R5-3A development result automatically authorizes either split.
