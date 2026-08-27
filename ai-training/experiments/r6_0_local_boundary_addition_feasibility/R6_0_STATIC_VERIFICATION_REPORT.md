# R6-0 Local Boundary Synthetic Static Verification

Status: `R6_0_LOCAL_BOUNDARY_STATIC_VERIFICATION_PASS`

## Identity

The frozen R6-0 manifest SHA-256 `EDB1A62CD6350AFC955C7A49B51C668BD0ED4217F7BBC9AD916525769DF8718A` and all eight payload entries passed. All ten frozen upstream anchors passed. The R6-0 contract was not modified.

## Implementation

The additive implementation contains three independent helpers:

- `r6_0_boundary_mapping.py`: GOLD boundary time, one-time MFA word-start subtraction, nominal output centers, lower-index tie handling, and truncated five-step windows.
- `r6_0_local_features.py`: float32 synthetic log-softmax/posteriors, adjacent expected-phone exclusion, one primary feature, and two descriptive controls.
- `r6_0_boundary_labels.py`: exact event-to-boundary labels, event multiplicity, and the frozen coverage definition.

The implementation imports no R5 module and contains no INSERT candidate enumeration, CTC target loss, BEST_INSERT ranking, KEEP/SUB/DELETE scoring, or learned fusion.

## Synthetic verification

Two complete runs each passed 36/36 tests. Their deterministic summaries are byte-identical:

`6815B9B47BCF7C1F849DD2127DA1D1A075DEAAEC99BA8630519EA9E65EEF3E44`

Verified mapping cases include exact output centers, lower/upper nearest centers, an exact tie selecting the lower index, start/final indices, one-time absolute-to-relative conversion, seconds consistency, and invalid out-of-crop rejection.

For `T=6`, frozen edge windows were exactly:

- `k=0`: `[0,1,2]`
- `k=1`: `[0,1,2,3]`
- `k=4`: `[2,3,4,5]`
- `k=5`: `[3,4,5]`

No padding, reflection, duplication, imputation, or wraparound occurred.

Synthetic asymmetric `[3,41]` logits verified posterior row mass, blank exclusion, adjacent-phone exclusion, ordinary frame averaging, peak identity/ties, nonblank equivalence, and isolation from frames outside the selected window.

Boundary fixtures verified correct words, single Addition, multiple events at distinct and shared boundaries, negative boundaries inside positive words, and mixed substitution/deletion metadata. Only Addition event identity creates a positive boundary.

## Frozen future experiment

No future definition changed. The primary score remains `MEAN_UNEXPECTED_PHONE_MASS`. R6-1 will report event coverage, pooled boundary ROC-AUC, twelve speaker ROC-AUC values, median speaker ROC-AUC, and speakers above 0.55.

The frozen gates remain:

- F1 coverage `>= 0.99`
- F2 pooled AUC `>= 0.65`
- F3 median speaker AUC `>= 0.60`
- F4 at least 9/12 speakers with AUC `> 0.55`

## Protocol

No checkpoint was loaded. No inference, audio access, annotation-content access, real posterior computation, ROC-AUC, speaker metric, classifier fitting, threshold search, VALIDATION access, or TEST access occurred.

## Final status

`R6_0_LOCAL_BOUNDARY_STATIC_VERIFICATION_PASS`

## Next action

authorize exactly one frozen TRAIN-only `R6-1_LOCAL_BOUNDARY_EVIDENCE_TRAIN_FEASIBILITY` execution.
