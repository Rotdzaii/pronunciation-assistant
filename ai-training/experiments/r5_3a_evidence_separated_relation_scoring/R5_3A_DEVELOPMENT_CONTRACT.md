# R5-3A Evidence-Separated Relation Scoring TRAIN Development Contract

## Decision frozen

R5-3A is a new addition-scoring generation. It tests whether separate `[A,S,D]` evidence channels can retain Addition evidence while allowing a fixed linear classifier to account for substitution and deletion confounders.

R5-2B remains closed and unchanged. Its hard maximum is not reused as a feature.

## Frozen evidence motivating the change

The R5-2C post-hoc diagnosis was `MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION`. R5-2B reduced substitution FAR from `0.05016116035455278` to `0.017123287671232876` and deletion FAR from `0.03349964362081254` to `0.006414825374198147`. However, SUB/DELETE won for 291/323 true Addition words, 40 R5-1A TPs became R5-2B FNs, and 249 correct TNs became R5-2B FPs. BEST_INSERT remained identical for all 16,582 words.

This supports retaining relation evidence but testing a different, preregistered fusion mechanism.

## Frozen contract summary

- Population: exact 16,582-word TRAIN join; 323 positive and 16,259 negative.
- Feature order: `[A,S,D]` only.
- A: frozen R5-1A BEST_INSERT-minus-KEEP evidence.
- S: frozen R5-2B BEST_SUB-minus-KEEP evidence.
- D: frozen R5-2B BEST_DELETE-minus-KEEP evidence.
- Scaler: calibration-only StandardScaler in every speaker fold.
- Model: fixed balanced L2 LogisticRegression with `C=1.0`, `lbfgs`, `max_iter=1000`.
- Continuous output: `predict_proba(X_scaled)[:,1]`.
- Development: exact 12-fold TRAIN speaker LOSO.
- Threshold: frozen calibration-only unique-score/nextafter procedure and existing tie order.
- Event: unchanged frozen BEST_INSERT phone and boundary.
- Gates: all eight preregistered gates must pass.
- Robust threshold: authorized only after 8/8 PASS.

## Contract-stage protocol

No neural training, checkpoint inference, classifier fitting, performance metrics, threshold search, VALIDATION access, or TEST access occurred. Frozen R5-1A, R5-2B, and R5-2C artifacts were not modified.

## Required next stage

R5-3A synthetic static verification must pass before any TRAIN development fitting or performance calculation.

## Status

`R5_3A_DEVELOPMENT_CONTRACT_FROZEN`
