# R5-2C Frozen Relation-Competition Failure Audit

## Identity and join

- R5-1A manifest: `C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6` — PASS (14/14)
- R5-2B manifest: `37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387` — PASS (21/21)
- Exact stable-identity join: 16,582 rows; 0 missing; 0 duplicate; 0 excluded.

## Fundamental score identity

The constructed identity term `BEST_NON_ADDITION - KEEP` was nonnegative for all 16,582 words. Frozen cross-execution `A-C` contained tiny float32 representation residuals down to -9.4622373580932617e-07; all were exactly explained by the separately frozen INSERT-minus-KEEP representation delta. No material score-identity violation occurred.

## Main diagnosis

- Positive BEST_NON_ADDITION winners: {'SUB': 248, 'DELETE': 43, 'KEEP': 32}.
- Addition/all AUC: 0.7734833025081417 -> 0.6878413803490975.
- Addition/correct AUC: 0.802352821409537 -> 0.669239360205041.
- Correct-only TN -> FP transitions: 249.
- BEST_INSERT phone/boundary identical: 16,582/16,582.

## Frozen interpretation

`MIXED_OVER_SUPPRESSION_AND_THRESHOLD_COMPENSATION`

Relation competition genuinely suppressed substitution/deletion false additions, but it also allowed SUB/DELETE explanations to suppress many true additions. The LOSO thresholds then moved downward to compensate, introducing correct-only false positives. Event localization itself did not change; event F1 declined because the binary decision set changed.

## Protocol

No training, checkpoint inference, audio access, threshold search, new scores, new predictions, VALIDATION access, or TEST access occurred. Frozen R5-1A and R5-2B artifacts were not modified.

## Status

`R5_2C_POSTHOC_FAILURE_AUDIT_COMPLETE`
