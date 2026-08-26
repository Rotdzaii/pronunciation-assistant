# R4-3B Frozen Word-Level Sequence Deletion Validation

Status: `R4_3B_DESIGN_INCOMPLETE`

The authoritative R4-3A preregistration was hashed and inspected before any TRAIN/VALIDATION acoustic inference or sequence scoring. It freezes the evidence source, 40 ms stride, allowed operations, TRAIN-only global Laplace-smoothed priors, and validation gates. It does **not** freeze enough numerical behavior to produce a unique deterministic path.

Missing items include the Laplace coefficient, prior denominator and cost transform, acoustic span aggregation and bounds, MATCH/SUBSTITUTION rule, ADVANCE_TIME recurrence/cost, DP boundary conditions, deterministic tie ordering, and near-equal ambiguity tolerance. Multiple plausible implementations comply with the prose but can produce different deletion predictions.

Per the task's stop condition, no operation priors were computed, no audio was read, no validation logits or paths were generated, and no metrics were calculated.

- R4 TEST accessed: **NO**
- TEST paths resolved: **NO**
- New neural training: **NO**
- R3 checkpoint modified: **NO**
- MFA runtime modified: **NO**

Next smallest action: create a new pre-validation design-freeze gate defining the complete numerical recurrence and tie policy. Do not run R4-3B validation until that artifact is frozen.
