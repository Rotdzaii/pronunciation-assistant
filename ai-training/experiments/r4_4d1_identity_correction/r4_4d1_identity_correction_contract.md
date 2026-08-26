# R4-4D1C Matched-Control Row-Identity Correction Contract

Status: **FROZEN BEFORE `locked_execution_v2`**

This contract is additive. It does not modify the original R4-4D1 preregistration, numerical contract, frozen v1 driver, threshold, acoustic checkpoint, scoring formula, metrics, gates, or the preserved `locked_execution_v1` failure evidence.

## Identity semantics

V4 is read with `csv.DictReader`, and its data records are enumerated from zero. Consequently, `source_index = 0` denotes the first data record. That record occupies physical CSV line 2 because physical line 1 contains the header.

The frozen matched-control field `source_csv_row` denotes the one-based physical CSV line including the header. The only canonical conversion is:

```text
canonical_source_csv_row = int(source_index) + 2
```

The v1 driver incorrectly exported `int(source_index)` directly. Metadata-only reproduction gives 1,201 mapped and 233 missing frozen identities. The canonical conversion maps all 1,434 identities with zero missing rows, duplicate collisions, ambiguities, or mismatches in speaker, utterance, timing, expected phone, or relation.

No other identity, acoustic, score, prediction, or ground-truth field changes.

## Authoritative threshold

- Score family: RAW
- Theta: `2.197946548461914`
- Artifact: `ai-training/experiments/r4_4d1_locked_hypothesis_validation/locked_execution_v1/train_calibrated_threshold.json`
- SHA-256: `36F6FD5AB6B7E98A607D499445E455DCAB8C3DD4ACDD19F252DC472FCDD07E94`

The v2 driver must load and verify this artifact. TRAIN scoring, threshold candidate construction, eligibility evaluation, calibration, and reselection are prohibited.

## Validation-only v2

The future `locked_execution_v2` is a technical re-execution. V1 remains a recorded technical identity-contract stop. V2 is scientifically justified because v1 fixed and hashed its threshold using TRAIN only, exposed no validation metrics, and made no validation-derived decision change.

The future sequence is:

1. Verify all frozen contracts and source hashes.
2. Verify the immutable threshold SHA and exact theta.
3. Verify the canonical matched-control metadata mapping is exactly 1,434/1,434.
4. Perform one independently generated VALIDATION hypothesis-score pass.
5. Apply the unchanged threshold, decision rule, metrics, gates, diagnostics, and classification.

RAW CTC scoring, KEEP/DELETE/39-SUB hypotheses, BEST_SUB tie behavior, `D_i`, all metric definitions, eight confirmation gates, Strong Partial, Threshold Transfer Fail, classification precedence, comparators, and diagnostic groupings remain exactly as frozen in R4-4D1A.

This task performs no TRAIN recalibration, VALIDATION inference/scoring/metrics, neural training, or R4 TEST access.
