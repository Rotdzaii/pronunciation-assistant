# Hybrid Diagnosis Pipeline

## Purpose

The hybrid diagnosis layer combines alignment, CNN Attention segment diagnosis, and segmental scoring into a clearer AI Worker output for the app.

It does not train a model, implement real GOP, or turn classifier confidence into a pronunciation score.

## Why Hybrid Is Needed

CNN Attention predicts error type. Alignment provides approximate or forced locations. Scoring provides pronunciation score/severity evidence. A single app response needs all three, but each signal has different meaning and reliability.

Hybrid diagnosis keeps these roles separate:

- diagnosis confidence: confidence in a predicted error class
- pronunciation score: segmental score from scoring output
- severity: user-facing issue level derived from score and diagnosis evidence
- feedback: practice guidance generated from the selected top issues

## Current Inputs

The current scaffold uses:

- alignment result from `alignment_service`
- CNN Attention segment predictions
- `heuristic_gop` scoring result

The hybrid layer selects top issues by combining:

- predicted error type is not `unknown`
- lower phone score when available
- higher diagnosis confidence

## Current Output

The hybrid result includes:

- `hybrid_status`
- `hybrid_method`
- `primary_error_type`
- `severity`
- `top_issues`
- `problem_phonemes`
- `feedback`
- `location_reliability`
- heuristic scoring flags

Aligned CNN Attention results include these fields under `diagnosis` and `metadata` while preserving backward-compatible top-level fields.

## Limitations

Fallback alignment is approximate and limits location reliability.

`heuristic_gop` is not real GOP. It is a temporary scaffold without acoustic posterior or phone likelihood scoring.

Classifier confidence remains diagnosis confidence. It must not be displayed as pronunciation correctness.

## Future Upgrade

Planned upgrades:

- replace fallback alignment with MFA phone boundaries where available
- replace `heuristic_gop` with real GOP/CaGOP scoring
- calibrate severity thresholds with validation data
- tune hybrid issue ranking using real alignment and scoring reliability

Until those pieces exist, hybrid output must continue to label fallback and heuristic components honestly.
