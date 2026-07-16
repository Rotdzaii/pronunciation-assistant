# Hybrid Diagnosis Pipeline

## Purpose

The hybrid diagnosis layer combines alignment and CNN Attention segment
diagnosis into a clearer AI Worker output for the app.

It does not train a model, add a GOP runtime, or turn classifier confidence
into a pronunciation score.

## Why Hybrid Is Needed

CNN Attention predicts error type. Alignment provides approximate or forced
locations. The current model does not provide pronunciation score/severity
evidence. A single app response needs diagnosis and location reliability, and
must keep those signals separate.

Hybrid diagnosis keeps these roles separate:

- diagnosis confidence: confidence in a predicted error class
- pronunciation score: unavailable until a learned quality head is trained
- severity: advisory issue level derived from diagnosis and reliability, not a
  pronunciation score
- feedback: practice guidance generated from the selected top issues

## Current Inputs

The current scaffold uses:

- alignment result from `alignment_service`
- CNN Attention segment predictions
- optional heuristic diagnostic data, never a public score

The hybrid layer selects top issues by combining:

- predicted error type is not `unknown`
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

`heuristic_gop` is internal diagnostic scaffolding, not a public pronunciation
score. GOP/CaGOP is not the Phoenix v2 roadmap.

Classifier confidence remains diagnosis confidence. It must not be displayed as pronunciation correctness.

## Future Upgrade

Planned upgrades:

- replace fallback alignment with MFA phone boundaries where available
- add audited correct samples and a learned correctness head
- research a learned quality head after obtaining suitable quality labels
- tune hybrid issue ranking using alignment and learned-output reliability

Until those pieces exist, hybrid output must continue to label fallback and heuristic components honestly.
