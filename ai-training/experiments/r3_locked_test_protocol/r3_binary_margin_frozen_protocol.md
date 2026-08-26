# R3 Binary Margin Frozen Test Protocol

Status: **VALIDATION_CALIBRATED**, **FROZEN_BEFORE_TEST**,
**NOT_PRODUCTION_CALIBRATED**, **RESEARCH_ONLY**.

This protocol was frozen before any TEST audio resolution, reading, feature
extraction, inference, threshold fitting, score inspection, or metric
calculation. The future TEST task must verify this document, the companion
manifest, and the acoustic checkpoint by SHA-256 before inference.

## Locked acoustic model

- Checkpoint: `ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt`
- Checkpoint SHA-256: `5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E`
- Selected epoch: 47
- V4 SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Input: audio-derived log-mel only
- Model: `SmallPronunciationCNNAttention`, CNN channels 16→32→64→96,
  temporal attention, `Dropout(0.2)`, `Linear(96,40)`
- Vocabulary by index: AA, AE, AH, AO, AW, AX, AY, B, CH, D, DH, EH,
  ER, EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S,
  SH, T, TH, UH, UW, V, W, Y, Z, ZH

The model must not be retrained or adapted before the locked TEST evaluation.
It runs in deterministic evaluation mode and generates all 40 logits plus its
ordinary argmax prediction.

## Locked preprocessing

Audio is mono at 16 kHz. The crop is centered at
`(start_time + end_time) / 2`, spans center ±0.25 seconds, and contains
exactly 8,000 samples. At utterance edges, available audio is clipped and
only the missing side is zero-padded.

Log-mel extraction uses 64 mel bins, FFT/window length 2,048, hop 512, Hann
window, centered constant padding, power 2, Slaney scale and normalization,
and power-to-dB relative to the per-sample maximum with 80 dB clipping. The
fixed feature shape is `[1,64,16]` per sample.

## Locked expected-phone usage and margin

`expected_phone_canonical` is forbidden as neural input. It may be retrieved
only after the acoustic model has produced all 40 observed-phone logits:

```text
audio
→ 40 observed-phone logits
→ retrieve expected_phone_canonical
→ expected-vs-best-alternative margin
→ frozen threshold
→ correct/substitution
```

The exact score is:

```text
expected_logit = logit[expected_phone_canonical]

best_alternative_logit =
    max(logit[p]) for all 39 phones p != expected_phone_canonical

expected_margin = expected_logit - best_alternative_logit
```

Softmax is not required for this decision.

## Frozen binary rule

```text
if expected_margin <= -1.293920:
    substitution
else:
    correct
```

The comparison is inclusive: equality is `substitution`. The operational
threshold is exactly the decimal `-1.293920`. Its source candidate at full
stored precision was `-1.2939202785491943`; rounding to the frozen decimal
changes zero validation predictions.

This threshold was selected using all six validation speakers—ABA, HKK,
HQTV, LXC, MBMPS, and SVBI—with calibration substitution recall constrained
to at least 0.50. Among feasible thresholds, selection maximized binary
Macro-F1, then substitution F1, then substitution precision, then preferred
the more conservative threshold producing fewer substitution predictions.

There is one global threshold. Per-phone, per-speaker, pair-specific, and
TEST-derived thresholds are forbidden.

## Validation calibration evidence

At the frozen decimal threshold, the 28,212 validation rows produce:

- Accuracy: 0.772934
- Balanced accuracy: 0.652840
- Binary Macro-F1: 0.588474
- Substitution precision: 0.227436
- Substitution recall: 0.501547
- Substitution F1: 0.312956
- Confusion matrix, labels `[correct, substitution]`:
  `[[20347,4956],[1450,1459]]`

## Locked TEST report

The locked TEST speakers are ASI, ERMS, SKA, THV, TXHC, and YDCK. No
speaker-specific threshold may be fitted.

Primary binary metrics:

- Binary Macro-F1
- Substitution precision, recall, and F1
- Balanced accuracy
- Accuracy
- Confusion matrix

Acoustic observed-phone metrics:

- Top-1 accuracy
- Macro-F1
- Balanced accuracy
- Top-3 accuracy

For each TEST speaker, report binary Macro-F1 and substitution precision,
recall, and F1. Report correct/substitution support and recalls for expected
phones TH, DH, R, V, D, T, S, and Z. If supported, inspect TH→T, DH→D,
R→L, V→W, and Z→S. These diagnostics cannot change the checkpoint or
threshold.

## Pre-registered interpretation bands

These are research interpretation criteria, not production acceptance
criteria.

`TEST_TRANSFER_CONFIRMED` requires all of:

- Binary Macro-F1 ≥0.55
- Substitution recall ≥0.45
- Substitution F1 ≥0.28
- No catastrophic speaker collapse

`TEST_TRANSFER_PARTIAL` applies when full confirmation is not achieved but
all of these hold:

- Binary Macro-F1 ≥0.50
- Substitution recall ≥0.35
- Substitution F1 ≥0.22
- No catastrophic speaker collapse

`TEST_TRANSFER_NOT_CONFIRMED` applies if any partial criterion fails or
catastrophic speaker collapse occurs.

To make the pre-test phrase operational rather than post-hoc, catastrophic
speaker collapse means any TEST speaker having at least 10 clean substitution
rows with substitution recall below 0.20 or binary Macro-F1 below 0.40.

## Known pre-TEST failure mode

DH→D is a known validation failure mode. R3-2B measured support 300,
detection recall 0.216667, 235 false negatives, and median expected margin
+0.051111. The acoustic model frequently continues to favor expected DH when
manual annotation indicates observed D.

Poor TEST behavior on this pair cannot trigger a post-hoc threshold or model
change within the same locked evaluation.

## Score and amendment policy

No expected-margin-to-0–100 mapping exists. Do not sigmoid-scale the margin,
min-max normalize validation, multiply posterior by 100, or claim calibrated
pronunciation scores. Current outputs are only correct/substitution, raw
expected margin, and best-alternative phone.

This freeze is immutable-style: any methodology change requires a new
version and new hashes. Do not overwrite these artifacts. Opening TEST
requires a separate explicit task.
