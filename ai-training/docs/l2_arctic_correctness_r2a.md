# Phoenix correctness research — R2-A baseline

Status: `R2A_VALIDATION_FAIL`

This experiment is **RESEARCH_ONLY**, **NOT_USED_BY_RUNTIME**, and
**NOT_PRODUCTION_MODEL**. It did not modify the V2/V3 datasets, the legacy
training entrypoint, the AI Worker, runtime code, or existing checkpoints.

## Locked experiment

- Dataset: `all_speakers_phone_correctness_v3.csv`
- SHA-256: `433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E`
- Labels: `correct -> 0`; `substitution + deletion -> 1`; addition excluded
- Eligible rows: 119,144 (101,626 correct; 17,518 incorrect)
- Train: BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA
- Validation: ABA, HKK, HQTV, LXC, MBMPS, SVBI
- Test: ASI, ERMS, SKA, THV, TXHC, YDCK
- Input: waveform-derived log-mel only; expected phone and all metadata fields
  were excluded from model features.
- Window: uniform centered one-second mono window at 16 kHz, with boundary-only
  zero padding.
- Model: randomly initialized legacy `SmallPronunciationCNNAttention` backbone,
  binary `Dropout(0.2) + Linear(96, 2)` head.
- Loss: train-only class-weighted cross entropy; weights 0.579494163 correct and
  3.644884973 incorrect.
- Seed 42; batch 8; Adam 1e-4; 12 epochs; no augmentation; no weighted sampler;
  argmax threshold.

## Result

The selected checkpoint was epoch 11 (highest validation Macro-F1). Validation:

- Accuracy: 0.764639
- Macro-F1: 0.542294
- Balanced accuracy: 0.542661
- Incorrect precision / recall / F1: 0.221027 / 0.225582 / 0.223282
- Substitution recall: 0.224964
- Deletion recall: 0.227783
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[21754, 3549], [3457, 1007]]`

All six locked validation thresholds failed. The test set was **not evaluated**,
and no `test_report.json` was created. Validation duration matching retained
4,434 pairs / 8,868 rows and produced Macro-F1 0.495820 and incorrect F1
0.333389. This is diagnostic only and does not alter the validation decision.

Per-speaker incorrect recall ranged from 0.136476 (SVBI) to 0.342043 (HKK).
ABA had especially low incorrect precision/F1 (0.088727 / 0.122214), while LXC
had the strongest per-speaker incorrect F1 (0.291667). These are diagnostic
anomalies, not grounds to tune or change the locked R2-A run.

Artifacts are under `ai-training/experiments/r2a_correctness_seed42/`. The
checkpoint is retained for diagnosis only and must not be integrated into the
runtime.
