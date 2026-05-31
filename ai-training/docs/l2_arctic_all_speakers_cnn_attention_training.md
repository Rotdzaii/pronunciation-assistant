# All-Speaker L2-ARCTIC CNN Attention Training

## Purpose

This experiment trains the selected CNN Attention phone-error classifier on the expanded all-speaker L2-ARCTIC dataset:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

The goal is to test whether the same clean addition/deletion/substitution schema benefits from more L2-ARCTIC phone-error examples while still evaluating Vietnamese learners separately.

## Dataset

The all-speaker dataset contains 18,610 phone-error rows from 24 speakers.

Error distribution:

- `addition`: 1,092
- `deletion`: 3,420
- `substitution`: 14,098

L1 distribution:

- `Vietnamese`: 4,919
- `Spanish`: 3,415
- `Chinese`: 3,244
- `Hindi`: 2,871
- `Arabic`: 2,202
- `Korean`: 1,959

Audio was validated locally before training. All metadata rows resolved to existing `.wav` files.

## Model

The model reuses the previous CNN Attention architecture:

- input: 16 kHz audio segment from the annotated phone interval
- feature: 64-bin log-mel spectrogram
- max segment length: 1.0 second with zero padding/truncation
- CNN blocks: 1 -> 16 -> 32 -> 64 -> 96 channels
- temporal attention pooling after frequency averaging
- classifier: dropout plus linear layer
- label order: `addition`, `deletion`, `substitution`

Training uses the same balancing strategy as the previous CNN Attention baseline:

- `WeightedRandomSampler` on the training split
- unweighted `CrossEntropyLoss`
- Adam optimizer
- learning rate: `1e-4`
- batch size: `8`
- epochs: `12`
- random seed: `42`

Checkpoint path:

`ai-training/models/l2_arctic_all_speakers_cnn_attention.pt`

The checkpoint is local only and must not be committed.

## Final Validation Metrics

Best epoch: 12

Validation overall:

- accuracy: 0.6744
- macro F1: 0.5484
- weighted F1: 0.7083

Validation per-class F1:

- `addition`: 0.2126
- `deletion`: 0.6711
- `substitution`: 0.7616

Vietnamese validation subset:

- accuracy: 0.6888
- macro F1: 0.5517
- weighted F1: 0.7114
- `addition` F1: 0.1584
- `deletion` F1: 0.7576
- `substitution` F1: 0.7390

## Test Metrics

All-speaker test:

- samples: 997
- accuracy: 0.6469
- macro F1: 0.5050
- weighted F1: 0.6835
- `addition` F1: 0.1698
- `deletion` F1: 0.5930
- `substitution` F1: 0.7522

Vietnamese test subset:

- samples: 333
- accuracy: 0.6366
- macro F1: 0.5420
- weighted F1: 0.6538
- `addition` F1: 0.2769
- `deletion` F1: 0.6667
- `substitution` F1: 0.6824

## Comparison With Vietnamese-Only CNN Attention

Previous Vietnamese-only CNN Attention test:

- accuracy: 0.6366
- macro F1: 0.5105
- weighted F1: 0.6521
- `addition` F1: 0.1754
- `deletion` F1: 0.6667
- `substitution` F1: 0.6893

All-speaker CNN Attention on Vietnamese test subset:

- accuracy: 0.6366
- macro F1: 0.5420
- weighted F1: 0.6538
- `addition` F1: 0.2769
- `deletion` F1: 0.6667
- `substitution` F1: 0.6824

The all-speaker model improves Vietnamese-subset macro F1 and addition F1, while substitution F1 is slightly lower. On the full all-speaker test set, macro F1 is 0.5050, which is slightly below the previous Vietnamese-only test macro F1, though the scopes differ.

## Limitations

The all-speaker training set includes Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese speakers. It should not be interpreted as fully modeling Vietnamese-specific pronunciation characteristics.

Vietnamese-specific performance must be interpreted separately from overall all-speaker performance. The classifier confidence remains model confidence only; it is not a pronunciation correctness score.

Class imbalance remains significant. `substitution` dominates the dataset, and `addition` remains the hardest class despite sampler balancing.

## Decision

Recommendation: keep the all-speaker CNN Attention model as a strong experiment, not an immediate replacement for the Vietnamese-only model.

Reasoning:

- Vietnamese-subset macro F1 improved from 0.5105 to 0.5420.
- Vietnamese-subset addition F1 improved from 0.1754 to 0.2769.
- Full all-speaker test macro F1 is only 0.5050.
- The current split is utterance-based, not speaker-disjoint.

The next feature should run speaker-disjoint evaluation before deciding whether to replace the Vietnamese-only CNN Attention baseline.
