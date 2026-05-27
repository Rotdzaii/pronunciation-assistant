# L2-ARCTIC Wav2Vec2 Context Encoder Training

## Purpose

The frozen Wav2Vec2 encoder classifier underperformed the best CNN runs on original phone-level crops. Many phone-level error segments are very short, often around 30 ms to 100 ms, so Wav2Vec2 may not receive enough acoustic context to produce useful representations. This experiment tests controlled context windows around each error segment.

## Wav2Vec2 ASR vs Encoder Classifier

| Approach | Description | Used here |
|---|---|---|
| Wav2Vec2 ASR baseline | Uses `Wav2Vec2ForCTC` to recognize text and compare recognized text with a target. | No |
| Wav2Vec2 encoder classifier | Uses `Wav2Vec2Model` hidden states as pretrained speech features, then trains an error-type classifier. | Yes |

This experiment does not use ASR. It does not compare recognized text with a target word. Classifier confidence is model confidence, not pronunciation correctness.

## Dataset

Clean Vietnamese L2-ARCTIC phone error dataset v2:

```txt
ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv
```

Target classes are `addition`, `deletion`, and `substitution`.

## Crop Modes

| Crop mode | Crop |
|---|---|
| `original_segment` | `start_time` to `end_time` |
| `context_0_10` | `start_time - 0.10s` to `end_time + 0.10s`, clipped to audio bounds |
| `context_0_15` | `start_time - 0.15s` to `end_time + 0.15s`, clipped to audio bounds |

All crops are padded or truncated to 1.0 second.

## Architecture

```txt
16 kHz mono audio crop
-> facebook/wav2vec2-base-960h Wav2Vec2Model
-> mean pooling over time
-> dropout
-> linear classifier
-> addition/deletion/substitution
```

The Wav2Vec2 encoder is frozen. Only the classifier head is trained.

## Training Setup

| Setting | Value |
|---|---|
| Encoder | `facebook/wav2vec2-base-960h` |
| Encoder state | frozen |
| Loss | normal `CrossEntropyLoss` |
| Balancing | `WeightedRandomSampler` only |
| Batch size | 4 |
| Epochs | 6 |
| Learning rate | 0.0001 |
| Pooling | mean |

Weighted loss is intentionally not used because previous CNN ablations showed that combining weighted sampling and weighted loss can over-balance minority classes.

## Evaluation Results

| Crop mode | Val macro F1 | Test macro F1 | Test addition F1 | Test deletion F1 | Test substitution F1 |
|---|---:|---:|---:|---:|---:|
| `original_segment` | 0.4054 | 0.3769 | 0.0860 | 0.5630 | 0.4818 |
| `context_0_10` | 0.4079 | 0.3803 | 0.1124 | 0.4257 | 0.6027 |
| `context_0_15` | 0.3529 | 0.3826 | 0.1143 | 0.3684 | 0.6650 |

Best crop mode by test macro F1: `context_0_15`.

Best crop mode by test addition F1: `context_0_15`.

Context improved addition F1 compared with the original Wav2Vec2 segment crop, but macro F1 stayed below the CNN baselines.

## Comparison

| Run | Test macro F1 | Test addition F1 | Test deletion F1 | Test substitution F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4657 | 0.0000 | 0.6341 | 0.7631 |
| v2 | 0.4835 | 0.1240 | 0.6444 | 0.6821 |
| sampler_only | 0.4803 | 0.0541 | 0.6555 | 0.7315 |
| binary_stage_pipeline | 0.4662 | 0.1096 | 0.6063 | 0.6828 |
| wav2vec2_encoder | 0.3769 | 0.0860 | 0.5630 | 0.4818 |
| wav2vec2_context_original_segment | 0.3769 | 0.0860 | 0.5630 | 0.4818 |
| wav2vec2_context_0_10 | 0.3803 | 0.1124 | 0.4257 | 0.6027 |
| wav2vec2_context_0_15 | 0.3826 | 0.1143 | 0.3684 | 0.6650 |

Ranking by test macro F1:

1. `v2`
2. `sampler_only`
3. `binary_stage_pipeline`
4. `baseline`
5. `wav2vec2_context_0_15`
6. `wav2vec2_context_0_10`
7. `wav2vec2_encoder`
8. `wav2vec2_context_original_segment`

Ranking by test addition F1:

1. `v2`
2. `wav2vec2_context_0_15`
3. `wav2vec2_context_0_10`
4. `binary_stage_pipeline`
5. `wav2vec2_encoder`
6. `wav2vec2_context_original_segment`
7. `sampler_only`
8. `baseline`

Conclusion: context helps Wav2Vec2 addition detection, but no Wav2Vec2 context model should be selected yet. CNN V2 remains the best overall model, and it still has the best addition F1.

## Limitations

The encoder is still frozen. Mean pooling is simple and may blur the localized phone error. Addition still has few samples. Context windows may include neighboring phones, silence, or noise that is not part of the target error. This is not a final pronunciation assessment model.

## Next Steps

If context helps, unfreeze the last Wav2Vec2 layers. Also try HuBERT, attention pooling, explicit localization features, and more addition samples.
