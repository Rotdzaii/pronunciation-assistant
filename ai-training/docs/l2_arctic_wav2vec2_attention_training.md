# L2-ARCTIC Wav2Vec2 Attention Pooling Training

## Purpose

Previous frozen Wav2Vec2 runs used mean pooling over time. Mean pooling can dilute phone-level error cues because the actual error may occupy only a few encoder frames. This experiment tests whether a lightweight attention pooling layer can focus the classifier on more informative frames.

## Wav2Vec2 Encoder, Not ASR

This experiment uses `Wav2Vec2Model` as a pretrained speech encoder. It does not use `Wav2Vec2ForCTC`, does not recognize text, and does not compare recognized text with a target word. Classifier confidence is model confidence, not pronunciation correctness.

## Dataset

Clean Vietnamese L2-ARCTIC phone error dataset v2:

```txt
ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv
```

Target classes are `addition`, `deletion`, and `substitution`.

## Crop Mode

The model uses the best context mode from the previous Wav2Vec2 context experiment:

| Crop mode | Crop |
|---|---|
| `context_0_15` | `start_time - 0.15s` to `end_time + 0.15s`, clipped to audio bounds |

Each crop is padded or truncated to 1.0 second.

## Mean Pooling vs Attention Pooling

Mean pooling gives every encoder frame equal weight. Attention pooling learns a scalar score for each Wav2Vec2 frame, applies softmax over time, and computes a weighted sum of hidden states. The goal is to let the classifier emphasize localized acoustic evidence instead of averaging it away.

## Architecture

```txt
16 kHz mono context crop
-> facebook/wav2vec2-base-960h Wav2Vec2Model
-> learned attention pooling over time
-> dropout
-> linear classifier
-> addition/deletion/substitution
```

The Wav2Vec2 encoder is frozen. Only the attention pooling layer and classifier head are trained.

## Training Setup

| Setting | Value |
|---|---|
| Encoder | `facebook/wav2vec2-base-960h` |
| Encoder state | frozen |
| Context | +/- 0.15s |
| Loss | normal `CrossEntropyLoss` |
| Balancing | `WeightedRandomSampler` only |
| Batch size | 4 |
| Epochs | 8 |
| Learning rate | 0.0001 |
| Pooling | attention |
| Dropout | 0.2 |

Weighted loss is intentionally not used because earlier experiments showed that combining weighted sampling and weighted loss can over-balance minority classes.

## Evaluation Results

| Split | Accuracy | Macro F1 | Weighted F1 | Addition F1 | Deletion F1 | Substitution F1 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.4364 | 0.3757 | 0.4833 | 0.1319 | 0.4686 | 0.5265 |
| test | 0.4324 | 0.3725 | 0.4697 | 0.1636 | 0.4344 | 0.5194 |

Test addition recall increased to 0.4737, but this came with many false positives and lower deletion/substitution performance.

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
| wav2vec2_attention | 0.3725 | 0.1636 | 0.4344 | 0.5194 |

Ranking by test macro F1:

1. `v2`
2. `sampler_only`
3. `binary_stage_pipeline`
4. `baseline`
5. `wav2vec2_context_0_15`
6. `wav2vec2_context_0_10`
7. `wav2vec2_encoder`
8. `wav2vec2_context_original_segment`
9. `wav2vec2_attention`

Ranking by test addition F1:

1. `wav2vec2_attention`
2. `v2`
3. `wav2vec2_context_0_15`
4. `wav2vec2_context_0_10`
5. `binary_stage_pipeline`
6. `wav2vec2_encoder`
7. `wav2vec2_context_original_segment`
8. `sampler_only`
9. `baseline`

## Conclusion

Attention pooling improved addition F1 compared with previous Wav2Vec2 mean-pooling runs and surpassed CNN V2 on addition F1. However, test macro F1 dropped to 0.3725, below CNN V2 and the context-window Wav2Vec2 mean-pooling runs. The model should be kept as an experiment, not selected as the main phone error classifier.

## Limitations

The encoder is still frozen. Attention pooling is simple and may not reliably locate the error frame. Addition still has few samples. Context windows may include neighboring phones or silence. This is not a final pronunciation assessment model.

## Next Steps

If attention pooling is promising, unfreeze the last Wav2Vec2 layers. Also try HuBERT, attention plus localization features, and more addition samples.
