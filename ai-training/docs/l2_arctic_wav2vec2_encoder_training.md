# L2-ARCTIC Wav2Vec2 Encoder Phone Error Classifier

## Purpose

CNN baselines are useful but limited for short phone-level pronunciation error segments. This experiment tests whether a stronger pretrained speech representation improves `addition`, `deletion`, and `substitution` classification on Vietnamese L2-ARCTIC data.

## ASR Baseline vs Encoder Classifier

There are two different uses of Wav2Vec2:

| Approach | Description | Used here |
|---|---|---|
| Wav2Vec2 ASR baseline | Uses `Wav2Vec2ForCTC` to recognize text, then compares recognized text with a target word or phrase. | No |
| Wav2Vec2 encoder classifier | Uses `Wav2Vec2Model` as a pretrained speech encoder and trains a classifier on hidden states. | Yes |

The old demo used Wav2Vec2 to recognize text. This experiment does not do ASR. It uses Wav2Vec2 as a pretrained speech encoder for phone-level error-type classification.

Model confidence is classifier confidence for the predicted error class. It is not pronunciation correctness.

## Dataset

Clean dataset v2:

```txt
ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv
```

Target classes:

| Class | Rows |
|---|---:|
| addition | about 198 |
| deletion | about 1566 |
| substitution | about 3155 |

Each sample uses the original `start_time` to `end_time` crop from the metadata CSV.

## Architecture

```txt
16 kHz mono audio segment
-> facebook/wav2vec2-base-960h Wav2Vec2Model
-> mean pooling over time
-> dropout
-> linear classifier
-> addition/deletion/substitution
```

The Wav2Vec2 encoder is frozen by default. Only the lightweight classifier head is trained.

## Training Setup

| Setting | Value |
|---|---|
| Model | `facebook/wav2vec2-base-960h` |
| Encoder | frozen |
| Pooling | mean over time |
| Loss | normal `CrossEntropyLoss` |
| Balancing | `WeightedRandomSampler` only |
| Batch size | 4 |
| Epochs | 8 |
| Learning rate | 0.0001 |
| Sample rate | 16000 |
| Max segment length | 1.0 second |

The sampler-only setup is intentional. Earlier CNN experiments showed that combining weighted sampling with weighted loss can over-balance the rare class and collapse common-class behavior.

## Evaluation Results

Checkpoint:

```txt
ai-training/models/l2_arctic_error_type_wav2vec2_encoder.pt
```

Best validation macro F1 during training: `0.4054` at epoch 3.

Validation:

| Metric | Value |
|---|---:|
| Accuracy | 0.4775 |
| Macro F1 | 0.4054 |
| Weighted F1 | 0.5028 |
| Addition F1 | 0.1507 |
| Deletion F1 | 0.5634 |
| Substitution F1 | 0.5022 |

Test:

| Metric | Value |
|---|---:|
| Accuracy | 0.4595 |
| Macro F1 | 0.3769 |
| Weighted F1 | 0.4873 |
| Addition F1 | 0.0860 |
| Deletion F1 | 0.5630 |
| Substitution F1 | 0.4818 |

Test confusion matrix:

| True / Predicted | addition | deletion | substitution |
|---|---:|---:|---:|
| addition | 4 | 6 | 9 |
| deletion | 17 | 76 | 22 |
| substitution | 53 | 73 | 73 |

## Comparison With CNN Baselines

Comparison output:

```txt
ai-training/datasets/l2-arctic/evaluation/wav2vec2_encoder_comparison.csv
```

| Run | Test macro F1 | Test addition F1 | Test deletion F1 | Test substitution F1 |
|---|---:|---:|---:|---:|
| baseline | 0.4657 | 0.0000 | 0.6341 | 0.7631 |
| v2 | 0.4835 | 0.1240 | 0.6444 | 0.6821 |
| sampler_only | 0.4803 | 0.0541 | 0.6555 | 0.7315 |
| binary_stage_pipeline | 0.4662 | 0.1096 | 0.6063 | 0.6828 |
| wav2vec2_encoder | 0.3769 | 0.0860 | 0.5630 | 0.4818 |

Ranking by test macro F1:

1. `v2`
2. `sampler_only`
3. `binary_stage_pipeline`
4. `baseline`
5. `wav2vec2_encoder`

Ranking by test addition F1:

1. `v2`
2. `binary_stage_pipeline`
3. `wav2vec2_encoder`
4. `sampler_only`
5. `baseline`

The frozen Wav2Vec2 encoder classifier should not be selected as the current model. It improved addition F1 over sampler-only, but its overall macro F1 and common-class F1 scores are below the CNN baselines.

## Limitations

The encoder is frozen, so the model may not adapt to phone-level L2 pronunciation cues. Segments are short and may omit useful context. The `addition` class still has very few examples. This is not a final pronunciation assessment model and should not be integrated until metrics are acceptable.

The current result suggests that frozen general-purpose speech representations are not enough by themselves for this phone-level error task. Fine-tuning or a better segment/context design is likely needed.

## Next Steps

1. Unfreeze the last Wav2Vec2 layers after the frozen-encoder baseline is understood.
2. Try HuBERT as a different pretrained speech encoder.
3. Try longer context windows with explicit localization features.
4. Collect or generate more reliable `addition` examples.
5. Improve alignment and phoneme localization quality.
6. Integrate only after validation and test metrics are acceptable.
