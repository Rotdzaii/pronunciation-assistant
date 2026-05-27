# L2-ARCTIC CNN Attention Phone Error Type Classifier

## Why CNN V2 Attention Was Tried

The consolidated AI experiment report selected CNN V2 as the strongest main model candidate because it had the best overall test macro F1 among completed runs:

- CNN V2 test macro F1: 0.4835
- CNN V2 test addition F1: 0.1240
- Wav2Vec2 attention test addition F1: 0.1636
- Wav2Vec2 attention test macro F1: 0.3725

The report recommended improving CNN V2 in a targeted way instead of randomly training more models. Since Wav2Vec2 attention improved the rare `addition` class but hurt overall macro F1, this experiment adds attention pooling to a CNN V2-style model.

## Difference From CNN V2

CNN V2 used a CNN feature extractor followed by global average pooling. It also used focal loss, weighted loss, weighted sampling, and augmentation in its original training script.

The CNN attention experiment keeps the segment-level CNN feature extractor style but replaces global average pooling with temporal attention pooling:

- Input: original phone error segment crop from `start_time` to `end_time`.
- Feature extraction: log-mel spectrogram into convolution blocks.
- Pooling: mean over frequency, then learned attention over time.
- Classifier: dropout plus linear output layer.
- Imbalance handling: `WeightedRandomSampler` only.
- Loss: normal unweighted `CrossEntropyLoss`.

This avoids combining weighted loss and weighted sampler, which previous experiments showed can over-balance the training distribution.

## Dataset

Dataset:

`ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv`

Classes:

- `addition`
- `deletion`
- `substitution`

Clean v2 split distribution:

| Split | Addition | Deletion | Substitution |
| --- | ---: | ---: | ---: |
| train | 149 | 1,274 | 2,652 |
| val | 30 | 177 | 304 |
| test | 19 | 115 | 199 |

Addition remains the smallest class.

## Model Architecture

The model is implemented in:

`ai-training/scripts/train_l2_arctic_error_type_cnn_attention.py`

Architecture summary:

- Log-mel input with 64 mel bins.
- Four convolution layers with batch normalization and ReLU.
- Two max-pooling stages.
- Frequency-mean reduction from CNN feature map.
- Learned temporal attention over the remaining time axis.
- Dropout 0.2.
- Linear classifier for three error classes.

## Training Setup

Command:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\train_l2_arctic_error_type_cnn_attention.py
```

Configuration:

| Setting | Value |
| --- | ---: |
| sample rate | 16000 |
| n_mels | 64 |
| max_seconds | 1.0 |
| batch_size | 8 |
| epochs | 12 |
| learning_rate | 0.0001 |
| num_workers | 0 |
| random_seed | 42 |
| dropout | 0.2 |

Best validation checkpoint:

- best epoch: 12
- best validation macro F1: 0.5388
- checkpoint: `ai-training/models/l2_arctic_error_type_cnn_attention.pt`

The checkpoint is generated locally and must not be committed.

GPU telemetry summary:

- CUDA available: true
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- peak allocated memory: about 72.15 MB
- peak reserved memory: about 92.00 MB
- `nvidia-smi` showed low GPU utilization during the small-CNN workload, which is expected for this lightweight model and audio-loading-heavy pipeline.

## Evaluation Results

Command:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\evaluate_l2_arctic_error_type_cnn_attention.py
```

Validation results:

| Metric | Value |
| --- | ---: |
| accuracy | 0.6673 |
| macro F1 | 0.5388 |
| weighted F1 | 0.6900 |
| addition F1 | 0.1782 |
| deletion F1 | 0.7087 |
| substitution F1 | 0.7296 |

Test results:

| Metric | Value |
| --- | ---: |
| accuracy | 0.6366 |
| macro F1 | 0.5105 |
| weighted F1 | 0.6521 |
| addition F1 | 0.1754 |
| deletion F1 | 0.6667 |
| substitution F1 | 0.6893 |

Test confusion matrix:

| Actual | Predicted addition | Predicted deletion | Predicted substitution |
| --- | ---: | ---: | ---: |
| addition | 5 | 4 | 10 |
| deletion | 7 | 85 | 23 |
| substitution | 26 | 51 | 122 |

Generated evaluation files:

- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_eval_metrics.json`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_confusion_matrix.csv`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_per_class_metrics.csv`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_per_speaker_metrics.csv`
- `ai-training/datasets/l2-arctic/evaluation/cnn_attention_misclassified_examples.csv`

Confidence values are classifier confidence for the predicted class, not pronunciation correctness.

## Comparison

Command:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\compare_error_type_cnn_attention.py
```

| Run | Test Accuracy | Test Macro F1 | Test Addition F1 | Test Deletion F1 | Test Substitution F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 0.6966 | 0.4657 | 0.0000 | 0.6341 | 0.7631 |
| CNN V2 | 0.6143 | 0.4835 | 0.1240 | 0.6444 | 0.6821 |
| Sampler-only CNN | 0.6667 | 0.4803 | 0.0541 | 0.6555 | 0.7315 |
| Binary-stage CNN | 0.5946 | 0.4662 | 0.1096 | 0.6063 | 0.6828 |
| Wav2Vec2 attention | 0.4324 | 0.3725 | 0.1636 | 0.4344 | 0.5194 |
| CNN attention | 0.6366 | 0.5105 | 0.1754 | 0.6667 | 0.6893 |

CNN attention improves over CNN V2:

- Test macro F1: 0.5105 vs 0.4835
- Test addition F1: 0.1754 vs 0.1240
- Test deletion F1: 0.6667 vs 0.6444
- Test substitution F1: 0.6893 vs 0.6821

CNN attention also beats Wav2Vec2 attention on both overall macro F1 and addition F1 in this run:

- Test macro F1: 0.5105 vs 0.3725
- Test addition F1: 0.1754 vs 0.1636

## Inference Demo

Command:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\infer_l2_arctic_error_type_cnn_attention.py --row-index 0
```

Result:

- speaker_id: HQTV
- utterance_id: arctic_a0003
- ground_truth_error_type: deletion
- predicted_error_type: deletion
- confidence: 0.485376
- class probabilities:
  - addition: 0.260469
  - deletion: 0.485376
  - substitution: 0.254154
- is_correct: true

Confidence is model confidence, not pronunciation correctness.

## Should CNN Attention Replace CNN V2?

Yes, CNN attention should replace CNN V2 as the current best experiment and main model candidate for the phone-level error type classifier.

It preserves and improves overall macro F1 while also improving addition F1. The result directly satisfies the goal of improving CNN V2 with attention pooling.

This should still be treated as a research checkpoint rather than a final pronunciation assessment system. A repeat run and error review are recommended before integrating it into any downstream application flow.

## Limitations

- Addition support is still very small, especially in validation and test.
- The model classifies known phone-level error segments; it does not detect arbitrary pronunciation errors end to end.
- Labels are derived from TextGrid annotations and inherit annotation limitations.
- Attention weights are not yet validated as reliable phone-localization explanations.
- The model uses original segment crops only; very short segments may still provide limited acoustic evidence.
- Confidence is classifier confidence, not pronunciation correctness.

## Next Steps

- Repeat the CNN attention run to check seed stability.
- Review addition false positives and false negatives.
- Test light addition-focused augmentation while keeping sampler-only imbalance handling.
- Consider threshold tuning for addition if the deployment target values addition recall.
- Later, evaluate a hybrid CNN attention model with explicit phone identity or Wav2Vec2 attention-derived features.
