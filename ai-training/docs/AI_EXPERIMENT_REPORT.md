# AI Experiment Report

## 1. Purpose

This report consolidates the completed AI training experiments for phone-level English pronunciation error classification in the Pronunciation Assistant project. It summarizes the experiment goals, datasets, evaluation metrics, model comparisons, key findings, and the recommended next training direction.

## 2. Research Context

The research topic is: "Nghien cuu va phat trien he thong chan doan loi phat am tieng Anh tu dong dua tren Deep Learning."

The project goal is to develop an automatic English pronunciation error diagnosis system based on Deep Learning. In the current AI training scope, the task is phone-level classification of pronunciation error type:

- `addition`
- `deletion`
- `substitution`

The Wav2Vec2 ASR demo and the Wav2Vec2 encoder experiments are different tasks:

- The Wav2Vec2 ASR demo recognizes spoken text and demonstrates automatic speech recognition behavior.
- The Wav2Vec2 encoder experiments use Wav2Vec2 as a pretrained speech representation extractor, freeze the encoder, and train classifier heads for phone-level error type classification.

## 3. Dataset

The clean v2 classification dataset is:

`ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv`

It is derived from L2-ARCTIC Vietnamese speakers and TextGrid phone-level annotations. The classification labels are `addition`, `deletion`, and `substitution`.

Clean v2 distribution:

| Class | Count |
| --- | ---: |
| addition | 198 |
| deletion | 1,566 |
| substitution | 3,155 |
| total | 4,919 |

The dataset is strongly imbalanced. `addition` is severely underrepresented, with only about 4.0% of the clean v2 examples.

## 4. Evaluation Metrics

The experiments use these metrics:

- Accuracy: proportion of all examples classified correctly.
- Macro F1: unweighted average of class-level F1 scores.
- Weighted F1: F1 averaged by class support.
- Per-class F1: F1 for each error class.

Macro F1 is more important than accuracy for this task because the dataset is imbalanced. A model can obtain reasonable accuracy by favoring `substitution` and `deletion` while failing the rare `addition` class. Macro F1 gives each class equal weight and better reflects whether the model handles all error types.

Addition F1 is tracked separately because `addition` is the smallest and hardest class. It is also important for pronunciation diagnosis because a model that never detects additions is incomplete even if its overall accuracy looks acceptable.

## 5. Experiment Summary Table

| Experiment | Model / Method | Test Accuracy | Test Macro F1 | Test Addition F1 | Test Deletion F1 | Test Substitution F1 | Selection Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CNN baseline | Segment-level CNN baseline | 0.6966 | 0.4657 | 0.0000 | 0.6341 | 0.7631 | Not selected; fails addition |
| CNN V2 | CNN with imbalance handling, focal loss/sampler/augmentation | 0.6143 | 0.4835 | 0.1240 | 0.6444 | 0.6821 | Selected main candidate |
| Context V3 | Context-window CNN experiment | N/A | 0.4227 (reported from prior run) | 0.0000 (reported from prior run) | N/A | N/A | Not selected |
| Sampler-only CNN | CNN with WeightedRandomSampler-only retrain | 0.6667 | 0.4803 | 0.0541 | 0.6555 | 0.7315 | Not selected; over-balancing risk |
| Binary-stage CNN | Addition-vs-other stage plus deletion/substitution stage | 0.5946 | 0.4662 | 0.1096 | 0.6063 | 0.6828 | Not selected; improves addition but hurts pipeline |
| Wav2Vec2 encoder | Frozen Wav2Vec2 mean pooling on original segment | 0.4595 | 0.3769 | 0.0860 | 0.5630 | 0.4818 | Not selected |
| Wav2Vec2 context 0.15 | Frozen Wav2Vec2 with 0.15s context-window mean pooling | 0.5225 | 0.3826 | 0.1143 | 0.3684 | 0.6650 | Not selected |
| Wav2Vec2 attention | Frozen Wav2Vec2 with 0.15s context and attention pooling | 0.4324 | 0.3725 | 0.1636 | 0.4344 | 0.5194 | Not selected as main model; best addition F1 |
| CNN attention | CNN V2-style CNN with temporal attention pooling; 3-seed mean | 0.6246 | 0.5124 | 0.1938 | 0.6701 | 0.6734 | Selected main candidate after stability check |

Metrics are read from available evaluation JSON/CSV files where present. Context V3 metrics are partially unavailable in the current tree, so the known prior-run values are marked explicitly.

## 6. Key Findings

- CNN V2 is currently the best overall model by test macro F1.
- Addition is the hardest class because it has very few samples and low test support.
- Weighted loss plus weighted sampling can over-balance the classes and cause unstable behavior or collapse toward minority-class predictions.
- The binary-stage CNN improves addition F1 compared with sampler-only CNN, but the full pipeline loses too much overall accuracy and macro F1.
- The frozen Wav2Vec2 encoder underperforms the CNN models on this phone-level task.
- Adding context around Wav2Vec2 crops helps addition F1, but not enough to beat CNN V2 overall.
- Wav2Vec2 attention gives the best addition F1, showing attention pooling is promising for addition detection, but its macro F1 is too low for main-model selection.

## 7. Selected Main Model Candidate

CNN attention is selected as the current main model candidate after the 3-seed stability check.

CNN V2 was the previous main candidate because it had the best test macro F1 among the earlier completed experiments. The CNN attention experiment improves on CNN V2 by adding temporal attention pooling while keeping sampler-only imbalance handling and normal cross entropy.

Across seeds 42, 123, and 2026, CNN attention reached mean test macro F1 0.5124 with standard deviation 0.0214, and mean test addition F1 0.1938 with standard deviation 0.0415. These means exceed CNN V2's test macro F1 0.4835 and addition F1 0.1240. Wav2Vec2 attention is still not selected as the main model because its overall macro F1 is too low.

## 7.1. CNN Attention Stability Check

A 3-seed stability check was run for CNN attention using seeds 42, 123, and 2026. The check reused the clean v2 dataset, original segment crops, log-mel spectrogram preprocessing, temporal attention pooling, `WeightedRandomSampler` only, and unweighted `CrossEntropyLoss`.

| Metric | Mean | Std |
| --- | ---: | ---: |
| test accuracy | 0.6246 | 0.0235 |
| test macro F1 | 0.5124 | 0.0214 |
| test weighted F1 | 0.6449 | 0.0152 |
| test addition F1 | 0.1938 | 0.0415 |
| test deletion F1 | 0.6701 | 0.0192 |
| test substitution F1 | 0.6734 | 0.0206 |

All three seeds individually beat CNN V2 on test macro F1 and test addition F1. Therefore, CNN attention replaces CNN V2 as the current selected model candidate. CNN V2 remains an important previous baseline in the experiment history.

## 8. Limitations

- Addition has too few samples in the clean v2 dataset.
- Test addition support is small, so addition F1 can be sensitive to a small number of predictions.
- Current labels are derived from TextGrid annotations and inherit the limits of the annotation process.
- CNN models are still baseline-level models, not final production pronunciation assessment models.
- Wav2Vec2 encoder experiments used a frozen encoder, so the pretrained representation was not adapted to this task.
- Mean pooling and attention pooling may still lose fine-grained localization needed for very short phone-level errors.
- Current models classify known phone-level error segments; they are not complete end-to-end pronunciation assessment systems.
- Confidence scores are model confidence for predicted classes, not direct pronunciation correctness scores.

## 9. Recommended Next Direction

Recommended feature branch:

`feature/ai-cnn-v2-attention-improvement`

Goal: improve CNN V2 while preserving macro F1 and improving addition F1.

Potential methods:

- CNN V2 plus attention pooling.
- Class-balanced batch sampling without excessive over-balancing.
- Addition-focused light augmentation.
- Threshold tuning for addition.
- Later, a hybrid CNN V2 model that uses a Wav2Vec2 attention-derived signal.

The next work should target CNN V2 specifically instead of randomly training more architectures.

## 10. Report Conclusion

The completed experiments establish a reproducible AI training pipeline for phone-level pronunciation error classification. CNN V2 was the strongest earlier baseline, and the CNN attention stability check now identifies CNN attention as the strongest current model candidate. Future work should focus on error analysis, addition false positives/false negatives, and further validation before integrating this classifier into a larger pronunciation assessment flow.
