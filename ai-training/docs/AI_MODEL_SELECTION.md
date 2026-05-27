# AI Model Selection

## 1. Purpose

This document records the selected model after the current AI training phase for the Pronunciation Assistant project. It summarizes the final model choice, supporting metrics, inference sanity checks, limitations, and the recommended next phase.

## 2. Selected Model

CNN attention phone error classifier is selected as the current main model candidate.

- Task: phone-level pronunciation error classification
- Classes: `addition`, `deletion`, `substitution`
- Dataset: L2-ARCTIC Vietnamese speakers clean v2
- Local checkpoint path: `ai-training/models/l2_arctic_error_type_cnn_attention.pt`
- Checkpoint status: generated locally only and not committed to Git

## 3. Why CNN Attention Was Selected

CNN V2 was the previous best model, with test macro F1 around 0.4835 and test addition F1 around 0.1240. CNN attention improved both mean test macro F1 and mean addition F1 across the 3-seed stability check.

CNN attention is selected because:

- It improves mean test macro F1 from CNN V2's 0.4835 to 0.5124.
- It improves mean test addition F1 from CNN V2's 0.1240 to 0.1938.
- All three stability seeds beat CNN V2 on test macro F1 and addition F1.
- Wav2Vec2 attention had strong addition F1 in a prior run, but its overall macro F1 was too low, so it was not selected.

## 4. Stability Check Summary

| Seed | Test Macro F1 | Test Addition F1 | Test Deletion F1 | Test Substitution F1 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.4913 | 0.1471 | 0.6484 | 0.6784 |
| 123 | 0.5341 | 0.2264 | 0.6848 | 0.6910 |
| 2026 | 0.5119 | 0.2078 | 0.6772 | 0.6507 |

Stability summary:

- Mean test macro F1: 0.5124
- Std test macro F1: 0.0214
- Mean test addition F1: 0.1938
- Std test addition F1: 0.0415
- Mean test accuracy: 0.6246
- Std test accuracy: 0.0235

## 5. Comparison Against Previous Best

| Model | Test Macro F1 | Test Addition F1 |
| --- | ---: | ---: |
| CNN V2 | 0.4835 | 0.1240 |
| CNN attention mean | 0.5124 | 0.1938 |

CNN attention improves both the main imbalanced-data metric, macro F1, and the rare-class metric, addition F1.

## 6. Inference Sanity Check

| Example | Speaker | Utterance | Ground Truth | Prediction | Confidence | Addition Prob. | Deletion Prob. | Substitution Prob. | Correct |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | HQTV | arctic_a0003 | deletion | deletion | 0.485376 | 0.260469 | 0.485376 | 0.254154 | true |
| 2 | HQTV | arctic_a0003 | deletion | deletion | 0.382947 | 0.356107 | 0.382947 | 0.260946 | true |
| 3 | HQTV | arctic_a0022 | substitution | deletion | 0.587655 | 0.208726 | 0.587655 | 0.203619 | false |

The first two examples are correct deletion predictions with moderate classifier confidence. The third example shows a substitution segment misclassified as deletion. These examples confirm that inference runs end to end, while also showing that deletion/substitution confusion remains a practical limitation.

## 7. Confidence Interpretation

Confidence is classifier confidence for the predicted class.

Confidence is not pronunciation correctness. It must not be presented as a pronunciation score. A final pronunciation score should be defined separately in the application layer, using model output as only one signal.

## 8. Limitations

- The dataset is imbalanced.
- `addition` still has few samples.
- Test addition support is small.
- The model can confuse `deletion` and `substitution`.
- The model is still a research/baseline classifier, not a full pronunciation assessment system.
- There is no production integration yet.
- Confidence is not scoring.

## 9. Next Phase

Recommended next phase: AI Worker integration and model serving preparation.

Recommended work:

- Prepare AI Worker integration.
- Define model packaging and checkpoint management.
- Define the inference input/output contract.
- Connect model output to the application feedback format.
- Later, improve with more data, better alignment, or fine-tuning.
