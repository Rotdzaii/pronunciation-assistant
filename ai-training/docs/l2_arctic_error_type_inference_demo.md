# L2-ARCTIC Phone Error Type Inference Demo

## Purpose

This demo loads the V2 phone-level error type classifier checkpoint and predicts
one of three diagnostic labels for an audio segment: addition, deletion, or
substitution.

The script is for local model inspection only. It does not generate learner
feedback and it does not decide whether a full pronunciation attempt is correct.

## Training vs Evaluation vs Inference

- Training updates model weights from labeled examples and saves a checkpoint.
- Evaluation measures a saved checkpoint on validation and test splits with
  aggregate metrics such as accuracy, macro F1, per-class F1, and confusion
  matrices.
- Inference loads a saved checkpoint and predicts the error type for one segment
  or a small demo subset.

## Row-Index Inference

Use this mode when the segment is already listed in the L2-ARCTIC classification
metadata:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\infer_l2_arctic_error_type_v2.py --row-index 0
```

The script reads:

```text
ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv
```

It prints metadata fields, the ground-truth error type, the predicted error type,
model confidence, class probabilities, and whether the prediction matches the
ground truth.

## Audio-Path Inference

Use this mode for a specific audio file and optional segment times:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\infer_l2_arctic_error_type_v2.py --audio-path path\to\audio.wav --start-time 1.2 --end-time 1.8
```

If `start-time` and `end-time` are provided, the script crops that segment before
building the log-mel spectrogram. If times are omitted, it uses the beginning of
the file and pads or truncates to the model input length.

## Batch Demo

The batch demo samples a small number of rows from each error class and writes:

```text
ai-training/datasets/l2-arctic/evaluation/v2_inference_demo_predictions.csv
```

Run it with:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\demo_l2_arctic_error_type_predictions.py
```

## Confidence

Confidence is the model probability for the predicted class. It is not a measure
of pronunciation correctness, and it should not be presented to learners as a
score. A high-confidence prediction can still be wrong.

## Current Limitations

- The classifier predicts only addition, deletion, or substitution for a segment.
- It assumes that segment boundaries are already known.
- It does not produce user-facing pronunciation feedback.
- It does not combine phone-level predictions into an utterance-level diagnosis.
- It depends on the local V2 checkpoint:
  `ai-training/models/l2_arctic_error_type_cnn_v2.pt`.

## Next Step

Connect this inference logic to the AI worker or FastAPI later, after the
segment-boundary source and response contract are defined.
