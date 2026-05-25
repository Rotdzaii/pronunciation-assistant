# L2-ARCTIC Phone Error Type Classifier V2

## Why V2 Was Needed

The baseline phone-level error type classifier predicts three labels: addition,
deletion, and substitution. Its validation macro F1 was 0.5104 and test macro F1
was 0.4657, while addition F1 was 0.0000. That means the model effectively
ignored the smallest class.

## Class Imbalance

The Vietnamese L2-ARCTIC phone-error dataset is heavily imbalanced:

| split | addition | deletion | substitution |
| --- | ---: | ---: | ---: |
| train | 293 | 2541 | 5301 |
| val | 58 | 354 | 609 |
| test | 38 | 222 | 396 |

Substitution dominates the training set, and addition has too few examples for a
standard shuffled cross-entropy baseline to learn reliably.

## Methods Used

V2 keeps the same segment-level phone error classification task and still uses
mel-spectrogram inputs from the existing `start_time` and `end_time` metadata.

- Weighted loss: class weights are computed from the training split and applied
  to cross entropy.
- Weighted sampler: the training loader uses `WeightedRandomSampler` so minority
  classes appear more often during optimization.
- Focal loss: enabled by default and applied on top of weighted cross entropy to
  focus learning on harder examples.
- Training-only augmentation: random gain, small Gaussian noise, and optional
  small time shift are applied only for the training split.
- Model selection: checkpoints are selected by validation macro F1, not accuracy.

## Results Before vs After

Results are written by:

```powershell
ai-training\.venv\Scripts\python.exe ai-training\scripts\evaluate_l2_arctic_error_type_cnn_v2.py
ai-training\.venv\Scripts\python.exe ai-training\scripts\compare_error_type_v2.py
```

| metric | baseline | V2 | delta |
| --- | ---: | ---: | ---: |
| validation accuracy | 0.7571 | 0.6660 | -0.0911 |
| validation macro F1 | 0.5104 | 0.5369 | +0.0265 |
| test accuracy | 0.6966 | 0.6143 | -0.0823 |
| test macro F1 | 0.4657 | 0.4835 | +0.0178 |
| validation addition F1 | 0.0000 | 0.1765 | +0.1765 |
| test addition F1 | 0.0000 | 0.1240 | +0.1240 |

## Did Addition Improve?

Yes. Addition F1 improved from 0.0000 to 0.1765 on validation and from
0.0000 to 0.1240 on test. The model now predicts addition for some true
addition segments instead of ignoring the class.

## Accuracy vs Macro F1 Trade-Off

The V2 objective intentionally prioritizes macro F1 and minority-class recall
over raw accuracy. That trade-off occurred: validation accuracy decreased from
0.7571 to 0.6660 and test accuracy decreased from 0.6966 to 0.6143, while
validation macro F1 increased from 0.5104 to 0.5369 and test macro F1 increased
from 0.4657 to 0.4835.

## Limitations

- Addition remains data-limited, especially in validation and test.
- Segment-level labels do not prove overall pronunciation correctness.
- The model predicts error type only; it does not produce learner-facing
  pronunciation feedback by itself.
- Augmentation is lightweight and may not cover all speaker or recording
  variability.

## Next Step

Review addition false positives and false negatives, then consider targeted data
augmentation or a feature representation that includes phone identity context.
