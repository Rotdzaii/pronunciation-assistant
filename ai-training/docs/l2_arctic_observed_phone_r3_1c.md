# R3-1C Observed-Phone Training to 36 Epochs

Status: **RESEARCH_ONLY**, **NOT_PRODUCTION**, **NOT_RUNTIME_CONNECTED**.

Final status: `R3_1C_PASS_VALIDATION`. Trend: `CONTINUED_IMPROVEMENT` with
`TRAINING_BUDGET_STILL_LIMITING`. `TEST_ELIGIBLE=YES`, but TEST remained
closed: no TEST audio-path resolution, feature materialization, or inference.

## Hypothesis and isolation

R3-1C tested one change only: increasing `max_epochs` from 24 to 36. It
repeated R3-1B from fresh random initialization with seed 42 and did not load
the R3-1B checkpoint. Dataset and SHA, PHONE_IDENTIFICATION_ELIGIBLE rows,
40-class vocabulary, S1 speaker split, 0.50-second audio crop, log-mel
preprocessing, CNN-attention architecture, class-weighted cross entropy,
optimizer, learning rate, batch size, and all other settings remained fixed.

Before epoch 25, all eight registered validation metrics at every epoch 1-24
were compared with the saved R3-1B trajectory. Reproduction passed exactly;
the maximum absolute delta was zero for every registered metric.

## Validation trajectory from the prior budget boundary

| Epoch | Val loss | Top-1 | Macro-F1 | Balanced | Top-3 | Correct Top-1 / MF1 | Substitution Top-1 / MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1.5182 | .5018 | .4633 | .5094 | .8037 | .5135 / .4915 | .3994 / .3516 |
| 25 | 1.5150 | .4956 | .4622 | .5161 | .7997 | .5084 / .4887 | .3840 / .3421 |
| 26 | 1.5262 | .4946 | .4604 | .5067 | .7974 | .5054 / .4887 | .4012 / .3399 |
| 27 | 1.5071 | .4954 | .4650 | .5219 | .7980 | .5080 / .4904 | .3850 / .3368 |
| 28 | 1.5106 | .4984 | .4651 | .5171 | .7971 | .5104 / .4905 | .3943 / .3403 |
| 29 | 1.5021 | .4939 | .4644 | .5130 | .7988 | .5074 / .4923 | .3771 / .3419 |
| 30 | 1.4820 | .5137 | .4782 | .5194 | .8127 | .5275 / .5068 | .3936 / .3473 |
| 31 | 1.4788 | .5062 | .4733 | .5239 | .8104 | .5197 / .4991 | .3888 / .3525 |
| 32 | 1.4804 | .5028 | .4723 | .5265 | .8071 | .5159 / .4988 | .3881 / .3461 |
| 33 | 1.4720 | .5144 | .4806 | .5262 | .8126 | .5277 / .5084 | .3984 / .3458 |
| 34 | 1.4740 | .5041 | .4721 | .5276 | .8083 | .5154 / .4959 | .4060 / .3562 |
| 35 | 1.4676 | .5076 | .4757 | .5308 | .8095 | .5228 / .5025 | .3750 / .3456 |
| 36 | 1.4598 | .5089 | .4806 | .5358 | .8080 | .5218 / .5062 | .3960 / .3646 |

Epoch 36 was selected by validation Macro-F1. Relative to R3-1B epoch 24,
validation loss fell by 0.0583, Macro-F1 rose by 0.0173, Top-1 rose by
0.0071, and balanced accuracy rose by 0.0265. Epoch 36 was also the minimum
validation-loss epoch. The late improvement exceeded the pre-registered
threshold without triggering the overfitting rule, so the curve is classified
as `CONTINUED_IMPROVEMENT`. Because the selected epoch is at the new maximum
budget, `TRAINING_BUDGET_STILL_LIMITING` is recorded; no further run was
started.

## Selected validation diagnostics

Selected overall metrics: loss 1.459835, Top-1 0.508861, Macro-F1 0.480603,
balanced accuracy 0.535815, macro precision 0.469370, and Top-3 0.808025.
Correct-origin Top-1/Macro-F1 were 0.521835/0.506190. Substitution-origin
Top-1/supported-class Macro-F1 were 0.396012/0.364606.

All 37 hard-supported classes reached recall at least 0.10, and no class with
validation support at least 200 had zero recall. AX remained the only
zero-recall class. All six validation speakers passed; speaker Macro-F1 ranged
from 0.416108 (SVBI) to 0.523255 (ABA), with median 0.488706.

The downstream binary diagnostic, unused for checkpoint selection or gates,
produced Macro-F1 0.468682 and substitution precision/recall/F1
0.157979/0.780337/0.262762.
