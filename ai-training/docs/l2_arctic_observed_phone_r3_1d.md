# R3-1D Observed-Phone Final Budget Extension

Status: **RESEARCH_ONLY**, **NOT_PRODUCTION**, **NOT_RUNTIME_CONNECTED**.

Final validation status: `R3_1D_PASS_VALIDATION`. Trend decision:
`TRAINING_BUDGET_STILL_LIMITING`. `TEST_ELIGIBLE=YES`, but TEST remained
closed: no TEST audio-path resolution, feature extraction, inference, or
prediction export.

## Controlled change and reproduction

R3-1D tested one change only: `max_epochs` increased from 36 to 48. The run
started from fresh random initialization with seed 42 and did not load the
R3-1C checkpoint. V4 dataset/SHA, PHONE_IDENTIFICATION_ELIGIBLE rows,
40-class vocabulary, S1 split, audio-only 0.50-second crop, log-mel settings,
CNN-attention architecture, class-weighted cross entropy and exact weights,
batch size, optimizer, learning rate, and checkpoint-selection rule remained
fixed.

Before epoch 37, all eight registered validation metrics for epochs 1-36 were
compared with the saved R3-1C trajectory. Reproduction passed exactly, with
maximum absolute delta zero for every registered metric.

## New-budget trajectory

| Epoch | Val loss | Top-1 | Macro-F1 | Balanced | Top-3 | Correct Top-1 / MF1 | Substitution Top-1 / MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.4598 | .5089 | .4806 | .5358 | .8080 | .5218 / .5062 | .3960 / .3646 |
| 37 | 1.4646 | .5195 | .4840 | .5267 | .8170 | .5332 / .5126 | .4001 / .3477 |
| 38 | 1.4560 | .5269 | .4900 | .5366 | .8182 | .5389 / .5142 | .4221 / .3674 |
| 39 | 1.4529 | .5170 | .4817 | .5365 | .8110 | .5318 / .5080 | .3884 / .3449 |
| 40 | 1.4677 | .5118 | .4812 | .5267 | .8110 | .5233 / .5064 | .4118 / .3537 |
| 41 | 1.4533 | .5141 | .4772 | .5287 | .8136 | .5261 / .5017 | .4098 / .3598 |
| 42 | 1.4387 | .5208 | .4861 | .5414 | .8162 | .5329 / .5102 | .4156 / .3681 |
| 43 | 1.4413 | .5134 | .4874 | .5366 | .8179 | .5305 / .5179 | .3651 / .3412 |
| 44 | 1.4394 | .5231 | .4905 | .5398 | .8168 | .5365 / .5160 | .4063 / .3657 |
| 45 | 1.4265 | .5237 | .4940 | .5433 | .8196 | .5376 / .5199 | .4025 / .3552 |
| 46 | 1.4309 | .5275 | .4913 | .5423 | .8242 | .5444 / .5213 | .3805 / .3444 |
| 47 | 1.4255 | .5283 | .4978 | .5435 | .8235 | .5424 / .5260 | .4056 / .3566 |
| 48 | 1.4331 | .5275 | .4902 | .5441 | .8220 | .5412 / .5152 | .4087 / .3667 |

Epoch 47 was selected by validation Macro-F1. Relative to R3-1C epoch 36,
selected validation loss improved by 0.034373, Top-1 by 0.019424, Macro-F1 by
0.017221, balanced accuracy by 0.007705, and Top-3 by 0.015454.
Substitution-origin Macro-F1 decreased by 0.008021.

The exact practical-plateau conjunction was false because overall Macro-F1
and Top-1 gains exceeded its ceilings. The selected epoch was 47, overall
Macro-F1 gain exceeded 0.010, validation loss decreased meaningfully, and the
registered sustained-overfitting rule was false. Therefore the required trend
decision is `TRAINING_BUDGET_STILL_LIMITING`. No R3-1E or 60-epoch run was
started.

## Selected validation diagnostics

Selected metrics were loss 1.425461, Top-1 0.528286, Macro-F1 0.497823,
balanced accuracy 0.543520, macro precision 0.488679, and Top-3 0.823479.
Correct-origin Top-1/Macro-F1 were 0.542386/0.525994. Substitution-origin
Top-1/supported-class Macro-F1 were 0.405638/0.356586.

All 37 hard-supported classes reached recall at least 0.10, and no class with
validation support at least 200 had zero recall. AX remained the only
zero-recall class. All six validation speakers passed; Macro-F1 ranged from
0.448772 (SVBI) to 0.533302 (ABA), with median 0.500687.

The downstream binary diagnostic, unused for selection, produced Macro-F1
0.483013 and substitution precision/recall/F1
0.165477/0.789275/0.273594.
