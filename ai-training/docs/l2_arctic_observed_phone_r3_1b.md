# R3-1B Observed-Phone Extended-Training Feasibility

Status: **RESEARCH_ONLY**, **NOT_PRODUCTION**, **NOT_RUNTIME_CONNECTED**.

Final status: `R3_1B_PASS_VALIDATION`. `TEST_ELIGIBLE=YES`, but TEST remained
closed: no TEST audio path resolution, feature materialization, or inference.

## Hypothesis and isolation

R3-1B tested exactly one hypothesis: R3-1A was undertrained because its
12-epoch budget ended while validation was improving. It repeated R3-1A from
fresh random initialization with seed 42. Dataset, SHA, eligible rows,
vocabulary, S1 speaker split, 0.50-second crop, log-mel preprocessing,
CNN-attention architecture, class weights, optimizer, learning rate, batch
size, and all other settings remained identical. Only `max_epochs` changed
from 12 to 24. R3-1A weights were not loaded.

The saved R3-1A trajectory was compared before epoch 13. All eight registered
metrics at every epoch 1-12 reproduced exactly; maximum absolute delta was zero.

## Validation trajectory

| Epoch | Val loss | Top-1 | Macro-F1 | Balanced | Top-3 | Correct Top-1 / MF1 | Substitution Top-1 / MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.5950 | .2562 | .1831 | .2213 | .5341 | .2574 / .1917 | .2454 / .1468 |
| 2 | 2.3224 | .3100 | .2473 | .2854 | .5929 | .3152 / .2599 | .2650 / .1913 |
| 3 | 2.1763 | .3405 | .2757 | .3154 | .6280 | .3440 / .2890 | .3097 / .2201 |
| 4 | 2.1052 | .3621 | .3057 | .3423 | .6478 | .3665 / .3201 | .3238 / .2421 |
| 5 | 2.0043 | .3743 | .3202 | .3637 | .6722 | .3779 / .3359 | .3438 / .2525 |
| 6 | 1.9373 | .3937 | .3369 | .3809 | .6972 | .3990 / .3560 | .3475 / .2673 |
| 7 | 1.8855 | .3961 | .3572 | .4039 | .6959 | .4014 / .3784 | .3506 / .2731 |
| 8 | 1.8355 | .4216 | .3772 | .4139 | .7277 | .4273 / .4013 | .3716 / .2850 |
| 9 | 1.8109 | .4289 | .3806 | .4192 | .7293 | .4348 / .4038 | .3771 / .2966 |
| 10 | 1.7691 | .4199 | .3865 | .4389 | .7302 | .4288 / .4110 | .3424 / .2858 |
| 11 | 1.7407 | .4379 | .3924 | .4358 | .7404 | .4445 / .4166 | .3805 / .2946 |
| 12 | 1.7204 | .4472 | .4103 | .4495 | .7482 | .4561 / .4355 | .3695 / .3054 |
| 13 | 1.6695 | .4487 | .4158 | .4658 | .7554 | .4568 / .4411 | .3781 / .3118 |
| 14 | 1.6621 | .4681 | .4224 | .4627 | .7610 | .4771 / .4483 | .3898 / .3123 |
| 15 | 1.6659 | .4569 | .4160 | .4703 | .7572 | .4650 / .4399 | .3864 / .3189 |
| 16 | 1.6263 | .4688 | .4302 | .4706 | .7694 | .4788 / .4544 | .3816 / .3143 |
| 17 | 1.6129 | .4684 | .4393 | .4853 | .7737 | .4768 / .4589 | .3953 / .3425 |
| 18 | 1.5881 | .4799 | .4432 | .4918 | .7840 | .4893 / .4633 | .3988 / .3362 |
| 19 | 1.5680 | .4934 | .4579 | .5007 | .7954 | .5053 / .4804 | .3895 / .3461 |
| 20 | 1.5760 | .4768 | .4456 | .4977 | .7857 | .4901 / .4719 | .3613 / .3341 |
| 21 | 1.5618 | .4917 | .4555 | .4985 | .7910 | .5017 / .4814 | .4053 / .3430 |
| 22 | 1.5594 | .4855 | .4455 | .4993 | .7881 | .4970 / .4731 | .3857 / .3273 |
| 23 | 1.5356 | .4882 | .4533 | .5010 | .7932 | .4988 / .4788 | .3967 / .3328 |
| 24 | 1.5182 | .5018 | .4633 | .5094 | .8037 | .5135 / .4915 | .3994 / .3516 |

Epoch 24 was selected by validation Macro-F1 and passed all unchanged R3-1A
validation gates. From epoch 12 to 24, loss fell 0.2022, Top-1 rose 0.0546,
Macro-F1 rose 0.0529, and substitution-origin Top-1 rose 0.0299. The selected
epoch equals the maximum budget, so the experiment records
`TRAINING_BUDGET_STILL_LIMITING`; no longer run was started.

## Selected diagnostics

All 37 hard-supported classes reached recall at least 0.10 and no class with
validation support at least 200 had zero recall. AX remained the only
zero-recall class. OY recall was 0.5577 and ZH recall was 0.2292.

All validation speakers passed: Macro-F1 ranged from 0.3993 (SVBI) to 0.5057
(MBMPS), with median 0.4771.

The downstream binary diagnostic, unused for selection, produced Macro-F1
0.4677 and substitution precision/recall/F1 0.1598/0.8047/0.2666.
