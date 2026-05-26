# L2-ARCTIC Phone Error Classifier - Sampler-Only Training

## Purpose

This experiment trains a phone-level English pronunciation error classifier for Vietnamese L2-ARCTIC speakers. The target classes are `addition`, `deletion`, and `substitution`.

## Why Sampler-Only Was Selected

The clean v2 dataset is strongly imbalanced, especially for `addition`. Prior balancing ablations showed that combining `WeightedRandomSampler` with weighted `CrossEntropyLoss` over-corrected the minority class and caused collapse in common classes.

The selected setup is:

```txt
WeightedRandomSampler only
+ normal CrossEntropyLoss
```

This keeps the training batches more balanced without also increasing the loss weight for the same minority samples.

## Ablation Conclusion

`WeightedRandomSampler + weighted CrossEntropyLoss` caused over-balancing and collapse. The sampler-only model improved overall macro F1 and avoided that collapse, while still leaving `addition` as the weakest class.

## Sampler-Only Result

Validation:

| Metric | Value |
|---|---:|
| Macro F1 | 0.5838 |

Test:

| Metric | Value |
|---|---:|
| Macro F1 | 0.4980 |
| Addition F1 | 0.1053 |
| Deletion F1 | 0.6555 |
| Substitution F1 | 0.7333 |

The sampler-only run improves overall macro F1 versus the previous baseline and V2 runs, mainly by avoiding the severe class collapse seen in over-balanced setups.

## Current Limitation

`addition` remains weak because the clean dataset has few examples. The clean v2 test split has only 19 addition samples, so addition metrics are noisy and hard to improve with the current three-way setup.

Model confidence is the model's confidence in the predicted class. It is not pronunciation correctness.

## Next Recommended Direction

Use a binary-stage classifier:

1. Stage 1: `addition` vs `non_addition`
2. Stage 2: `deletion` vs `substitution`

This separates the rare `addition` detection problem from the more stable deletion/substitution decision.
