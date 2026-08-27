# R3-3 One-Time Locked TEST Evaluation

RESEARCH_ONLY — NOT_PRODUCTION — NO_0_TO_100_SCORE

The frozen artifacts were verified before TEST audio access. No training, threshold fitting,
checkpoint selection, preprocessing change, or TEST-specific adjustment occurred.

## Locked identity

- Checkpoint SHA-256: `5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E`
- V4 SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Threshold: `-1.293920` with `margin <= threshold` mapping to substitution
- Margin: `expected_logit - max(other 39 logits)`
- TEST rows: `28216`
- Inference seconds: `3.905`

## Acoustic observed-phone TEST metrics

- Top-1: `0.530798`
- Top-3: `0.816416`
- Macro-F1: `0.488234`
- Balanced accuracy: `0.534415`
- Macro precision: `0.480809`

## Primary frozen binary TEST metrics

- Accuracy: `0.762794`
- Balanced accuracy: `0.633143`
- Binary Macro-F1: `0.585048`
- Substitution precision: `0.236679`
- Substitution recall: `0.464015`
- Substitution F1: `0.313468`
- Confusion matrix: `[[19995, 4928], [1765, 1528]]`

Speaker assessment: **NO_CATASTROPHIC_SPEAKER_COLLAPSE**

TEST interpretation: **TEST_TRANSFER_CONFIRMED**

## Per-speaker binary metrics

| Speaker | Rows | Correct | Substitution | Macro-F1 | Sub P | Sub R | Sub F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASI | 4504 | 4053 | 451 | 0.506668 | 0.124494 | 0.272727 | 0.170952 |
| ERMS | 4722 | 4101 | 621 | 0.578048 | 0.243243 | 0.405797 | 0.304164 |
| SKA | 4783 | 4328 | 455 | 0.595836 | 0.228474 | 0.589011 | 0.329238 |
| THV | 4571 | 3698 | 873 | 0.684546 | 0.452166 | 0.573883 | 0.505805 |
| TXHC | 4770 | 4322 | 448 | 0.562914 | 0.188406 | 0.464286 | 0.268041 |
| YDCK | 4866 | 4421 | 445 | 0.548788 | 0.168099 | 0.395506 | 0.235925 |

## Frozen diagnostics

### Expected phones

- `TH`: `{"correct_support": 82, "substitution_support": 125, "correct_recall": 0.4268292682926829, "substitution_recall": 0.664, "substitution_precision": 0.6384615384615384, "median_margin_correct": -1.5604185163974762, "median_margin_substitution": -1.9379958510398865}`
- `DH`: `{"correct_support": 367, "substitution_support": 537, "correct_recall": 0.7656675749318801, "substitution_recall": 0.1824953445065177, "substitution_precision": 0.532608695652174, "median_margin_correct": 0.16187584400177002, "median_margin_substitution": 0.3475920185446739}`
- `R`: `{"correct_support": 924, "substitution_support": 50, "correct_recall": 0.775974025974026, "substitution_recall": 0.6, "substitution_precision": 0.12658227848101267, "median_margin_correct": -0.008638232946395874, "median_margin_substitution": -1.8514822721481323}`
- `V`: `{"correct_support": 444, "substitution_support": 121, "correct_recall": 0.8333333333333334, "substitution_recall": 0.45454545454545453, "substitution_precision": 0.4263565891472868, "median_margin_correct": 0.4715012460947037, "median_margin_substitution": -1.0290906876325607}`
- `D`: `{"correct_support": 1080, "substitution_support": 99, "correct_recall": 0.6796296296296296, "substitution_recall": 0.5858585858585859, "substitution_precision": 0.14356435643564355, "median_margin_correct": -0.7246993035078049, "median_margin_substitution": -1.548794835805893}`
- `T`: `{"correct_support": 1495, "substitution_support": 75, "correct_recall": 0.7036789297658863, "substitution_recall": 0.7866666666666666, "substitution_precision": 0.11752988047808766, "median_margin_correct": -0.5346143841743469, "median_margin_substitution": -2.426304578781128}`
- `S`: `{"correct_support": 1116, "substitution_support": 39, "correct_recall": 0.9560931899641577, "substitution_recall": 0.5897435897435898, "substitution_precision": 0.3194444444444444, "median_margin_correct": 1.4041368663311005, "median_margin_substitution": -1.8413102626800537}`
- `Z`: `{"correct_support": 477, "substitution_support": 448, "correct_recall": 0.8448637316561844, "substitution_recall": 0.45982142857142855, "substitution_precision": 0.7357142857142858, "median_margin_correct": 0.4327871799468994, "median_margin_substitution": -1.1396443247795105}`

### Predeclared substitution pairs

- `TH->T`: `{"support": 70, "detected_substitutions": 44, "detection_recall": 0.6285714285714286, "false_negatives": 26, "median_margin": -1.632900446653366}`
- `DH->D`: `{"support": 452, "detected_substitutions": 49, "detection_recall": 0.1084070796460177, "false_negatives": 403, "median_margin": 0.5062505602836609, "known_validation_reference": {"support": 300, "detection_recall": 0.21666666666666667, "false_negatives": 235, "median_margin": 0.051110975444316864}, "descriptive_classification": "DH_D_TEST_BEHAVIOR_WORSE_THAN_VALIDATION"}`
- `R->L`: `{"support": 6, "detected_substitutions": 6, "detection_recall": 1.0, "false_negatives": 0, "median_margin": -2.888643566519022}`
- `V->W`: `{"support": 24, "detected_substitutions": 14, "detection_recall": 0.5833333333333334, "false_negatives": 10, "median_margin": -2.099978029727936}`
- `Z->S`: `{"support": 430, "detected_substitutions": 192, "detection_recall": 0.44651162790697674, "false_negatives": 238, "median_margin": -1.1078359335660934}`

### Margin diagnostics

- ROC-AUC (substitution positive): `0.704610`
- PR-AUC (substitution positive): `0.254069`
- Correct-origin: `{"mean": 0.12716468488341282, "median": 0.2103954404592514, "p10": -2.203889012336731, "p25": -0.9689726531505585, "p75": 1.2859587967395782, "p90": 2.399079948663711}`
- Substitution-origin: `{"mean": -1.4631307720528526, "median": -1.1201216578483582, "p10": -4.319837975502014, "p25": -2.571274548768997, "p75": 0.1437678337097168, "p90": 1.015530547499657}`

No mapping from margin to a 0–100 score was created.
