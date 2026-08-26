# R5-1B Frozen VALIDATION Transfer Result

Status: `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED`

## Identity

- R5-1B contract: `BD169175C0777B1C37506E95350CB6AD90A992045527396DDE5DA4419779AC4A`
- R5-1A execution manifest: `C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6`
- V4: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Checkpoint: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`

## Population

- Source/runtime words: 8,494/7,955
- Positive/negative runtime words: 227/7,728
- Source/runtime addition events: 296/245

## Alignability

- KEEP impossible: 0
- INSERT total/alignable/impossible: 1,412,080/1,411,982/98

## Continuous transfer

- Addition vs non-addition ROC-AUC: 0.751280885
- Addition vs correct-only ROC-AUC: 0.771707231
- Deltas vs TRAIN: -0.022202418, -0.030645590

## Fixed threshold

- ROBUST_THETA: `0.74858840306599927`
- TP/FP/FN/TN: 66/641/161/7087
- Binary Macro-F1: 0.543887636
- Addition P/R/F1: 0.093352192/0.290748899/0.141327623

## Event localization

- Exact-event P/R/F1: 0.025459689/0.073469388/0.037815126
- Multiple-addition words remain included; one BEST_INSERT cannot recover every event.

## Frozen gates

- G1: **PASS** (0.751280884888 >= 0.7)
- G2: **PASS** (0.771707230923 >= 0.7)
- G3: **FAIL** (0.543887636349 > 0.548179)
- G4: **PASS** (0.141327623126 > 0.129246)
- G5: **FAIL** (0.0832481079975 <= 0.054352)
- G6: **PASS** (0.0378151260504 > 0.026688)

Passed: 4/6

This is iterative development-validation transfer evidence, not independent final confirmation.
