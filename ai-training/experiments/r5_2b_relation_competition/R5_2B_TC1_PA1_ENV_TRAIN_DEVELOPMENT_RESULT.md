# R5-2B Environment-Corrected Frozen TRAIN Development Result

Status: `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED`

## Population

- Words: 16,582 (323 positive; 16,259 negative)
- Source/runtime addition events: 423/342

## Candidate audit

- KEEP impossible: 0
- Candidates KEEP/INSERT/SUB/DELETE: 16,582/2,977,040/2,255,916/57,844
- Impossible KEEP/INSERT/SUB/DELETE: 0/196/0/0
- Empty DELETE / one-phone words: 710/710

## Continuous scoring

- Addition vs non-addition AUC: 0.687841380
- Addition vs correct-only AUC: 0.669239360
- Addition vs substitution-only AUC: 0.705114529
- Addition vs deletion-only AUC: 0.784931840

## LOSO decision

- Thresholds: 0.29666223128636671, 0.19463801383972168, 0.21131936709086097, 0.19463801383972168, 0.19463801383972168, 0.19463801383972168, 0.22895944913228361, 0.19463801383972168, 0.19463801383972168, 0.21131936709086097, 0.19463801383972168, 0.28233603636423754
- TP/FP/FN/TN: 39/541/284/15718
- Binary Macro-F1: 0.530403032
- Addition P/R/F1: 0.067241379/0.120743034/0.086378738

## False-addition mechanism

- Correct-only FAR: 0.043230174
- Substitution-negative FAR: 0.017123288
- Deletion-negative FAR: 0.006414825

## Event localization

- Exact-event P/R/F1: 0.025862069/0.043859649/0.032537961
- Multiple-addition words remain included; one BEST_INSERT cannot recover every event.

## Frozen gates

- G1: **FAIL** (0.687841380349 >= 0.7)
- G2: **FAIL** (0.669239360205 >= 0.7)
- G3: **FAIL** (0.530403032327 > 0.55519787679)
- G4: **FAIL** (0.0863787375415 > 0.137998056365)
- G5: **FAIL** (0.0432301740812 <= 0.034912959381)
- G6: **PASS** (0.0171232876712 < 0.0501611603546)
- G7: **PASS** (0.0064148253742 < 0.0334996436208)
- G8: **FAIL** (0.0325379609544 >= 0.043893129771)

Passed: 2/8

Robust threshold: `R5_2_ROBUST_THETA_NOT_AUTHORIZED`
