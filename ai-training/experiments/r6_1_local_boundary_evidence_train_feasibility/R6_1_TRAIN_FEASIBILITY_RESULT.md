# R6-1 Frozen TRAIN Local Boundary Evidence Feasibility

Final status: `R6_1_LOCAL_BOUNDARY_EVIDENCE_NOT_CONFIRMED`

## Identity and environment

All frozen identities passed. Interpreter: C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\.venv\Scripts\python.exe
Python 3.12.10; torch 2.12.0.dev20260408+cu128; CUDA 12.8; GPU NVIDIA GeForce RTX 5060 Laptop GPU.

## Structural accounting

Words=16582; boundaries=74426; positive=324; negative=74102; events=342.

## Coverage

Events covered=308/342 (0.9005847953216374).

## Primary score

Primary score: `MEAN_UNEXPECTED_PHONE_MASS`.
Pooled boundary ROC-AUC: 0.5265224849834894.

## Speaker ROC-AUC

- BWC: 0.42906832298136643
- EBVS: 0.5284156563257023
- HJK: 0.722483660130719
- NCC: 0.626666396804836
- NJS: 0.46962396069538925
- PNV: 0.4946772812478304
- RRBI: 0.2955908798550858
- TLV: 0.5926649838429278
- TNI: 0.22901710835290173
- YBAA: 0.678290717341558
- YKWK: 0.5654390135794012
- ZHAA: 0.709814845158119

Median: 0.5469273349525517; count > 0.55: 6/12.

## Position diagnostics

- BEFORE_FIRST: positive=40, negative=15896, AUC=0.18940928535480628, positive median=0.0410033242436402, negative median=0.34407813471233073
- BETWEEN: positive=161, negative=40961, AUC=0.5843141809941618, positive median=0.08072141082380418, negative median=0.028077658322735056
- AFTER_FINAL: positive=92, negative=15186, AUC=0.653246840625519, positive median=0.23474521823896274, negative median=0.11313893702170041

## Mixed and multiple diagnostics

- single_addition_words: boundaries=276, events=276, median=0.10904628402609581
- multiple_addition_words: boundaries=17, events=32, median=0.056566380928789003
- mixed_substitution_addition: boundaries=105, events=110, median=0.06495789097662229
- mixed_deletion_addition: boundaries=23, events=25, median=0.06207653101910201

## Descriptive controls (non-gating)

Peak unexpected posterior positive/negative medians: 0.21516814827919006 / 0.1366606503725052.
Mean nonblank mass positive/negative medians: 0.21412167884409428 / 0.26510092850367073.

## Frozen gates

- F1: FAIL (0.9005847953216374 >= 0.99)
- F2: FAIL (0.5265224849834894 >= 0.65)
- F3: FAIL (0.5469273349525517 >= 0.6)
- F4: FAIL (6 >= 9)

Result: 0/4.

## Protocol

Training=False; checkpoint inference=True; classifier fitting=False; threshold search=False; VALIDATION=False; TEST=False; execution count=1.
