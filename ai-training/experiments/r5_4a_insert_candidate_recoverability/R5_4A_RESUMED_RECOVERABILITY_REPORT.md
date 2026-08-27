# R5-4A Resumed INSERT Candidate Recoverability Audit

Final status: `R5_4A_INSERT_CANDIDATE_RECOVERABILITY_NOT_CONFIRMED`

## Truth mapping

- Addition-positive words/events: 323 / 342
- Single-addition exactly mappable: 304 / 304
- Multiple-addition words/events: 19 / 38

## Single-addition recoverability

- Top-1 / Top-3 / Top-5 / Top-10: 0.134868421 / 0.240131579 / 0.279605263 / 0.388157895
- Mean / median truth rank: 57.101974 / 19.000000
- Q25 / Q75 / Q90 / max: 4.000000 / 75.750000 / 172.700000 / 364.000000
- MRR: 0.219815891

## Incremental gains and gaps

- Top-1 -> Top-3: 0.105263158
- Top-3 -> Top-5: 0.039473684
- Top-5 -> Top-10: 0.108552632
- Score-gap mean / median: 1.029737303 / 0.787745791

## Frozen feasibility gates

- G1: PASS (1 >= 0.99)
- G2: FAIL (0.279605263158 >= 0.4)
- G3: FAIL (0.388157894737 >= 0.55)
- G4: FAIL (19 <= 10.0)

Passed: 1 / 4

## Interpretation

- Q1: NO: most exact truth candidates are ranked below the frozen top-ten region.
- Q2: YES: Top-5 gains 0.144737 and Top-10 gains 0.253289 absolute recall over Top-1.
- Q3: NO: the correct event is usually too far down the INSERT family for top-1 selection alone to explain the low event F1.
- Q4: Speaker recoverability varies by 0.789474 Top-10 recall; weakest is ZHAA and strongest is TNI. The variation is substantial.
- Q5: Weakest position by Top-10 is BEFORE_FIRST. The weakest phones with N>=5 are AX, G, K, EH, W; rare-phone counts are preserved without interpretation.
- Q6: NO: the frozen recoverability gates do not all pass, so this audit does not justify that next generation.

No checkpoint, audio, new candidate score, classifier, threshold search, word-level performance metric, VALIDATION, or TEST data was used.
