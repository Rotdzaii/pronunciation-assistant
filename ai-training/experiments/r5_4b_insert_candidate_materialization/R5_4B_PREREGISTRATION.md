# R5-4B Preregistration

R5-4A was blocked because the frozen R5-1A/R5-2B outputs retained only BEST_INSERT and aggregate counts. R5-4B therefore preregisters one truth-blind TRAIN-only diagnostic execution that reproduces the frozen R5-2B acoustic and exact-CTC INSERT scoring path and serializes all 2,977,040 candidate records.

The execution is valid only when the exact 16,582-word population, per-word candidate identities, global alignability counts, frozen BEST_INSERT identities, exact winning scores, and all artifact hashes reproduce. All M1-M10 gates are required.

R5-4B does not evaluate Addition recoverability or performance. Truth labels do not enter candidate enumeration, scoring, ordering, persistence, or gating. Candidate artifacts must be frozen before any later truth join. No model training, scorer change, threshold search, classifier fitting, VALIDATION access, or TEST access is authorized.

Contract status upon successful freeze: `R5_4B_MATERIALIZATION_CONTRACT_FROZEN`.
