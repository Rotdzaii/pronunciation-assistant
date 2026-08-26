# Phoenix R4-4C0 CTC failure-mode audit

RESEARCH_ONLY / NOT_PRODUCTION / R4_TEST_CLOSED

- Frozen artifact verification: PASS
- Selected checkpoint: epoch 35, `R4_4B_ctc_phone_sequence_seed42_best_validation_per.pt`
- Validation decode reproduction: PASS (7,728 words)
- Mean per-word decoded/manual-target length ratio: 0.822278
- Aggregate decoded/manual-target length ratio: 0.775676
- Median blank-argmax occupancy: 0.837838
- PER composition: substitution 56.316%, deletion 40.448%, insertion 3.236%
- CTC target-phone deletion errors / real manual expected deletions: 6,199 / 914 = 6.782x
- Clean all-correct word PER: 0.595460
- TRAIN-vs-VALIDATION: NO_MAJOR_GENERALIZATION_GAP
- Primary failure: **CTC_MIXED_ACOUSTIC_FAILURE**
- Architecture implication: **BIGRU_NEXT_EXPERIMENT_JUSTIFIED**
- Training occurred: NO
- R4 TEST accessed: NO

The selected model under-generates and also exhibits broad identity confusion. CTC/PER deletion errors are omissions from the manual observed sequence and must not be interpreted as real pronunciation deletions.
