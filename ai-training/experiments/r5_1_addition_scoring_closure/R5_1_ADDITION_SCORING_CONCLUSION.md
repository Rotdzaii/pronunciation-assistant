# R5-1 Addition Scoring Generation Conclusion

Status: `R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED`

## Scientific conclusion

R5-0 established that addition annotations and speaker-disjoint support were sufficient for research. R5-1 then stopped before scientific metrics because 196 INSERT hypotheses across 65 words were CTC-impossible under its frozen scoring behavior; this was a technical stop, not a performance failure. R5-1A introduced alignability-safe probability-zero semantics before performance, passed all six TRAIN development gates, and froze `ROBUST_THETA = 0.7485884030659993`.

The one R5-1B fixed-threshold VALIDATION transfer retained useful continuous ranking evidence: addition/non-addition ROC-AUC was `0.7512808848879525`, and addition/correct-only ROC-AUC was `0.7717072309229656`. However, Binary Macro-F1 was `0.5438876363494940`, and correct-only false-addition rate increased to `0.0832481079975455`. Gates G3 and G5 failed, so the authoritative result remains `R5_1B_VALIDATION_TRANSFER_NOT_CONFIRMED`.

The post-hoc R5-1C audit found that five of six VALIDATION speakers exceeded the historical correct-FAR reference, major cohort score medians shifted upward by approximately 0.13-0.15, and false positives occurred across correct-only, substitution, deletion, and mixed negative cohorts. The cautious exploratory interpretation is `MIXED_CALIBRATION_AND_CLASS_OVERLAP`.

The frozen CTC scorer retained meaningful addition-ranking evidence, but its score distributions shifted across speakers and addition/non-addition overlap remained substantial. Consequently, the TRAIN-derived global threshold did not transfer with adequate false-addition control.

## What remains scientifically useful

- Addition annotations are sufficiently supported across speakers.
- Frozen CTC can represent insertion hypotheses.
- Alignability-safe exact CTC scoring is technically valid.
- Continuous addition ranking was above chance on TRAIN and VALIDATION.
- Exact scoring improved over the frozen greedy development comparators.
- Exact-event evidence remained above the greedy comparator.
- R5-1A controlled correct-word false additions on TRAIN.
- The failure emerged during fixed-threshold transfer to VALIDATION.

These findings do not confirm an addition detector and do not establish production readiness.

## Evidence status

Current VALIDATION addition performance is consumed. Any future use must acknowledge that history and cannot describe this split as untouched independent confirmation for a scorer influenced by R5-1.

R5 TEST remains completely untouched. No TEST audio, inference, or performance was consumed. Because R5-1B failed its frozen transfer contract, the current generation is not eligible to open TEST.

## Limitations

- Severe class imbalance and low absolute precision.
- One BEST_INSERT per word structurally limits multiple-addition recall.
- MFA cannot create arbitrary added-phone slots.
- The checkpoint used current VALIDATION speakers for PER-based epoch selection.
- R5-1B is iterative rather than independent confirmation.
- R5-1C is post-hoc exploratory evidence.
- Speaker calibration variation and class overlap both remain.

## Claim policy

Allowed thesis-style claim:

> Exact CTC insertion-hypothesis scoring produced meaningful addition-ranking evidence and passed all preregistered TRAIN development gates. However, fixed-threshold transfer to the iterative VALIDATION split was not confirmed because false-addition rates increased substantially across unseen speakers.

Vietnamese defense explanation:

> Mô hình có học được tín hiệu của âm thêm và vẫn xếp hạng khá tốt khi đổi người nói. Tuy nhiên cùng một ngưỡng từ tập TRAIN lại báo nhầm addition quá nhiều trên VALIDATION. Kiểm tra sau đó cho thấy vừa có lệch điểm giữa người nói, vừa có sự chồng lấn giữa addition và các trường hợp không phải addition.

Claims that Phoenix accurately detects additions, that R5 addition detection was confirmed, or that calibration alone caused the failure are prohibited.

## Closure

The current R5-1 generation is closed. Its threshold, normalization, scorer, population, and model must not be retuned as a continuation. Future addition research remains possible only as a new named generation with a new hypothesis, preregistration, transparent VALIDATION-consumption statement, predefined numeric gates, and continued TEST preservation.
