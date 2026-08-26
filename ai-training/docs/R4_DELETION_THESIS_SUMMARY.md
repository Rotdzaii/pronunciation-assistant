# R4 Pronunciation Deletion Research — Thesis Summary

## Research aim

The Phoenix R4 research cycle investigated whether an expected English phone omitted by a learner could be detected from word audio without relying on a manually annotated deletion interval. This was treated as a separate problem from R3, where correct/substitution research had already been confirmed on its locked TEST split.

R4 used speaker-disjoint L2-ARCTIC TRAIN and VALIDATION sets. The six-speaker R4 TEST split was reserved as an independent holdout and was never accessed because no R4 development candidate passed the final confirmation contract.

## Method development

The research proceeded through several deliberately frozen hypotheses. A duration-only analysis initially achieved Binary Macro-F1 0.668146 and deletion F1 0.364164, but it depended on a manual deletion-slot duration that is unavailable at runtime. A first learned classifier and expected-phone mismatch scoring did not outperform this diagnostic baseline or distinguish deletion reliably from substitution.

Temporal analysis found some localization evidence, while an MFA-based missing-phone anchor was rejected because approximately 99% of manually annotated deletions still retained the expected MFA phone label. The research therefore moved from fixed phone slots to word-level sequence modeling.

A deterministic expected-versus-observed sequence formulation was highly representable under an oracle, but frozen R3 sliding evidence plus dynamic programming produced deletion F1 0.025465. TRAIN-only prior rescaling could not rescue this family.

Phoenix then trained a self-contained word-level CTC phone recognizer. Its targets emitted the expected phone for a correct realization, the observed phone for a substitution, and no phone for a deletion. The first CNN-only model learned phone sequences but under-generated heavily: PER was 0.602840, 48.85% of words decoded shorter than their targets, and 6,199 target phones were omitted by the recognizer. These acoustic omissions were frequently misinterpreted as pronunciation deletions.

Adding one bidirectional GRU substantially improved sequence modeling. PER fell to 0.453330, the decoded/target ratio increased to 0.924911, and CTC target-phone deletion errors fell to 2,769. Nevertheless, greedy decoded omissions still produced weak deletion decisions, with deletion F1 0.185464.

The final decision layer therefore used exact CTC sequence likelihood rather than greedy omission. For each expected-phone position, it compared full-word KEEP, local DELETE, and 39 local SUBSTITUTION hypotheses. TRAIN-only analysis showed strong continuous discrimination. A final 12-fold speaker leave-one-speaker-out audit selected TARGET-length normalization and froze the median threshold at `0.16184102947061696`.

## Final validation

The final R4-4D2B method combined the frozen CNN+BiGRU CTC model, TARGET-normalized CTC hypothesis scoring, and the TRAIN-speaker-LOSO threshold.

| Metric | Final value |
|---|---:|
| Binary Macro-F1 | 0.652102 |
| Balanced Accuracy | 0.670750 |
| Deletion precision | 0.298077 |
| Deletion recall | 0.373085 |
| Deletion F1 | 0.331390 |
| Correct false-deletion rate | 0.031548 |
| Substitution false-deletion rate | 0.031907 |
| Three-relation Macro-F1 | 0.465742 |
| Matched-control Macro-F1 | 0.593491 |
| Matched-control deletion F1 | 0.485769 |
| Deletion vs non-deletion ROC-AUC | 0.854124 |
| Deletion vs substitution ROC-AUC | 0.862234 |

Only 3 of 8 preregistered gates passed. Binary Macro-F1, deletion recall, deletion F1, matched-control Macro-F1, and matched-control deletion F1 remained below their required values. The final method also remained below the non-deployable duration-only comparator on Binary Macro-F1 and deletion F1.

## Interpretation

Phoenix demonstrated measurable acoustic evidence for pronunciation deletion through a self-trained CNN+BiGRU CTC sequence model and CTC hypothesis scoring. However, the final frozen validation protocol did not meet the predefined deletion confirmation criteria. Therefore deletion detection is retained as a current research limitation and was not evaluated on the untouched final R4 TEST split.

The result should be described as **MEASURABLE DELETION-RELATED SIGNAL** but **INSUFFICIENT ROBUST DELETION DECISION PERFORMANCE**. It must not be presented as a validated or production-ready deletion classifier.

### Plain-language defense explanation

> The model learned meaningful acoustic evidence indicating whether an expected phone may be missing. However, the deletion score distribution changed between speakers. A threshold calibrated on the training speakers therefore missed too many real deletions on validation speakers. The system could rank likely deletion cases reasonably well, but it could not convert that signal into sufficiently stable deletion decisions.

### Giải thích ngắn bằng tiếng Việt

> Mô hình có nhận ra tín hiệu của âm bị bỏ, nhưng ngưỡng để quyết định một âm thật sự bị deletion chưa ổn định khi đổi sang người nói mới. Vì vậy hệ thống còn bỏ sót nhiều deletion thật và chưa đạt các tiêu chí validation đã đặt trước.

## Research conclusion and scope

Final status: **R4_DELETION_RESEARCH_CLOSED_NOT_CONFIRMED**.

The R4 branch is closed. No additional R4 threshold, normalization, model architecture, or validation iteration is authorized. Any future deletion investigation must begin as a new research generation with a new hypothesis and an independent evaluation protocol. R3 correct/substitution findings remain separate and TEST-confirmed; R5 addition feasibility is the next planned research stage, but it is not implemented here.
