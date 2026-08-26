# Phoenix correctness research — R2-B audio + phone

Status: `R2B_VALIDATION_FAIL`  
TEST opened: **NO**  
TEST eligible: **NO**

This experiment is **RESEARCH_ONLY**, **NOT_USED_BY_RUNTIME**, and
**NOT_PRODUCTION_MODEL**. It did not modify V2/V3 dataset semantics, R2-A,
runtime code, the AI Worker, or legacy checkpoints.

## Locked experiment

- V3 SHA-256: `433F006AB0ABCE47955C2305FCD131F2FFD9741417891BE125798163ADD28F7E`
- TRAIN: 59,572 rows; VALIDATION: 29,767 rows.
- Input: R2-A log-mel audio plus canonical expected-phone ID only.
- Phone vocabulary: `<UNK>, AA, AE, AH, AO, AW, AY, B, CH, D, DH, EH, ER,
  EY, F, G, HH, IH, IY, JH, K, L, M, N, NG, OW, OY, P, R, S, SH, T, TH,
  UH, UW, V, W, Y, Z, ZH` (IDs 0 through 39).
- Vowel stress is removed before lookup. `AX`, malformed `d/l`, and unseen
  phones map to `<UNK>`. `Embedding(40,16,padding_idx=0)` keeps ID 0 fixed zero.
- Architecture: unchanged 96-d CNN-attention audio branch; 16-d phone
  embedding; concatenate to 112-d; `Dropout(0.2) + Linear(112,2)`.
- Random initialization; no pretrained checkpoint and no hidden MLP.
- Seed 42; batch 8; Adam 1e-4; weight decay 0; 12 epochs; class-weighted
  cross entropy; weights 0.579494163 / 3.644884973; no sampler, focal loss,
  augmentation, or early stopping.

## FULL validation by epoch

| Epoch | Loss | Accuracy | Macro-F1 | BalAcc | Incorrect P/R/F1 | Sub R | Del R |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | .629714 | .824772 | .600595 | .588918 | .374750/.252016/.301366 | .302726 | .071502 |
| 2 | .638101 | .752276 | .592843 | .616197 | .282055/.421819/.338061 | .446485 | .334014 |
| 3 | .608632 | .789633 | .623511 | .636603 | .337432/.418011/.373424 | .441894 | .332993 |
| 4 | .612588 | .799879 | .620072 | .624181 | .345285/.373208/.358704 | .405739 | .257406 |
| 5 | .612124 | .805657 | .621890 | .622875 | .354867/.361783/.358292 | .391105 | .257406 |
| 6 | .605002 | .792623 | .627760 | .640760 | .344438/.423835/.380034 | .440172 | .365679 |
| 7 | .601801 | .784694 | .632067 | .654638 | .341380/.468862/.395092 | .479484 | .431052 |
| 8 | .617633 | .818927 | .623535 | .616935 | .379990/.328405/.352319 | .357245 | .225741 |
| 9 | .600005 | .814190 | .632985 | .632045 | .378391/.371864/.375099 | .394835 | .290092 |
| 10 | .600432 | .769040 | .626675 | .660374 | .325820/.505152/.396135 | .517647 | .460674 |
| 11 | .596584 | .794101 | .635196 | .651223 | .352837/.447133/.394427 | .461980 | .394280 |
| 12 | .594859 | .803474 | .641128 | .652308 | .368800/.436380/.399754 | .446198 | .401430 |

Epoch 12 was selected solely by highest FULL validation Macro-F1.

## Selected-checkpoint ablations

| Mode | Macro-F1 | BalAcc | Incorrect P/R/F1 | Sub R | Del R | Confusion matrix |
|---|---:|---:|---|---:|---:|---|
| FULL | .641128 | .652308 | .368800/.436380/.399754 | .446198 | .401430 | `[[21969,3334],[2516,1948]]` |
| NO-PHONE | .531489 | .533802 | .200157/.229167/.213681 | .198852 | .337079 | `[[21215,4088],[3441,1023]]` |
| NO-AUDIO | .615989 | .671548 | .304658/.574373/.398137 | .597991 | .490296 | `[[19451,5852],[1900,2564]]` |

FULL minus NO-AUDIO was only +0.025139 Macro-F1 and +0.001617 incorrect
F1, below both modality-use thresholds. The learned joint model therefore
relied strongly on the phone condition and did not demonstrate the required
incremental use of audio.

FULL improved over fixed phone-only R2-B0 by +0.042849 Macro-F1, +0.071447
balanced accuracy, and +0.117364 incorrect F1. It failed the +0.05 Macro-F1
improvement requirement as well as absolute Macro-F1 0.65 and incorrect-F1
0.40 requirements.

## Phone and duration diagnostics

For both FULL and NO-AUDIO, DH and Z were predicted incorrect for every row:
correct recall 0 and incorrect recall 1. M, F, and OY were predicted correct
for every row: correct recall 1 and incorrect recall 0. Validation supports
were DH 505/410, Z 493/488, M 874/8, F 631/18, and OY 50/1
(correct/incorrect). The mean per-phone Macro-F1 over the 36 predeclared
sufficiently supported phones was 0.484408 for FULL, 0.504406 for NO-PHONE,
and 0.389670 for NO-AUDIO.

Duration matching retained 4,434 pairs / 8,868 rows. Matched Macro-F1 was
0.634629 and matched incorrect F1 was 0.556512. Full-to-matched drops were
+0.006499 Macro-F1 and -0.156758 incorrect F1, so the duration gate passed.

Failure precedence makes the final result `R2B_VALIDATION_FAIL`. Although the
modality-use gate also failed, `R2B_PHONE_SHORTCUT_FAIL` applies only after the
primary gate passes. TEST was never decoded, materialized, or evaluated.

Artifacts are under `ai-training/experiments/r2b_audio_phone_seed42/`; the
selected checkpoint is diagnostic research material only.
