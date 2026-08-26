# R3-1A Observed-Phone Acoustic Baseline

Status: **RESEARCH_ONLY**, **NOT_PRODUCTION**, **NOT_RUNTIME_CONNECTED**.

Final validation status: `R3_1A_VALIDATION_FAIL`. TEST remains closed and is
not eligible for evaluation.

## Locked experiment

- Dataset: `all_speakers_expected_observed_v4.csv`
- SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Eligible subset: `PHONE_IDENTIFICATION_ELIGIBLE` only
- Input: audio-derived log-mel only
- Target: `observed_phone_canonical`, 40 alphabetical ARPAbet classes
- Excluded: deletion, addition, unresolved, and non-speech
- Train: BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA
- Validation: ABA, HKK, HQTV, LXC, MBMPS, SVBI
- TEST metadata speakers: ASI, ERMS, SKA, THV, TXHC, YDCK
- Counts: train 57,559; validation 28,212; TEST metadata 28,216
- TEST audio/features/inference: not accessed

Vocabulary indexes are: `AA=0, AE=1, AH=2, AO=3, AW=4, AX=5, AY=6,
B=7, CH=8, D=9, DH=10, EH=11, ER=12, EY=13, F=14, G=15, HH=16,
IH=17, IY=18, JH=19, K=20, L=21, M=22, N=23, NG=24, OW=25,
OY=26, P=27, R=28, S=29, SH=30, T=31, TH=32, UH=33, UW=34,
V=35, W=36, Y=37, Z=38, ZH=39`.

## Preprocessing and model

Every row uses the same mono 16 kHz, 0.50-second crop centered at the manual
annotation interval. Available audio is clipped at utterance boundaries and
only the missing side is zero-padded. The resulting 8,000 samples are converted
to a fixed `[1,64,16]` log-mel representation using 64 mel bins, FFT/window
2,048, hop 512, Hann window, Slaney scale/normalization, and dB relative to the
per-sample maximum.

The randomly initialized `SmallPronunciationCNNAttention` uses the unchanged
16/32/64/96 CNN and temporal-attention backbone, followed by dropout 0.2 and
`Linear(96,40)`. Training used class-weighted CrossEntropy only, Adam at
`1e-4`, batch size 8, seed 42, and 12 epochs. No sampler, focal loss,
augmentation, or pretrained checkpoint was used.

## Validation epochs

| Epoch | Loss | Top-1 | Macro-F1 | Balanced | Top-3 | Correct top-1 | Substitution top-1 | Substitution MF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.5950 | 0.2562 | 0.1831 | 0.2213 | 0.5341 | 0.2574 | 0.2454 | 0.1468 |
| 2 | 2.3224 | 0.3100 | 0.2473 | 0.2854 | 0.5929 | 0.3152 | 0.2650 | 0.1913 |
| 3 | 2.1763 | 0.3405 | 0.2757 | 0.3154 | 0.6280 | 0.3440 | 0.3097 | 0.2201 |
| 4 | 2.1052 | 0.3621 | 0.3057 | 0.3423 | 0.6478 | 0.3665 | 0.3238 | 0.2421 |
| 5 | 2.0043 | 0.3743 | 0.3202 | 0.3637 | 0.6722 | 0.3779 | 0.3438 | 0.2525 |
| 6 | 1.9373 | 0.3937 | 0.3369 | 0.3809 | 0.6972 | 0.3990 | 0.3475 | 0.2673 |
| 7 | 1.8855 | 0.3961 | 0.3572 | 0.4039 | 0.6959 | 0.4014 | 0.3506 | 0.2731 |
| 8 | 1.8355 | 0.4216 | 0.3772 | 0.4139 | 0.7277 | 0.4273 | 0.3716 | 0.2850 |
| 9 | 1.8109 | 0.4289 | 0.3806 | 0.4192 | 0.7293 | 0.4348 | 0.3771 | 0.2966 |
| 10 | 1.7691 | 0.4199 | 0.3865 | 0.4389 | 0.7302 | 0.4288 | 0.3424 | 0.2858 |
| 11 | 1.7407 | 0.4379 | 0.3924 | 0.4358 | 0.7404 | 0.4445 | 0.3805 | 0.2946 |
| 12 | 1.7204 | 0.4472 | 0.4103 | 0.4495 | 0.7482 | 0.4561 | 0.3695 | 0.3054 |

Epoch 12 was selected by highest validation Macro-F1. Its macro precision was
0.4184. Correct-origin Macro-F1 was 0.4355; substitution-origin supported-class
Macro-F1 was 0.3054.

## Diagnostics and gate

Class coverage passed: all 37 hard-supported classes had recall at least 0.10,
and no class with validation support at least 200 had zero recall. AX was the
only zero-recall class (73 validation rows); OY recall was 0.3077 and ZH recall
was 0.2500.

All six speaker gates passed. Speaker Macro-F1 ranged from 0.3829 (SVBI) to
0.4268 (MBMPS), with median 0.4177.

The downstream expected-versus-predicted-observed diagnostic produced binary
Macro-F1 0.4317, substitution precision 0.1454, recall 0.8051, and F1 0.2464.
It did not affect training, selection, or the R3-1A gate.

Attention had shape `[28212,4]`, no NaN, mean normalized entropy 0.1399, mean
maximum weight 0.9220, and 22.01% near-one-hot rows. This is only a concentration
diagnostic, not an explanation.

The substitution, class-coverage, and speaker conditions passed. The overall
gate failed solely because selected top-1 accuracy was 0.4472, below the locked
0.50 threshold. No configuration was changed after observing the result, and
no additional experiment was run.
