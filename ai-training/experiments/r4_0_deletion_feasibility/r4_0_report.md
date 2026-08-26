# R4-0 Deletion Feasibility Audit

RESEARCH_ONLY — NO_TRAINING — R4_TEST_CLOSED

V4 contains 1,612 TRAIN, 979 VALIDATION, and 827 TEST-metadata deletion rows.
R4-0 used audio from TRAIN and VALIDATION only. R4 TEST paths were not resolved or
read.

## Finding

The verdict is `DELETION_SIGNAL_UNCERTAIN` and the gate is
`R4_0_DELETION_READY_WITH_WARNINGS`.

Deletion annotation duration is a high-severity shortcut: a TRAIN-fit duration
threshold of 39.766 ms obtains validation ROC-AUC 0.858851, Macro-F1 0.668146,
and deletion F1 0.364164. Central RMS and silence ratio are weak (ROC-AUC
0.521354 and 0.469275), so deletion is not simply low-energy silence.

The strict validation control matches speaker, canonical expected phone, and a
10 ms duration bin. It retains 782 deletion/non-deletion pairs across 33 phones
and all six validation speakers. Duration falls to balanced accuracy 0.499361;
central RMS reaches only 0.516624. This removes the known trivial cues but does
not by itself prove that useful acoustic absence evidence remains.

## Window

The recommended first window is 0.30 s: it covers 99.923% of deletion intervals,
requires edge padding for 4.863%, and contains median 0.26 s of context outside
the tagged deletion interval. A 1.00 s window contains median 0.96 s of non-target
context and pads 23.389% of deletion rows.

## Next experiment

R4-1 may run one expected-phone-conditioned compatibility experiment, not a blind
audio-only CNN. It must use a uniform 0.30 s tensor, TRAIN-only class-weighted
CrossEntropy, the unchanged S1 split, and mandatory FULL/NO-AUDIO/NO-PHONE
validation ablations. The strict matched subset is a hard diagnostic and R4 TEST
remains closed.

The detailed pre-registered gates are stored in `final_status.json`.

## Timing semantics

V4 reads only manual `annotation/*.TextGrid` phones `IntervalTier` data, not MFA
or `/textgrid/`. Every eligible deletion is a positive-duration interval labeled
`expected,sil,d` and mapped to observed `<SIL>`. These manual boundaries are not
available automatically at runtime; future compatibility requires an alignment
or sequence mechanism.
