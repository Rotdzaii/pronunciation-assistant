# R4-4C2 frozen CNN+BiGRU CTC experiment contract

RESEARCH_ONLY / NOT_PRODUCTION / R4_TEST_CLOSED / ONE TRAINING RUN

## Research question

R4-4C2 tests one hypothesis only: adding exactly one small bidirectional GRU after the unchanged R4-4B time-preserving CNN improves acoustic phone-sequence modeling enough to reduce broad phone confusion and CTC under-generation. No other architecture, data, loss, decoding, or training variable changes.

## Frozen sources and data

The V4 SHA is `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`. The R4-4B selected checkpoint SHA is `A154DFAC573D69B8ED1A71CBCDC23227EA3E80929890AD87E97ED85667142106`; it is a comparator only and is not loaded. The R4-4B artifact manifest SHA is `56A109E7D62A72CC27BB29B4C1DDF98E7DE8B095E232603534C0ABBA008EDC1B`, and the R4-4C0 audit manifest SHA is `11BF1244477BDA50E6F749FDB841AA4FC53952A3F672415321FD1F70B26A232D`.

Reuse the exact R4-4A/R4-4B word policy: 16,259 TRAIN words from BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA; 7,728 VALIDATION words from ABA, HKK, HQTV, LXC, MBMPS, SVBI. Addition-containing, unresolved, and malformed words remain excluded. The 28 TRAIN and 15 VALIDATION empty targets remain included. TEST speakers ASI, ERMS, SKA, THV, TXHC, YDCK remain closed.

For each expected position, correct emits the expected canonical phone, substitution emits the manually observed canonical phone, and deletion emits nothing. Expected phones and relation labels never enter the neural network. The vocabulary is the frozen 40-phone order AA through ZH at indices 0–39 plus `<CTC_BLANK>` at index 40.

## Features and model

Input is the full MFA word span, mono 16 kHz, without side context. Extract 64-bin log-mels with `n_fft=512`, 400-sample Hann window, 160-sample/10 ms hop, centered constant padding, power 2, Slaney scale and normalization, and per-word max-relative dB floored at -80 dB. No phone interval or duration feature is used.

The CNN is byte-for-byte structurally equivalent to R4-4B: Conv/BN/ReLU blocks with channels 1→16→32→64→96, 3×3 kernels and padding 1; pools `(2,2)` then `(2,1)`; Dropout2d 0.05 then 0.10. Only the first pool downsamples time, producing approximately 20 ms spacing. Mean only the frequency axis to obtain `[B,T,96]`.

Insert one `GRU(input_size=96, hidden_size=96, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)`. Pack using true encoder lengths and `enforce_sorted=False`, run the GRU, then pad with `batch_first=True` and the padded CNN time as `total_length`. Effective recurrent context must never include padding. The result is `[B,T,192]`, followed directly by `Dropout(0.20)` and `Linear(192,41)`. CTC receives raw logits. Exact trainable parameters: CNN 79,104; BiGRU 111,744; head 7,913; total 198,761.

All parameters use fresh seed-42 initialization. R3, R4-4B, and external weights are forbidden.

## Loss and training

Use `CTCLoss(blank=40, reduction="mean", zero_infinity=True)` only. Train exactly 36 epochs, seed 42, batch size 8, length-bucketed variable-length padded batches, Adam at `1e-4`, weight decay 0, gradient clipping norm 5.0. No scheduler, early stopping, augmentation, sampler, rebalancing, auxiliary loss, or second seed.

Checkpoint selection is lowest VALIDATION PER, then higher deletion F1, then earlier epoch. Train all 36 epochs.

## Decode and relation recovery

Greedy decode only: timestep argmax, collapse consecutive identical labels, remove blank. No beam, LM, lexicon, expected-phone constraint, blank adjustment, insertion bonus, or temperature. Recover relations with unit-cost expected-vs-decoded Levenshtein using deterministic priority MATCH > SUBSTITUTION > DELETE_FROM_EXPECTED > INSERT_IN_DECODED.

## Frozen metrics and gates

Report all 36 epochs: train/validation CTC loss, PER, exact sequence accuracy, binary Macro-F1, deletion P/R/F1, substitution false-deletion, and three-relation Macro-F1. At the selected checkpoint report S/D/I edits, binary and three-class confusion matrices, all 40 observed-target phone metrics, each validation speaker, and the frozen 717-pair/1,434-row matched control (`D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798`).

Confirmation requires all: PER ≤0.55; binary Macro-F1 ≥0.70; deletion recall ≥0.45; deletion F1 ≥0.40; substitution false-deletion ≤0.25; matched Macro-F1 ≥0.60; matched deletion F1 ≥0.55; every speaker with at least 30 deletions has recall ≥0.25; three-relation Macro-F1 ≥0.40.

R4-4B comparison values are PER 0.602840, binary Macro-F1 0.476941, deletion F1 0.111433, correct false-deletion 0.256118, and substitution false-deletion 0.280781.

## Under-generation audit and outcome rules

Reproduce manual and decoded phone totals, aggregate decoded/target ratio, fraction of shorter decodes, blank argmax occupancy, blank posterior mean/median, PER deletion edits, and clean-word false deletion. R4-4B references are ratio 0.7757, shorter 0.4885, blank occupancy 0.8277, blank posterior mean 0.8019, 6,199 deletion edits, and clean-word false deletion 0.2416.

Development improvement is numerical: PER improves by at least 0.02; binary Macro-F1 by at least 0.02; deletion F1 by at least 0.03. Under-generation improves if at least two occur: ratio +0.05, shorter fraction -0.05, blank occupancy -0.05, deletion edits -10%, clean-word false deletion -0.05.

Classification precedence is: `R4_4C2_BIGRU_CTC_CONFIRMED` when all confirmation gates pass; `R4_4C2_BIGRU_CTC_IMPROVED` when all three development deltas and under-generation improvement pass; `R4_4C2_BIGRU_CTC_MIXED` when at least one development dimension passes but the full improvement rule does not; otherwise `R4_4C2_BIGRU_CTC_NOT_IMPROVED`.

NOT_IMPROVED stops CTC architecture expansion. MIXED requires a failure audit before training. IMPROVED permits only a separate decision about one further cycle. CONFIRMED still requires a separate freeze and authorization before any R4 TEST access.

## Static guarantees

The unchanged ×2 temporal geometry retains zero theoretical CTC-unalignable words in both TRAIN and VALIDATION. Synthetic checks must verify `[B,T,96] → packed BiGRU → [B,T,192] → Linear → [B,T,41]`, correct packed lengths, padding isolation, and 198,761 parameters before training.

No training, validation model inference, or R4 TEST access is authorized by this document.
