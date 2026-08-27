# R5-1 Frozen CTC Addition Scoring Development Contract

Status: **R5_1_CONTRACT_FROZEN**  
Scope: contract and preregistration only; no scorer implementation, inference, training, VALIDATION access, or TEST access.

## Research hypothesis

R5-0 found directional but weak greedy CTC insertion behavior. R5-1 asks one question: can exact TARGET-normalized CTC likelihood distinguish a single explicit INSERT sequence from the canonical KEEP sequence more reliably than greedy `INSERT_IN_DECODED`?

The frozen R4-4C2 CNN+BiGRU checkpoint, vocabulary, blank handling, word audio span, features, downsampling, and exact CTC likelihood semantics remain unchanged.

## Source identities

- V4 SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- R4-4C2 checkpoint SHA-256: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`
- R5-0 preregistration SHA-256: `14CBFADAC0BE35D53C01DC966A030A71F24EC4D86EA88384BC449544CE12AEF7`
- R5-0 report SHA-256: `DF484A09333A2F53A05E98E1C66B3C330CBA35F029B04B0BBB1623A3A9BAA3BA`
- R5-0 final status SHA-256: `29F06679FD24F8A2BBBED25B9B57A693D15046444A97BF5D951D575E5850D0B6`
- R5-0 manifest SHA-256: `3A6F62C5BD354E5FD21149135ECC9FB4CD673947A0E5E213A513B4F2005339E8`

All identities matched before this contract was written.

## Exact hypotheses and score

For expected canonical sequence `E=[e1,...,eN]`, `N>=1`:

- `H_KEEP = E`
- legal boundary `b` is any integer from `0` through `N`
- added phone `p` is any of the frozen 40 canonical phone indices
- `H_INSERT(E,b,p) = E[:b] + [p] + E[b:]`

Each word has exactly `1 + (N+1)*40` named candidates before equivalent-sequence memoization.

The exact inherited R4 formula is:

```text
log_probs = log_softmax(logits, dim=-1)
RAW_SCORE(H) = -CTCLoss(
    log_probs,
    H,
    input_lengths=[T],
    target_lengths=[len(H)],
    blank=40,
    reduction="none",
    zero_infinity=True,
)
TARGET_SCORE(H) = RAW_SCORE(H) / max(len(H), 1)
```

Therefore:

```text
KEEP = RAW_SCORE(E) / N
INSERT(b,p) = RAW_SCORE(H_INSERT(E,b,p)) / (N+1)
BEST_INSERT = max over every legal (b,p) INSERT(b,p)
A = BEST_INSERT - KEEP
```

This is mathematically well-defined: KEEP length is `N>=1`, while every INSERT length is `N+1>=2`. No new normalization, length coefficient, duration term, phone prior, speaker prior, insertion penalty, RAW branch, or TIME branch is permitted.

The best-candidate tie order is higher TARGET score, lower insertion boundary, then lower canonical phone index, using exact float64 equality. Identical target sequences can arise from different boundaries inside repeated-phone runs; the implementation must report that equivalence and must not claim that CTC distinguished those boundaries.

Before scoring, every KEEP and INSERT candidate must pass the CTC minimum-step audit `len(H) + adjacent-identical-count <= T`. Any theoretically unalignable candidate or nonfinite score stops execution before calibration.

## TRAIN population

R5-1 reuses the frozen R5-0 runtime-evaluable TRAIN reconstruction:

- 16,582 eligible words
- 323 addition-positive words
- 16,259 addition-negative words
- 342 clean addition events inside eligible words
- 423 clean TRAIN source additions remain in total data accounting

The gap between 423 source events and 342 runtime-evaluable events must be reported by frozen exclusion reason. It must not be hidden. Mixed addition+substitution, addition+deletion, and multiple-addition words remain positive and are separately reported.

## Speaker-LOSO threshold protocol

Use exactly 12 folds. In each fold, one TRAIN speaker is held out and the other 11 speakers calibrate a single global threshold. `A >= theta` predicts addition.

The candidate thresholds are the lower `np.nextafter` edge, every sorted unique finite float64 calibration score, and the upper `np.nextafter` edge. Select by:

1. higher calibration Binary Macro-F1;
2. higher calibration addition F1;
3. lower calibration correct-only false-addition rate;
4. higher threshold.

All ties use exact float64 comparison. Freeze the held-out predictions before the next fold and concatenate the 12 held-out sets once for OOF metrics.

Only if all six development gates pass, define `ROBUST_THETA` as the ordinary float64 median of the 12 fold thresholds—the float64 arithmetic mean of the sixth and seventh sorted values. No weighting or post-hoc adjustment is allowed.

## Metrics

Continuous TRAIN metrics:

- addition vs all non-addition ROC-AUC;
- addition vs correct-only ROC-AUC;
- score distributions by speaker and by true addition, correct-only, substitution-negative, and deletion-negative cohorts;
- descriptive Pearson/Spearman relationships with expected length and word duration.

OOF word metrics:

- TP, FP, FN, TN, accuracy, balanced accuracy, Binary Macro-F1;
- addition precision, recall, and F1;
- false-addition rates on correct-only, substitution-containing negative, and deletion-containing negative words.

For a threshold-positive word, emit the BEST_INSERT phone and boundary. Exact event matching requires word identity, phone, and boundary, using deterministic one-to-one multiset matching. Multiple true additions remain separate; because the scorer emits at most one event per word, unmatched true events are false negatives. Report event precision/recall/F1 overall and for BEFORE_FIRST, BETWEEN, and AFTER_FINAL. Timestamps do not enter primary matching.

## Frozen development gates

All six must pass:

1. addition vs all non-addition ROC-AUC `>= 0.70`;
2. addition vs correct-only ROC-AUC `>= 0.70`;
3. OOF Binary Macro-F1 `> 0.548179`;
4. OOF addition F1 `> 0.129246`;
5. correct-only false-addition rate `<= 0.054352`;
6. exact-event F1 `> 0.026688`.

All pass: `R5_1_INSERTION_HYPOTHESIS_SCORING_DEVELOPMENT_PASS`. Any failure: `R5_1_INSERTION_HYPOTHESIS_SCORING_NOT_CONFIRMED`. Neither status authorizes TEST.

## Evaluation firewall

- TRAIN speaker-LOSO is development evidence only.
- Current VALIDATION is not independent confirmation because it selected R4-4C2 epoch 35 by PER. It may be used only by a later separately frozen iterative transfer evaluation. R5-1 must not resolve or read its audio or calculate its scores.
- TEST remains untouched and is only a candidate independent final confirmation split after development, scorer freeze, required iterative validation gates, and a dedicated locked TEST preregistration.

No model training, checkpoint selection, beam search, language model, second score family, learned calibration, insertion penalty, or phone/boundary/speaker-specific tuning is permitted.
