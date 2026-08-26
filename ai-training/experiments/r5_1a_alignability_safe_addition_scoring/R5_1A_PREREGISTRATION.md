# R5-1A — Alignability-Safe Exact CTC Addition Scoring

Status: `R5_1A_CONTRACT_FROZEN`

R5-1A is a new, pre-metric contract. It does not modify R5-1, whose attempted execution remains preserved as `R5_1_EXECUTION_TECHNICAL_FAILURE_CTC_ALIGNABILITY`. R5-1 produced no model scores, LOSO thresholds, performance metrics, scientific classification, or evaluated gates.

## Hypothesis and frozen model

The hypothesis remains that exact CTC sequence likelihood can provide better addition evidence than greedy `INSERT_IN_DECODED`. The acoustic model remains the frozen R4-4C2 CNN+BiGRU CTC checkpoint with the exact frozen preprocessing, 40 canonical phones at indices 0–39, and blank index 40. No training, fine-tuning, checkpoint selection, architecture change, beam search, or language model is permitted.

For expected sequence `E=[e1,...,eN]`, `N>=1`:

- `H_KEEP = E`
- `H_INSERT(E,b,p) = E[:b] + [p] + E[b:]`
- `b in {0,...,N}`
- `p in {0,...,39}`

All `40*(N+1)` INSERT identities remain in hypothesis accounting.

## Exact CTC alignability

For target `H=[h1,...,hU]`:

```text
ADJACENT_REPEAT_COUNT(H) = count(i in 2..U where h_i == h_(i-1))
MIN_CTC_STEPS(H) = len(H) + ADJACENT_REPEAT_COUNT(H)
ALIGNABLE(H,T) iff T >= MIN_CTC_STEPS(H)
```

The extra steps are required because adjacent identical target labels need an intervening CTC blank.

Every KEEP target must be alignable. If any KEEP target is impossible, future execution stops as `R5_1A_EXECUTION_BLOCKED_KEEP_ALIGNABILITY`; no word may be excluded or redefined.

## Extended-real score semantics

An impossible target has exact probability zero:

```text
P(H|X) = 0
log P(H|X) = -Infinity
RAW_SCORE(H) = -Infinity
TARGET_SCORE(H) = -Infinity
```

The zero returned by `CTCLoss(zero_infinity=True)` for an impossible target must never be interpreted as a score.

For alignable targets only:

```text
log_probs = log_softmax(logits, dim=-1)
raw_nll(H) = CTCLoss(
    log_probs,
    H,
    input_lengths=[T],
    target_lengths=[len(H)],
    blank=40,
    reduction="none",
    zero_infinity=True,
)
RAW_SCORE(H) = -raw_nll(H)
TARGET_SCORE(H) = RAW_SCORE(H) / max(len(H), 1)
```

If an alignable target nevertheless returns a non-finite loss, execution stops. No clipping, epsilon, finite sentinel, or penalty is permitted.

## BEST_INSERT and addition score

BEST_INSERT is the maximum TARGET score among alignable INSERT candidates. Exact ties prefer lower boundary and then lower canonical phone index. Every output preserves total, alignable, and impossible candidate counts.

If all INSERT candidates are impossible:

- `best_insert_exists=false`
- `BEST_INSERT_SCORE=-Infinity`
- `A=-Infinity`
- phone and boundary are null
- predicted insertion event is none
- the word remains in binary evaluation

Otherwise:

```text
A = TARGET_SCORE(BEST_INSERT) - TARGET_SCORE(KEEP)
```

## Frozen population

Population membership is unchanged:

- 16,582 runtime-evaluable TRAIN words
- 323 positive words
- 16,259 negative words
- 423 clean source addition events
- 342 runtime events
- 19 multiple-addition words
- 117 mixed substitution/addition words
- 26 mixed deletion/addition words

Impossible INSERT candidates do not exclude words.

## TRAIN speaker-LOSO development

A future separately authorized execution uses the same 12 TRAIN speakers and 12 speaker-held-out folds. Each fold calibrates on the other 11 speakers. Threshold candidates are the two `np.nextafter` edges around the finite score range plus every unique finite float64 calibration score. `-Infinity` is not a threshold candidate and remains negative under every finite threshold.

Selection precedence remains:

1. higher Binary Macro-F1;
2. higher addition F1;
3. lower correct-only false-addition rate;
4. higher threshold.

`A>=theta` predicts addition. ROBUST_THETA is authorized only if all six gates pass and equals the ordinary float64 median of the 12 fold thresholds.

## Extended-real ROC-AUC

Negative infinity is a legitimate lowest score, tied with every other negative-infinity score. ROC-AUC must be calculated directly by exact extended-real ordering using the Mann–Whitney rank formula with average ranks for exact ties:

```text
AUC = (sum_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
```

JSON serializes negative infinity as the string `"-Infinity"`, restored to IEEE-754 negative infinity before ranking. No finite replacement is permitted. If an implementation cannot preserve ordering and ties exactly, it stops as `R5_1A_EXECUTION_BLOCKED_NONFINITE_METRIC_SUPPORT` before metrics.

## Unchanged gates

- G1: addition vs all non-addition ROC-AUC `>= 0.70`
- G2: addition vs correct-only ROC-AUC `>= 0.70`
- G3: OOF Binary Macro-F1 `> 0.548179`
- G4: OOF addition F1 `> 0.129246`
- G5: correct-only false-addition rate `<= 0.054352`
- G6: exact-event F1 `> 0.026688`

## Event and audit policy

An exact event requires the same word, canonical added phone, and expected-sequence insertion boundary. There is at most one predicted event per word. Multiple-addition words remain included but cannot have every event recovered by this single-event scorer.

Before any future performance interpretation, report KEEP impossibility; total, alignable, and impossible INSERT counts; affected/all-impossible words; and counts by speaker, class, expected length, and adjacent-repeat cause. The audit cannot change scoring.

## Held-out policy

VALIDATION and TEST paths may not be resolved, their audio may not be read, and inference, scores, predictions, and performance may not be produced during R5-1A development. This contract task performs no inference, training, thresholding, or metrics.
