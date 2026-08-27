# R5-2B Relation-Competitive Exact CTC TRAIN Development Preregistration

## 1. Stage boundary

R5-2B is a contract and preregistration stage only. It freezes one new relation-competitive exact CTC development experiment before implementation, synthetic static verification, or real TRAIN inference.

Contract status:

`R5_2B_DEVELOPMENT_CONTRACT_FROZEN`

No scorer was implemented. No checkpoint was loaded. No TRAIN, VALIDATION, or TEST audio was accessed. No performance metric or threshold was calculated.

R5-1 remains closed as `R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED`. R5-2B is a new generation and does not alter R5-1 predictions, metrics, threshold, interpretation, closure, or documentation.

## 2. Verified provenance

| Artifact | SHA-256 |
|---|---|
| R5-2A artifact manifest | `F8E476F8FFA6AB2CD0F833B17D667D0D9B6CE3FC6B9FA1A7F5C1A3D71BA753E1` |
| V4 metadata | `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D` |
| Frozen R4-4C2 checkpoint | `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085` |
| Frozen R5-1A scorer | `4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90` |
| R5-1 closure manifest | `C8E71EDE56902D594A60F1194ABF8A72AB0A7EFBE4F212F57BC71FCEF21B1D69` |
| R5 documentation manifest | `1E9AE8829F7C3DAAB25FD8E1FBFB30EE6BFE3746E9A16DFF2379CD39C074D813` |

Every R5-2A manifested artifact was re-read and verified before this contract was created.

## 3. Scientific hypothesis

R5-1 compared `BEST_INSERT` only with `KEEP`. Frozen R5-1B/R5-1C evidence showed that useful addition ranking transferred, but the fixed threshold produced excessive false additions. A substitution or deletion can make KEEP acoustically weak, allowing an INSERT sequence to appear preferable even when that INSERT does not represent the true relation.

R5-2B tests one new hypothesis:

> An INSERT counts as addition evidence only to the extent that it outperforms the strongest permitted KEEP, single-SUBSTITUTION, or single-DELETE non-addition explanation in the same TARGET-normalized, alignability-safe exact CTC likelihood space.

This mechanism specifically attempts to reduce false additions in substitution- and deletion-containing negatives. It is not claimed to work before evaluation and is not claimed to eliminate speaker-dependent calibration shift.

## 4. Expected sequence and vocabulary

Let:

```text
E = [e1, e2, ..., eN]
N >= 1
```

Canonical phone indices are `0..39`. CTC blank is index `40`.

## 5. Frozen hypothesis families

### 5.1 KEEP

Exactly one target:

```text
H_KEEP = E
```

Candidate count: `1`.

### 5.2 INSERT

For `b = 0..N` and `p = 0..39`:

```text
H_INSERT(E,b,p) = E[:b] + [p] + E[b:]
```

Candidate count: `40(N+1)`.

BEST_INSERT tie order:

1. lower insertion boundary;
2. lower canonical phone index.

### 5.3 SUBSTITUTION

For `i = 0..N-1` and each `p != E[i]`, replace `E[i]` with `p`.

Candidate count: `39N`.

BEST_SUB tie order:

1. lower expected-phone position;
2. lower replacement canonical phone index.

No R3 probability, decision, margin, or threshold enters the score.

### 5.4 DELETE

For `i = 0..N-1`:

```text
H_DELETE(E,i) = E with E[i] removed
```

Candidate count: `N`.

BEST_DELETE ties use the lower deleted expected-phone position. When adjacent identical expected phones generate identical target sequences, the lower position wins for metadata.

### 5.5 Candidate-count identity

```text
KEEP + INSERT + SUB + DELETE
= 1 + 40(N+1) + 39N + N
= 80N + 41
```

This identity is frozen.

## 6. CTC alignability and scoring

### 6.1 Nonempty targets

For nonempty target `H`:

```text
ADJACENT_REPEAT_COUNT(H)
  = number of adjacent equal canonical labels

MIN_CTC_STEPS(H)
  = len(H) + ADJACENT_REPEAT_COUNT(H)

ALIGNABLE iff T >= MIN_CTC_STEPS(H)
```

An impossible target receives:

```text
RAW_SCORE(H) = -infinity
TARGET_SCORE(H) = -infinity
```

The scorer must not interpret a `zero_infinity=True` zero loss as an impossible target's mathematical score.

For an alignable nonempty target:

```text
RAW_SCORE(H) =
  -CTCLoss(
      log_softmax(logits, dim=-1),
      H,
      input_lengths=[T],
      target_lengths=[len(H)],
      blank=40,
      reduction="none",
      zero_infinity=True
  )

TARGET_SCORE(H) = RAW_SCORE(H) / max(len(H), 1)
```

An unexpected non-finite result for an alignable target is a fatal technical stop. No clipping or finite sentinel is permitted.

### 6.2 Empty DELETE target

For a one-phone expected sequence, DELETE produces `[]`.

Under standard CTC collapse semantics, only the all-blank alignment produces an empty output. Therefore:

```text
RAW_SCORE([])
  = sum over t=1..T of log_softmax(logits[t], dim=-1)[40]

TARGET_SCORE([])
  = RAW_SCORE([]) / max(0,1)
  = RAW_SCORE([])
```

The empty target is mathematically alignable for nonnegative `T`; for `T=0`, the empty sum is zero.

This is compatible with frozen R4 provenance. R4's numerical contract allowed exact `target_length=0` CTC behavior, and its recorded real one-phone DELETE check returned finite NLL `6.206483840942383` over eight encoder steps. The explicit blank-sum equation is the mathematical expansion of that same empty-output convention, not a new penalty or alternate score family.

Future synthetic static verification must independently compare the explicit blank-log-probability sum with an empty-target CTC calculation. One-phone words must not be removed, treated as KEEP, or assigned a penalty.

## 7. BEST_NON_ADDITION and relation-competitive score

Define:

```text
BEST_NON_ADDITION_SCORE = max(
    TARGET_SCORE(H_KEEP),
    TARGET_SCORE(BEST_SUB),
    TARGET_SCORE(BEST_DELETE)
)
```

KEEP is required to be available for every valid frozen-population word.

For diagnostic winning-family metadata only, exact family tie priority is:

1. KEEP;
2. SUB;
3. DELETE.

Within SUB and DELETE, the internal tie rules above apply. Family priority does not change the numeric maximum.

Freeze exactly one R5-2 score:

```text
C = TARGET_SCORE(BEST_INSERT) - BEST_NON_ADDITION_SCORE
```

Higher `C` means an INSERT is preferred more strongly than every permitted non-addition explanation.

No KEEP-only branch, ratio, relation softmax, RAW/TIME score, learned normalization, relation weight, or insertion penalty is allowed in R5-2B.

## 8. Binary and event output

Future decision rule:

```text
C >= theta  -> ADDITION
C < theta   -> NON_ADDITION
```

A threshold-positive word uses BEST_INSERT's canonical phone and expected-sequence boundary as its predicted event. Relation competition changes only the binary evidence margin; it does not redefine insertion localization.

When NON_ADDITION wins, the winning KEEP/SUB/DELETE family may be recorded as a diagnostic. It is not a confirmed multiclass correct/substitution/deletion prediction.

## 9. Frozen population

R5-2B must reproduce the exact R5-1A TRAIN population:

| Population item | Count |
|---|---:|
| Runtime-evaluable words | 16,582 |
| Positive words | 323 |
| Negative words | 16,259 |
| Clean source addition events | 423 |
| Runtime addition events | 342 |
| Multiple-addition words | 19 |
| Mixed substitution/addition words | 117 |
| Mixed deletion/addition words | 26 |

Mixed-error, multiple-addition, and one-phone words remain. R5-2-specific exclusions are forbidden. Failure to reproduce the frozen population requires a stop before metric interpretation.

## 10. TRAIN-only speaker LOSO

TRAIN speakers:

`BWC, EBVS, HJK, NCC, NJS, PNV, RRBI, TLV, TNI, YBAA, YKWK, ZHAA`

Use exactly 12 folds. For held-out speaker `S`, calibrate using the other 11 TRAIN speakers and apply the selected threshold once to `S`. Concatenate the 12 held-out prediction sets once.

Threshold candidates are:

- `np.nextafter` below the minimum unique finite calibration score;
- every unique finite float64 calibration `C` score;
- `np.nextafter` above the maximum unique finite calibration score.

`-infinity` is not a threshold candidate. Prediction equality is ADDITION.

Threshold-selection tie order remains:

1. higher Binary Macro-F1;
2. higher addition F1;
3. lower correct-only false-addition rate;
4. higher threshold.

Substitution/deletion FAR must not enter threshold selection.

## 11. Metrics

Continuous metrics:

- addition vs all non-addition ROC-AUC;
- addition vs correct-only ROC-AUC;
- descriptive addition vs substitution-only ROC-AUC;
- descriptive addition vs deletion-only ROC-AUC;
- score distributions by TRAIN speaker and frozen relation cohort.

Use the already static-verified deterministic extended-real ROC-AUC convention where applicable.

OOF binary metrics:

- TP, FP, FN, TN;
- Accuracy and Balanced Accuracy;
- Binary Macro-F1;
- Addition Precision, Recall, and F1;
- correct-only, substitution-containing, and deletion-containing negative FAR.

Exact event match requires the same word, canonical added phone, and expected-sequence boundary. Report event precision, recall, F1, and BEFORE_FIRST/BETWEEN/AFTER_FINAL breakdowns. One BEST_INSERT maximum per word remains a structural limitation.

## 12. Frozen R5-1A comparator

The following TRAIN values are historical comparators and must not be recomputed:

| Metric | R5-1A value |
|---|---:|
| Addition vs non-addition ROC-AUC | 0.7734833025081417 |
| Addition vs correct-only ROC-AUC | 0.8023528214095370 |
| Binary Macro-F1 | 0.5551978767901391 |
| Addition F1 | 0.1379980563654033 |
| Correct-only FAR | 0.03491295938104449 |
| Substitution-negative FAR | 0.05016116035455278 |
| Deletion-negative FAR | 0.03349964362081254 |
| Exact-event F1 | 0.04389312977099236 |

## 13. Eight frozen development gates

All eight must pass using full-precision values:

1. **G1:** addition vs all non-addition ROC-AUC >= `0.70`.
2. **G2:** addition vs correct-only ROC-AUC >= `0.70`.
3. **G3:** OOF Binary Macro-F1 > `0.5551978767901391`.
4. **G4:** OOF Addition F1 > `0.1379980563654033`.
5. **G5:** correct-only FAR <= `0.03491295938104449`.
6. **G6:** substitution-negative FAR < `0.05016116035455278`.
7. **G7:** deletion-negative FAR < `0.03349964362081254`.
8. **G8:** exact-event F1 >= `0.04389312977099236`.

G6 and G7 are required because reducing substitution/deletion false additions is the preregistered causal mechanism. Aggregate improvement cannot confirm the mechanism if either confusion rate worsens.

If all eight pass, `R5_2_ROBUST_THETA` is the ordinary float64 median of the 12 LOSO fold thresholds. Preserve speaker order, sorted order, and the exact median. If any gate fails: `R5_2_ROBUST_THETA_NOT_AUTHORIZED`.

Future scientific status is exactly:

- all pass: `R5_2_RELATION_COMPETITION_DEVELOPMENT_PASS`;
- any valid gate fails: `R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED`.

## 14. Evaluation contamination and TEST policy

Current VALIDATION speakers ABA, HKK, HQTV, LXC, MBMPS, and SVBI were consumed in R5-1B. R5-2 is motivated by R5-1B and R5-1C, so this split is not untouched or independent confirmation.

R5-2B development must not resolve VALIDATION audio, run validation inference, or calculate new validation R5-2 scores. A later separately authorized protocol may decide whether the split has value as transparent iterative repair evidence.

TEST speakers ASI, ERMS, SKA, THV, TXHC, and YDCK remain untouched. No TEST audio path, inference, score, sample inspection, or performance is authorized. A TRAIN development pass does not automatically authorize TEST.

## 15. Required synthetic static verification

Contract freeze does not authorize TRAIN inference. A separate R5-2B static-verification stage using synthetic tensors only must first test:

- KEEP scoring;
- INSERT construction and count;
- SUB construction and count;
- DELETE construction and count;
- total `80N+41` identity;
- alignable and impossible targets;
- empty DELETE exact all-blank likelihood;
- one-phone word behavior;
- BEST_INSERT, BEST_SUB, and BEST_DELETE ties;
- BEST_NON_ADDITION family tie priority;
- exact `C` arithmetic;
- impossible candidates cannot win;
- extended-real serialization;
- unchanged threshold helper;
- deterministic repeated execution.

No such test was executed during contract creation.

## 16. Anti-drift and stop policy

R5-2B forbids training, fine-tuning, checkpoint changes, R3 classifier outputs, alternate normalization or margins, relation weights, penalties, validation formula choice, speaker calibration, arbitrary multi-edit hypotheses, beam search, language models, cohort filtering, gate changes after metrics, and TEST access.

Any meaningful scientific change requires a newly named cycle.

## 17. Contract-stage protocol audit

- Training: NO.
- Model inference: NO.
- TRAIN audio/rerun/scores: NO.
- Performance metrics: NO.
- VALIDATION access/rerun/scores: NO.
- Threshold search: NO.
- TEST audio/inference/performance: NO.
- R5-1 modification: NO.
- R4 modification/reopening: NO.
- R5-2 scorer implementation: NO.
- Synthetic static verification: NOT YET EXECUTED.

## 18. Next-stage boundary

The only next action authorized by this contract is a separate synthetic R5-2B static-verification stage. It must not process real TRAIN audio.
