# R5-2A Competitive Relation-Scoring Feasibility Audit

## 1. Scope and scientific boundary

R5-2A is a new R5 generation motivated by the frozen R5-1B transfer failure and R5-1C post-hoc audit. It does not repair, reopen, or reinterpret R5-1. The authoritative prior status remains:

`R5_1_ADDITION_SCORING_GENERATION_CLOSED_NOT_CONFIRMED`

This audit used source-code and frozen-artifact inspection only. It ran no training, model inference, audio processing, threshold search, validation optimization, or performance calculation. R5 TEST remained untouched.

All required identities passed, including V4, the frozen R4-4C2 checkpoint, the R5-1A scorer, the R5-1 closure manifest, and the R5 documentation manifest.

## 2. Frozen motivation

R5-1B retained useful continuous ranking on iterative VALIDATION:

- addition vs non-addition ROC-AUC: 0.7512808848879525;
- addition vs correct-only ROC-AUC: 0.7717072309229656.

The frozen global threshold nevertheless produced Binary Macro-F1 0.5438876363494940, addition F1 0.1413276231263383, and correct-only FAR 0.0832481079975455. G3 and G5 failed.

R5-1C found false positives across several non-addition cohorts: 407 correct-only, 167 substitution-only, 44 deletion-only, and 23 substitution-plus-deletion words. It also found broad speaker score-location movement and substantial class overlap. The frozen exploratory interpretation was **MIXED_CALIBRATION_AND_CLASS_OVERLAP**.

This evidence motivates, but does not validate, the question of whether an INSERT hypothesis should compete against explicit SUBSTITUTION and DELETE explanations rather than KEEP alone.

## 3. Mechanistic hypothesis

R5-1 used:

```text
BEST_INSERT versus KEEP
```

R5-2A audits the feasibility of a future comparison conceptually shaped as:

```text
BEST_INSERT versus BEST_NON_ADDITION
BEST_NON_ADDITION drawn from KEEP / SUBSTITUTION / DELETE
```

No final mathematical margin, numeric criterion, threshold, or cross-family tie precedence is frozen in R5-2A.

For expected sequence `E = [e1, ..., eN]`:

- `KEEP` is the unchanged `E`;
- `INSERT(E,b,p)` inserts phone `p` at boundary `b` in `0..N`;
- `SUB(E,i,q)` replaces `E[i]` with canonical phone `q != E[i]`;
- `DELETE(E,i)` removes `E[i]`.

### Why this can target class overlap

A true substitution can make KEEP acoustically poor. In that situation an INSERT hypothesis may beat KEEP despite replacement being the better sequence explanation. An explicit BEST_SUB can raise the non-addition reference when substitution is acoustically preferred.

A true deletion can similarly make KEEP poor. BEST_DELETE supplies an omitted-phone explanation instead of forcing every non-addition word to be represented by KEEP.

The mechanism is less direct for correct-only words because KEEP already competed in R5-1. SUB/DELETE can help only where model confusion gives an edited sequence a stronger score than KEEP. Correct-only words were the largest false-positive cohort, so relation competition cannot be assumed to solve the dominant error source.

Class-overlap assessment: **MECHANISTICALLY_JUSTIFIED_BUT_NOT_GUARANTEED**.

### Whether this targets speaker shift

Every candidate is evaluated on the same word, posterior tensor, speaker, and TARGET normalization. Common within-word score components may therefore cancel. However, R5-1 was already a same-word INSERT-minus-KEEP comparison and still showed speaker movement. Relative preferences among relation families can also shift.

Calibration-shift assessment: **PARTIALLY_TARGETS**. R5-2A does not claim speaker shift will cancel.

## 4. R3, R4, and R5 mechanism reuse

### R3

R3 used a separately trained local 40-class CNN-attention phone classifier and separately calibrated posterior/logit margins. Its decisions should not be combined with R5 CTC outputs because they occupy different model, temporal, and calibration spaces.

R5-2 may reuse only the canonical substitution concept: replace one expected phone with another canonical phone. Exact substitution sequences can be scored directly under the frozen CTC posterior, keeping all relation competitors in one coherent likelihood space.

Reuse classification: **REUSE_CANONICAL_SUBSTITUTION_CONCEPT_ONLY**.

### R4

The inspected R4-4D code constructs:

- KEEP as the full expected sequence;
- one DELETE target per expected position;
- 39 SUB targets per expected position;
- TARGET score as `RAW_SCORE(H) / max(len(H), 1)`;
- BEST_SUB ties by lowest canonical phone index in the frozen per-position comparison.

R4's scientific deletion result remains closed and is not reused as evidence that deletion detection works. Its deterministic sequence constructors, vocabulary, batching structure, and score definition can inform a new implementation without modifying R4 artifacts.

The original R4 scorer counted CTC minimum steps but used `zero_infinity=True`. New R5-2 code must apply R5-1A safe impossible-target semantics before interpreting loss.

### R5-1A

Reusable technical mechanisms include minimum-step alignability, `-infinity` for impossible targets, non-finite guards, TARGET normalization, BEST_INSERT tie behavior, extended-real serialization/AUC support, finite threshold candidates, and the batched CTCLoss adapter pattern.

The frozen scorer itself must not be modified. R5-1A's threshold, predictions, and R5-1B metrics cannot become R5-2 parameters or formula-selection evidence.

## 5. Hypothesis-space validity

For 40 canonical phones and expected length `N`:

| Family | Count | Target length |
|---|---:|---:|
| KEEP | `1` | `N` |
| INSERT | `40(N+1)` | `N+1` |
| SUBSTITUTION | `39N` | `N` |
| DELETE | `N` | `N-1` |
| All non-addition candidates | `1+40N` | mixed |
| Full relation competition | `80N+41` | mixed |

R5-1 scored `40N+41` KEEP/INSERT candidates. Relation competition adds `40N` SUB/DELETE candidates. Total hypothesis scoring remains linear in word-phone length with vocabulary size fixed at 40.

Every family uses the same CTC rule:

```text
MIN_CTC_STEPS(H) = len(H) + ADJACENT_REPEAT_COUNT(H)
alignable iff T >= MIN_CTC_STEPS(H)
impossible target score = -infinity
```

Adjacent identical phones can change minimum steps after insertion, replacement, or deletion; therefore alignability must be checked per constructed target.

For a one-phone expected word, DELETE produces an empty target. Empty-target CTC likelihood is well-defined and R4 explicitly exercised it, but the R5-1A scorer rejects empty targets. A new contract must define and statically verify safe empty-target handling rather than importing `score_hypothesis` unchanged.

Within-family tie rules and cross-family BEST_NON_ADDITION precedence must be preregistered before execution. R5-2A intentionally does not select final cross-family tie semantics.

## 6. Computational and runtime feasibility

One acoustic forward pass can supply the logits for all `80N+41` exact CTC hypotheses. Existing code demonstrates the needed mechanics:

- reuse one log-softmax tensor for every hypothesis of a word;
- reject impossible targets before CTCLoss;
- batch flattened alignable targets;
- group work by encoder or target length to reduce padding;
- score KEEP once;
- preserve only family maxima and deterministic identities when full candidate exports are unnecessary.

Runtime inputs are only word audio, expected text converted to canonical phones, the frozen preprocessing/model/vocabulary, and deterministic hypothesis construction. Manual annotation, known observed phone, speaker lookup, per-speaker oracle threshold, future same-speaker audio, and TEST information are not required.

The binary output remains coherent: ADDITION can later be defined only when BEST_INSERT beats the strongest permitted non-addition explanation by a frozen criterion. The selected INSERT still supplies an added-phone identity and insertion boundary. If KEEP, SUB, or DELETE wins, the binary output is NON_ADDITION; the winning relation may remain a research diagnostic without claiming a validated complete relation classifier.

No latency benchmark was run, so runtime plausibility is structural rather than measured.

## 7. Multiple-error limitation

One-edit hypotheses cannot fully explain:

- substitution plus addition;
- deletion plus addition;
- multiple additions;
- substitution plus deletion;
- multiple same-family errors.

The scorer can still answer a small causal question: does adding explicit one-edit non-addition competition reduce explanation confounding relative to KEEP-only comparison? Mixed and multiple-error words must remain included and separately diagnosed in a future contract; they must not be silently dropped.

One-edit competition is therefore valid as a limited feasibility experiment, not as a complete pronunciation-sequence decoder. Arbitrary multi-edit search is outside R5-2A.

## 8. Evaluation contamination and valid options

Current VALIDATION addition performance was consumed by R5-1B, and R5-1C used it to motivate the present hypothesis. It cannot serve as untouched R5-2 confirmation.

Valid future options are:

1. **TRAIN-only LOSO development — preferred smallest step.** Freeze one formula, ties, metrics, gates, threshold protocol, and stop policy before scoring. Use 12-fold TRAIN-speaker LOSO only.
2. **Transparent iterative VALIDATION transfer.** Possible later under a separate frozen contract, but only as prior-used iterative evidence and never for formula/threshold selection or independent-confirmation language.
3. **New external or newly held-out speakers.** If available under a frozen collection/evaluation protocol, these provide fresher transfer evidence while preserving current TEST.
4. **Future locked TEST.** Only after development requirements pass, under a dedicated preregistration and separate authorization. TEST must never become development validation.

No consumed VALIDATION score was used to rank formulas, thresholds, normalization, weights, penalties, or operating points in this audit.

## 9. Intervention-family comparison

| Family | Direct target | Strength | Main limitation | Smallest valid experiment |
|---|---|---|---|---|
| A: calibration-only | Speaker score-location shift | Directly matches broad FAR/location movement | Does not address intrinsic overlap; runtime calibration context is undefined; high contamination risk | One separately preregistered TRAIN-speaker LOSO calibration audit |
| B: relation competition | Relation-confounded class overlap; partial shift robustness | Smallest coherent-likelihood change; direct mechanism for SUB/DELETE false-positive cohorts | Correct-only errors dominate; one-edit limitation; no guarantee on shift | One frozen relation score in a TRAIN-only 12-fold LOSO comparison |
| C: combined | Both mechanisms in principle | Matches the mixed post-hoc interpretation | Confounds causal attribution and adds the most degrees of freedom after VALIDATION consumption | Only after isolated mechanisms; factorial TRAIN-only design |

Relation competition is preferred because it is the smallest runtime-compatible and causally interpretable change with a direct mechanism. Calibration-only work remains plausible but is not preferred for this first new generation. A combined intervention is not yet justified because it would prevent clean attribution and multiply contamination risk.

## 10. Final feasibility decision

Final status:

`R5_2A_RELATION_COMPETITION_FEASIBLE`

This means relation-competitive scoring is technically and operationally representable and has sufficient mechanistic justification for one small, separately preregistered TRAIN-only experiment. It does not mean performance will improve.

Preferred next hypothesis:

> A single preregistered relation-competitive exact CTC score should compare BEST_INSERT with the strongest permitted KEEP/SUBSTITUTION/DELETE explanation in one TARGET-normalized, alignability-safe likelihood space.

R5-2A does not freeze the final margin, threshold, gates, cross-family ties, or execution contract. It does not implement the scorer.

## 11. Protocol audit

- Training: NO.
- Model inference: NO.
- New TRAIN predictions: NO.
- TRAIN or VALIDATION rerun: NO.
- Validation optimization: NO.
- Threshold search: NO.
- New performance metrics: NO.
- Audio paths resolved: NO.
- TEST audio/inference/performance: NO.
- R5-1 modified: NO.
- R4 reopened or modified: NO.
- R5-2 scorer implemented: NO.

## 12. Next action boundary

The smallest scientifically valid next action is to create a separate R5-2B TRAIN-only development contract that freezes exactly one relation-competition formula, empty-target handling, deterministic tie rules, metrics, gates, and stop policy before any inference. This action was not implemented in R5-2A.
