# R4-3B0 Complete Numerical DP Contract

Status: `R4_3B0_NUMERICAL_CONTRACT_FROZEN`

This artifact completes the numerical choices missing from R4-3A. It freezes methodology only. No validation acoustic inference, validation sequence scoring, model training, or R4 TEST access occurred.

## Frozen inputs

- Acoustic model: frozen R3-1D 40-phone checkpoint, SHA-256 `5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E`.
- V4 dataset SHA-256: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`.
- R4-3A preregistration SHA-256: `044DBDE804521D2AC0D149097CC9E95BB39EA11C98DA4827AD2FEA6F243E36F2`.
- Phone order: `AA AE AH AO AW AX AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH`.

## Probe grid and acoustic evidence

Use MFA word boundaries only. For a positive-duration word, the first center is `word_start`. Repeatedly append `previous + 0.040` while that candidate is less than `word_end - 1e-9`. Append `word_end` when it differs from the last center by more than `1e-9`. Thus a short positive-duration word has probes at both boundaries, and a non-grid-aligned word has a shorter final step. A non-positive-duration word has one midpoint probe.

Every center receives the unchanged 0.50 s / 8,000-sample R3 crop. The crop may extend beyond the word and is zero-padded only at utterance edges. Each probe produces 40 logits. Define

`L[t,p] = log_softmax(logits[t])[p]`.

Store the time-by-phone log-probability matrix and convert values to float64 before DP accumulation. R3 receives audio only.

## TRAIN-only priors

Usable TRAIN words: `16,259`. Their clean expected-phone positions contain:

- `C_MATCH = 48,893`
- `C_SUB = 5,867`
- `C_DEL = 1,544`
- `N = 56,304`

With Laplace `alpha = 1.0`, denominator `N + 3 = 56,307`:

- `P_MATCH = 48,894 / 56,307 = 0.868346741968139`; `ln(P_MATCH) = -0.14116417177608467`
- `P_SUB = 5,868 / 56,307 = 0.10421439607864032`; `ln(P_SUB) = -2.261305001061487`
- `P_DEL = 1,545 / 56,307 = 0.027438861953220737`; `ln(P_DEL) = -3.595794950992514`

Natural logarithms only; no scaling or temperature.

## Spans and scores

An emitted expected phone consumes any contiguous span `S=[t,t+k)` where `k>=1` and `t+k<=T`, subject only to a complete monotonic path. Deletion consumes zero probes. There are no phone-, relation-, or duration-specific span bounds.

For every phone `p`:

`A(S,p) = (1/k) * sum_{u=t}^{t+k-1} L[u,p]`.

For expected phone `e`:

- `MATCH_SCORE(e,S) = ln(P_MATCH) + A(S,e)`.
- `q* = argmax_{q != e} A(S,q)`, breaking exact phone ties by the lowest canonical index.
- `SUB_SCORE(e,S) = ln(P_SUB) + A(S,q*)`.
- `DELETE_SCORE(e) = ln(P_DEL)`.

The mean competing-phone evidence is the complete definition of sustained substitution evidence. There is no posterior, majority, margin, or minimum-winning-probe threshold.

`ADVANCE_TIME` is structural: increasing `k` consumes another contiguous probe within the same emitted-phone span. It has no prior, continuation penalty, duration bonus, or duration penalty.

## Dynamic program

Let `E[0..n-1]` be the expected sequence and let `T` be the number of probes. `DP[i,t]` is the best score after consuming exactly the first `i` expected phones and first `t` probes.

Initialization:

- `DP[0,0] = 0`.
- Every other state is negative infinity.

For each reachable `DP[i,t]`, `i<n`:

- For every `k>=1`, `t+k<=T`, propose MATCH at `DP[i+1,t+k]` using `MATCH_SCORE(E[i],[t,t+k))`.
- For the same spans, propose SUBSTITUTION at `DP[i+1,t+k]` using `SUB_SCORE(E[i],[t,t+k))` and store `q*`.
- Propose DELETE_EXPECTED at `DP[i+1,t]` using `DELETE_SCORE(E[i])`.

There is no insertion transition. The only valid termination is `DP[n,T]`; no leading or trailing probes are free. If unreachable, return `NO_VALID_PATH`.

## Tie and ambiguity policy

Accumulate in float64. Candidate A is strictly better than B only when `score_A > score_B + 1e-8`. Values within `1e-8` are tied and resolve globally in this order:

1. MATCH over SUBSTITUTION over DELETE_EXPECTED.
2. For MATCH/SUBSTITUTION, shorter span.
3. For SUBSTITUTION, lower canonical `q*` index.
4. Lower acoustic start index of the current predecessor transition.
5. Lexicographically smaller full operation-record sequence. Each expected-position record is `(operation_rank, probe_start, probe_end, q*_or_-1)`, with MATCH=0, SUBSTITUTION=1, DELETE_EXPECTED=2. This also deterministically resolves paths having the same relation sequence but different earlier span partitions.

Maintain the best and second-best distinct complete paths with exact k=2 bookkeeping where practical. Path identity is the ordered full operation-record sequence defined above. Define `path_score_gap = best - second_best`; a word is `NUMERICALLY_AMBIGUOUS` when a distinct second path exists and the gap is at most `1e-8`. The deterministic tie policy still emits exactly one path.

## Frozen validation gates

- Binary Macro-F1 `>= 0.70`.
- Deletion F1 `>= 0.40`.
- Deletion recall `>= 0.45`.
- Binary Macro-F1 gain over frozen duration baseline `>= +0.03`.
- Strict-matched Macro-F1 `>= 0.60`.
- Strict-matched deletion F1 `>= 0.55`.
- Substitution false-deletion rate `<= 0.25`.
- Each validation speaker with at least 30 deletions must have deletion recall `>= 0.25`.

No additional phone/generalization hard gate existed in the R4-3A preregistration.

## Closure

- Validation acoustic scoring: **NO**
- R4 TEST paths resolved/accessed/inferred: **NO**
- Training: **NO**
- R3 checkpoint modification: **NO**

The next R4-3B run must verify the hashes of this JSON and Markdown contract before any validation inference.
