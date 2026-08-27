# R5-2B Relation-Competitive Scorer Static Verification

## Scope and result

The frozen R5-2B scorer was implemented and verified using synthetic labels, logits, scores, and events only. Two complete executions of the 50-test suite passed with byte-identical summaries. No Phoenix audio, checkpoint inference, LOSO evaluation, threshold search, or performance metric was used.

Final static status: `R5_2B_STATIC_VERIFICATION_PASS`.

## Identity verification

The complete frozen R5-2B contract manifest was re-read and every recorded size and SHA-256 was verified before implementation. The required contract, preregistration, R5-2A, V4, checkpoint, and R5-1A scorer identities matched exactly.

## Candidate construction

The implementation constructs exactly one KEEP target, `40(N+1)` INSERT targets, `39N` SUB targets, and `N` DELETE targets, for a total of `80N+41`. Totals were verified for `N = 1, 2, 3, 5` as 121, 201, 281, and 441. SUB never replaces a phone with itself. DELETE candidates with identical target sequences retain distinct position metadata.

## CTC score semantics

For every nonempty target, minimum CTC steps equal target length plus adjacent-repeat count. Impossible targets receive mathematical negative infinity and bypass CTC loss. Alignable nonempty targets use the frozen target-normalized exact CTC score.

The new one-phone DELETE path was checked independently. For deterministic synthetic logits, the explicit all-blank sum and PyTorch empty-target CTC result were both `-19.43882697170362`; absolute difference was `0.0`, within the frozen `1e-10` tolerance.

## Relation competition

BEST_INSERT, BEST_SUB, and BEST_DELETE internal tie rules passed. BEST_NON_ADDITION uses metadata priority KEEP, then SUB, then DELETE on exact numeric ties without changing the maximum score. Positive, zero, and negative `C = BEST_INSERT - BEST_NON_ADDITION` cases passed. Mechanism fixtures confirmed that SUB or DELETE can reverse a positive INSERT-versus-KEEP margin while INSERT remains positive when it beats all competitors.

When all INSERT targets are impossible, no insertion metadata is fabricated and `C` is negative infinity. Impossible SUB/DELETE candidates cannot defeat finite candidates. KEEP-impossible detection is exposed as a future execution guard.

## Helpers and determinism

Standards-compatible extended-real serialization passed for KEEP, INSERT, SUB, DELETE, BEST_NON_ADDITION, and C. Threshold candidate construction excludes negative infinity, retains unique finite float64 scores and frozen `nextafter` edges, uses `C >= theta`, and preserves the frozen tie order. Correct-only, substitution, and deletion cohort FAR helpers passed. Exact event matching requires word, phone, and boundary, uses deterministic one-to-one matching, and enforces at most one predicted BEST_INSERT event per word.

Both runs passed 50 of 50 tests. Their byte-identical summary SHA-256 was `F0E54A0A4DBC1378F424E6C8FE8B0A9861CDD72DCBE72C045B96660324723197`.

## Protocol audit

- Neural training: no
- Checkpoint inference: no
- TRAIN audio accessed: no
- Phoenix performance metrics calculated: no
- VALIDATION accessed: no
- TEST accessed: no
- R5-1 modified: no
- Frozen R5-2B contract modified: no
- Production runtime modified: no

This result establishes only implementation fidelity under synthetic static verification. It is not scientific performance evidence.
