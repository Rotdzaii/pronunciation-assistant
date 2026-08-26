# R5-1A Static Verification

Status: `R5_1A_STATIC_VERIFICATION_PASS`

## Scope

This stage implemented the frozen R5-1A scorer and verified it using only deterministic synthetic labels, logits, and tensors. It did not load the frozen checkpoint, resolve or read audio, run real inference, calculate Phoenix performance metrics, execute LOSO, or derive a robust threshold.

## Identity verification

- R5-1A contract: `A6BE2C1C6A09AC0007E9330E44C1C7F45A91CCB76E47EE63ACEB99D0781A1BEB`
- R5-1A preregistration: `2CE9F25B91139B9EA38E2AB552B11C29AA1397B252CE957833D1E3A80D689141`
- Frozen contract manifest: `93EB772243E066AB0D2A406F9FD4FEABDB6D9B99F55A794DDC58BD29E18A80FC`
- V4: `160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D`
- Checkpoint: `F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085`

All identities matched. The checkpoint was hashed but never loaded.

## Implementation results

All required cases A–L passed:

- exact adjacent-repeat counts and minimum-step rules;
- impossible targets intercepted before CTCLoss and scored as mathematical negative infinity;
- finite alignable synthetic TARGET scores;
- unique BEST_INSERT selection;
- boundary and phone tie rules;
- impossible candidates unable to beat finite candidates;
- all-INSERT-impossible output with no fabricated phone or boundary;
- KEEP-impossible stop `R5_1A_EXECUTION_BLOCKED_KEEP_ALIGNABILITY`.

The zero-infinity regression fixture reproduced PyTorch's zero loss for an impossible target under `zero_infinity=True`. The R5-1A guard returned negative infinity without calling CTCLoss for that target.

## Extended-real behavior

The custom Mann–Whitney rank AUC passed finite, negative-infinity, tied-negative-infinity, finite-tie, and mixed-class-tie cases. The finite fixture agreed with sklearn to one ULP; the test uses a fixed `1e-15` cross-check tolerance. Negative infinity is never replaced by a finite value.

Threshold generation excludes negative infinity, includes every unique finite float64 value and both frozen `np.nextafter` edges, and preserves `A>=theta`.

JSON round trips use `score_value=null` plus `score_is_neg_inf=true`. Finite values remain numeric. Bare non-standard `-Infinity` is never written.

## Determinism

The complete 17-test suite ran twice. Both serialized outputs were byte-identical:

`571A8E637BC697EFEFFF8F17DA3313835F11DF9A0F0C71F7F54F355F40CDD622`

## Protocol closure

- Neural training: NO
- Real checkpoint inference: NO
- TRAIN audio accessed: NO
- Phoenix performance metrics: NO
- VALIDATION accessed: NO
- TEST accessed: NO
- R5-1 modified: NO
- Frozen R5-1A contract modified: NO
