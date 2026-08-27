# R5-2B-PA1 Path-Resolution Correction Contract

## 1. Identity verification

PASS. The scientific contract, frozen scorer, TC1 static manifest, previous corrected-execution manifest, PA0 manifest, V4 metadata, and checkpoint match their required SHA-256 identities. All 25 entries referenced by the three required manifests match their recorded byte sizes and hashes.

## 2. Frozen PA0 finding

PA0 classified the failure as `PATH_ROOT_MISCONFIGURATION`.

The failed driver used:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0`

The authoritative raw L2-ARCTIC root is:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0`

PA0 confirmed 1,799 expected TRAIN TextGrids, 1,799 existing, zero missing, and zero ambiguous mappings.

## 3. Frozen root contract

`MANUAL_TEXTGRID_ROOT` and `AUDIO_ROOT` are independently named/configured inputs. Their current frozen values are the authoritative main-repository raw dataset root above. Neither may be inferred from the other.

`V4_METADATA_ROOT` is:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research\ai-training\datasets\l2-arctic\metadata`

It is not a raw dataset root and must not be used to derive one.

The exact manual annotation formula is:

`MANUAL_TEXTGRID_ROOT / <speaker> / annotation / <utterance_id>.TextGrid`

## 4. Allowed correction

The sole permitted correction is `PATH_RESOLUTION_ONLY`. A new additive driver may explicitly supply the authoritative roots to unchanged inherited helpers. It may not change scientific or data semantics, and it may not copy, link, download, restore, or edit dataset files.

## 5. Future pre-inference guard

Before annotation-content or audio access, the path-corrected execution must existence-check only the 12 authorized TRAIN speakers and require exactly 1,799 expected, 1,799 existing, zero missing, and zero ambiguous mappings. Any mismatch stops with `R5_2B_PA1_EXECUTION_BLOCKED_ANNOTATION_EXISTENCE`; no file may be skipped.

The first-file provenance regression is `BWC/annotation/arctic_a0006.TextGrid`, 8,123 bytes, SHA-256 `EF3056511B71013C399F5F69F40B6A8865756FE26091DF72CC5EDB8E7D14A337`. It is not a sample-selection rule.

## 6. Scientific immutability

The scientific contract, scorer, population semantics, model, threshold policy, gates, and TC1 numerical policy remain unchanged. R5-2B performance remains UNKNOWN and zero of eight gates have been evaluated.

## 7. VALIDATION / TEST status

VALIDATION and TEST remain prohibited. Their paths were not resolved or existence-checked during PA1.

## 8. Protocol audit

No training, checkpoint inference, TRAIN audio access, TextGrid-content reading, performance consumption, threshold search, VALIDATION access, or TEST access occurred. No prior evidence was modified, and the path correction was not implemented.

## 9. Final status

`R5_2B_PA1_PATH_CORRECTION_CONTRACT_FROZEN`

## 10. Next action

Perform one path-corrected continuation of the frozen R5-2B TRAIN development execution using the existing TC1 guard.
