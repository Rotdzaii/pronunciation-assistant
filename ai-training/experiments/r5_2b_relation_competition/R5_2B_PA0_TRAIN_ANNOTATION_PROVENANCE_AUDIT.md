# R5-2B-PA0 TRAIN Annotation-Root Provenance Audit

## 1. Identity verification

PASS. The scientific contract, frozen scorer, original failed execution, TC0, TC1 contract, TC1 static regression, corrected execution, V4 metadata, and checkpoint identities match. Seven manifests covering 59 recorded entries were re-read; every byte size and SHA-256 matched.

## 2. TC1 failure-path trace

The TC1 driver derives `REPO_ROOT` from its own location in the research worktree, then hardcodes:

`AUDIO_ROOT = REPO_ROOT / ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0`

It passes this value directly to the inherited R5-1A population functions. For BWC/arctic_a0006 it generated:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0\BWC\annotation\arctic_a0006.TextGrid`

That raw root does not exist. The root came from newly written TC1 driver logic, not from the environment-selected R5-1A behavior.

## 3. Frozen prior-pipeline provenance

The prior pipeline is consistent:

- R5-0 records `C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0` as the root used for data and TRAIN inference.
- R5-1A requires `L2_ARCTIC_ROOT` and resolves manual annotations as `audio_root / speaker / annotation / (utterance + .TextGrid)`.
- R4-2C and R4-3A use the same root and path schema.
- The V4 builder documents `l2arctic_release_v5.0/<speaker>/annotation/*.TextGrid` and enumerates that directory.

The authoritative mapping is therefore:

`MANUAL_TEXTGRID_ROOT / <speaker> / annotation / <utterance_id>.TextGrid`

`AUDIO_ROOT` and `MANUAL_TEXTGRID_ROOT` are distinct logical roles but share the same raw L2-ARCTIC directory in this frozen pipeline. `V4_METADATA_ROOT` is the separate research-worktree metadata directory and is not an annotation root.

## 4. First missing-file check

Authoritative path:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0\BWC\annotation\arctic_a0006.TextGrid`

Exists: YES. Size: 8,123 bytes. SHA-256: `EF3056511B71013C399F5F69F40B6A8865756FE26091DF72CC5EDB8E7D14A337`.

No TextGrid content was read.

## 5. TRAIN existence audit

Distinct TRAIN `(speaker_id, utterance_id)` pairs in frozen V4 metadata were mapped using the authoritative formula.

| Speaker | Expected | Existing | Missing |
|---|---:|---:|---:|
| BWC | 150 | 150 | 0 |
| EBVS | 150 | 150 | 0 |
| HJK | 150 | 150 | 0 |
| NCC | 150 | 150 | 0 |
| NJS | 150 | 150 | 0 |
| PNV | 150 | 150 | 0 |
| RRBI | 150 | 150 | 0 |
| TLV | 150 | 150 | 0 |
| TNI | 150 | 150 | 0 |
| YBAA | 149 | 149 | 0 |
| YKWK | 150 | 150 | 0 |
| ZHAA | 150 | 150 | 0 |
| **Total** | **1,799** | **1,799** | **0** |

Duplicate or ambiguous mappings: 0.

## 6. Root-cause classification

`PATH_ROOT_MISCONFIGURATION`

The files and frozen schema are intact. The corrected driver selected the wrong physical root by deriving it from the research worktree.

## 7. Minimal fixability assessment

Only a future `PATH_RESOLUTION_ONLY` correction is justified: an additive execution driver/configuration may pass the established external manual-annotation root to the unchanged population helpers. It may not alter annotation contents, population semantics, scorer, C formula, model, gates, LOSO, or prior evidence. No correction was implemented here.

## 8. Scientific impact

R5-2 performance remains UNKNOWN. Zero of eight gates were evaluated. No scientific PASS/FAIL is assigned.

## 9. VALIDATION / TEST status

VALIDATION was not accessed and no VALIDATION path was resolved. TEST was not accessed and no TEST path was resolved.

## 10. Protocol audit

No training, checkpoint inference, TRAIN audio access, annotation-content reading, threshold search, or performance calculation occurred. Only TRAIN annotation existence was checked. The contract, scorer, and previous executions were not modified.

## 11. Final status

`R5_2B_PA0_ANNOTATION_PROVENANCE_CONFIRMED`

## 12. Next action

Freeze a path-resolution-only technical-correction contract for the R5-2B execution driver.
