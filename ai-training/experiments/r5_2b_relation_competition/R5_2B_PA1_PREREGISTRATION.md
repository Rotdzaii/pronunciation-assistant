# R5-2B-PA1 Path-Resolution-Only Technical Correction Preregistration

## Status entering PA1

The R5-2B scientific contract and scorer remain frozen. The TC1 numerical guard passed static regression. The first corrected TRAIN execution stopped before TRAIN audio, checkpoint inference, scores, LOSO, or performance because it resolved manual annotations under an absent research-worktree raw-data root.

PA0 established `PATH_ROOT_MISCONFIGURATION`, confirmed one authoritative dataset root, and found 1,799 of 1,799 required TRAIN TextGrids with zero missing or ambiguous mappings. R5-2B performance remains unknown; zero of eight gates have been evaluated.

## Frozen roots and roles

`MANUAL_TEXTGRID_ROOT`:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0`

`AUDIO_ROOT`:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\datasets\l2-arctic\raw\l2arctic_release_v5.0`

`V4_METADATA_ROOT`:

`C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research\ai-training\datasets\l2-arctic\metadata`

`AUDIO_ROOT` and `MANUAL_TEXTGRID_ROOT` are independent logical inputs even though frozen provenance currently gives them the same physical value. Neither may be inferred from the other. The manual root must not be derived from the driver location, research-worktree location, or V4 metadata location.

The exact annotation mapping is:

`MANUAL_TEXTGRID_ROOT / <speaker> / annotation / <utterance_id>.TextGrid`

## Only authorized correction

The only permitted future implementation change is `PATH_RESOLUTION_ONLY`. A new additive driver may explicitly configure and pass the authoritative roots to the unchanged inherited population and audio-resolution helpers.

The correction may not change population eligibility, TextGrid parsing, V4, scorer, C, hypothesis families, model, checkpoint, preprocessing, TC1 numerical policy, LOSO, threshold selection, gates, or event semantics. Previous failed drivers, manifests, reports, and TC0/TC1/PA0 evidence must remain unchanged.

No dataset restoration, copying, symlinking, downloading, or annotation editing is authorized.

## Required pre-inference guard

Before future TRAIN annotation content or audio access, the additive driver must existence-check only the authorized TRAIN mapping and require:

- expected: 1,799
- existing: 1,799
- missing: 0
- ambiguous: 0

Speaker counts are BWC 150, EBVS 150, HJK 150, NCC 150, NJS 150, PNV 150, RRBI 150, TLV 150, TNI 150, YBAA 149, YKWK 150, and ZHAA 150.

Any mismatch must stop before TRAIN inference with `R5_2B_PA1_EXECUTION_BLOCKED_ANNOTATION_EXISTENCE`; no file may be silently skipped.

The guard must also verify `BWC/annotation/arctic_a0006.TextGrid` against the frozen PA0 provenance identity: 8,123 bytes and SHA-256 `EF3056511B71013C399F5F69F40B6A8865756FE26091DF72CC5EDB8E7D14A337`. This is a provenance regression, not a sample-selection rule.

## Split policy

The guard and any later corrected execution are TRAIN-only. VALIDATION and TEST paths must not be resolved, existence-checked, or accessed.

## Next-stage policy

No additional static path suite is required. After this contract freezes, the only next authorized action is one path-corrected continuation of the frozen R5-2B TRAIN development execution using the unchanged scorer and existing TC1 numerical guard.
