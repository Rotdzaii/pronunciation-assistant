# L2-ARCTIC All-Speaker Phone Error Dataset

## Purpose

This dataset expands the existing Vietnamese-only L2-ARCTIC phone error classification metadata to all available L2-ARCTIC speakers while preserving the same clean v2 schema used by:

`ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification_v2.csv`

The output file is:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

This is a data preparation feature only. It does not train models and does not mix in unrelated datasets.

## Scope

The dataset uses only L2-ARCTIC manual annotation TextGrid metadata. It keeps the same phone-error labels as the Vietnamese v2 dataset:

- `addition`
- `deletion`
- `substitution`

Rows are filtered to the `phones` tier so duplicate IPA/comment tiers are not included in the classifier dataset.

The CSV keeps the Vietnamese v2 column layout for compatibility with existing training scripts:

- `dataset`
- `speaker_id`
- `l1`
- `gender`
- `split`
- `audio_path`
- `utterance_id`
- `target_text`
- `tier_name`
- `label`
- `error_type`
- `start_time`
- `end_time`
- `original_duration`
- `context_start_time`
- `context_end_time`
- `context_duration`

The `dataset` column is the source dataset marker and remains `l2_arctic`. The `l1` column is the inferred L1 group.

## Relationship To Vietnamese v2

The Vietnamese v2 dataset remains the clean Vietnamese-only baseline. The all-speaker dataset uses the same label normalization, tier filtering, duplicate removal key, split convention, and context-window calculation.

Vietnamese speakers in the all-speaker file are still:

- `HQTV`
- `PNV`
- `THV`
- `TLV`

Vietnamese-only evaluation should continue to use either the Vietnamese v2 CSV or the Vietnamese subset inside the all-speaker CSV.

## Expected Benefit

Using all available L2-ARCTIC speakers increases the number of phone-error samples for the same classification schema. This should help the model learn a broader representation of general phone-error patterns and provide more examples for:

- additions
- deletions
- substitutions

This is useful for the next training feature because Phase 1 selected CNN Attention as the current main model candidate and the classifier can now be trained with a larger same-schema metadata file.

## Limitations

This expansion does not prove Vietnamese accent-specific modeling. Most rows are from non-Vietnamese L1 groups, so Vietnamese-specific performance must still be evaluated separately.

The dataset should not be described as solving Vietnamese pronunciation diagnosis by itself. It only expands L2-ARCTIC phone-error metadata using the same schema.

This feature intentionally does not include Common Voice, speechocean762, Speak & Improve, raw audio commits, checkpoints, model files, archives, or quarantine folders.

## Generated Review Outputs

The review script writes all-speaker inspection files under:

`ai-training/datasets/l2-arctic/evaluation/`

Expected files:

- `all_speakers_error_distribution.csv`
- `all_speakers_error_by_l1.csv`
- `all_speakers_error_by_speaker.csv`
- `all_speakers_split_distribution.csv`
- `all_speakers_duration_summary.csv`
- `all_speakers_dataset_review.json`

## Next Recommended Training

Recommended next feature branch:

`feature/ai-train-all-l2-arctic-cnn-attention`

Suggested training/evaluation setup:

- train CNN Attention on the all-speaker L2-ARCTIC v2 metadata
- evaluate Vietnamese subset performance separately
- optionally add speaker-disjoint evaluation to measure generalization across unseen speakers
