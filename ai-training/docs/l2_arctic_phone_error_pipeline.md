# L2-ARCTIC Phone Error Metadata Pipeline

## Scope

This pipeline inspects and extracts phone-level pronunciation error labels from the manually annotated L2-ARCTIC TextGrid files for Vietnamese speakers:

- HQTV
- PNV
- THV
- TLV

The source files are the `annotation/*.TextGrid` files under each speaker folder. The full forced-alignment `textgrid/*.TextGrid` folders were counted but not used for the first raw error-label dataset because the `annotation/` files contain the manually examined labels.

## Files Inspected

Each Vietnamese speaker has:

- 150 files in `annotation/`
- 1132 files in `textgrid/`

The required parser run processed 600 manually annotated TextGrid files total.

## TextGrid Tiers Found

The annotation TextGrids use three interval tiers:

- `words`
- `phones`
- `IPA`

Raw extraction produced 40,838 interval rows:

- `phones`: 23,261
- `IPA`: 9,452
- `words`: 8,125

## Label Interpretation

The manually annotated phone tiers include comma-separated labels such as:

- `DH,D,s`
- `R,sil,d`
- `sil,Z,a`

The pipeline conservatively maps only clear final error codes:

- `s` -> `substitution`
- `d` -> `deletion`
- `a` -> `addition`

All labels without a clear error code are kept in the raw CSV as `unknown`. The script does not infer correctness from unlabeled phones because those labels may simply be alignment labels rather than reviewed correctness labels.

Raw possible error type distribution:

- `unknown`: 31,026
- `substitution`: 6,306
- `deletion`: 3,117
- `addition`: 389

The clean classification CSV keeps only explicit phone-tier and IPA-tier error labels with valid segment times and existing audio files. It contains 9,812 rows:

- `substitution`: 6,306
- `deletion`: 3,117
- `addition`: 389

## Outputs

- `ai-training/datasets/l2-arctic/metadata/vietnamese_phone_annotations_raw.csv`
- `ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_summary.json`
- `ai-training/datasets/l2-arctic/metadata/vietnamese_phone_error_classification.csv`

The optional segment-level CNN baseline was trained from the clean classification CSV and saved a local checkpoint at `ai-training/models/l2_arctic_error_type_cnn.pt`. The checkpoint must remain uncommitted.

## Why This Is Closer to Mispronunciation Detection

Native-vs-non-native classification only asks whether an utterance comes from a native reference dataset or an L2 speaker dataset. High accuracy on that task can be inflated by speaker identity, recording conditions, corpus source differences, or dataset artifacts.

The phone error pipeline uses manually annotated L2-ARCTIC phone intervals and explicit error labels. This moves the modeling target closer to real pronunciation diagnosis: identifying concrete substitution, deletion, and addition events at time-aligned phone segments.

## Limitations

- The raw extraction keeps many `unknown` rows because most word labels and normal phone labels do not explicitly encode an error type.
- The current mapping is conservative and only trusts clear final error codes.
- No `correct` class was created because the TextGrid format does not clearly mark reviewed correct pronunciations.
- Segment-level CNN performance should not be treated as pronunciation correctness. It is a first baseline for explicit annotation labels only.
- Class distribution is imbalanced, especially for `addition`.

## Next Step

Use the clean phone error classification CSV to build a stronger segment-level model with class balancing and speaker-independent evaluation. The next metadata improvement should pair each error segment with target phone, realized phone, word context, and possibly neighboring phone context.
