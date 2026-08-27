# L2-ARCTIC Correctness Dataset V3

## Status

- Gate R0: `PASS`.
- Dataset usage: `RESEARCH_ONLY`.
- Runtime usage: `NOT_USED_BY_RUNTIME`.
- Current checkpoint usage: `NOT_USED_BY_CURRENT_CHECKPOINT`.
- The legacy/V2 builder, metadata, checkpoints, and three-class semantics are unchanged.

## Verified source and semantics

V3 reads only the 24 scripted-speech speaker directories under
`l2arctic_release_v5.0/<speaker>/annotation/*.TextGrid` and only the `phones`
`IntervalTier`. It does not read `/textgrid/`, `words`, `IPA`, or
`suitcase_corpus.zip`.

The mapping follows the corpus manual-annotation convention:

- `CPL,PPL,s` -> `substitution`
- `CPL,sil,d` -> `deletion`
- `sil,PPL,a` -> `addition`
- plain corpus ARPAbet phones, including vowel stress `0/1/2`, -> `correct`
- empty, `sp`, and `sil` -> excluded `non_speech`
- every other label -> excluded `unknown`

Unknown labels are preserved exactly and are not repaired or inferred.

## Gate R0 verified distribution

| Category | Count |
| --- | ---: |
| correct | 101,626 |
| substitution | 14,098 |
| deletion | 3,420 |
| addition | 1,092 |
| non_speech | 15,644 |
| unknown | 10 |
| total phone intervals | 135,890 |

The V3 training metadata contains only `correct`, `substitution`, `deletion`,
and `addition`, for 120,236 rows. This dataset does not define a binary Stage-1
split and does not select sampling, weighting, loss, threshold, calibration, or
model architecture. Those decisions belong to later research gates.

## Reproducibility

Builder:

```text
ai-training/scripts/build_l2_arctic_all_speakers_correctness_v3.py
```

Outputs:

```text
ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3.csv
ai-training/datasets/l2-arctic/metadata/all_speakers_phone_correctness_v3_audit.json
```

The builder validates the Gate R0 counts and all required invariants before it
writes either V3 output. It refuses to overwrite an existing V3 output.
