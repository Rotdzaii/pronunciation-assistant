# L2-ARCTIC Expected/Observed V4 Contract

Status: **RESEARCH_ONLY**, **NOT_PRODUCTION**, **NOT_RUNTIME_CONNECTED**, and
**NOT_USED_BY_CURRENT_CHECKPOINTS**.

Gate R3-0 concluded `R3_0_PASS_WITH_WARNINGS`: the manual annotation is usable
for expected-versus-observed research, but malformed perceived-phone labels and
speaker-dependent annotation quality require explicit isolation. V4 implements
that isolation without changing V2, V3, raw TextGrid files, or runtime code.

## Source and accounting

The builder reads only the `phones` `IntervalTier` from all 3,599 files under
the 24 speakers' `annotation/*.TextGrid` directories. It does not read the
`textgrid` directory, words tier, IPA tier, or any archive.

V4 is a full audit ledger, not a training split: every one of the 135,890 raw
phone intervals has one row. No train/validation/test assignment is created.

| Relation | Rows | Status |
|---|---:|---|
| correct | 101,626 | clean main target |
| substitution | 12,361 | clean main target |
| deletion | 3,418 | clean, separately isolated |
| addition | 1,044 | addition audit only |
| non-speech | 15,644 | excluded non-speech |
| unresolved | 1,797 | retained for audit only |

The clean main set is 117,405 rows. `PHONE_IDENTIFICATION_ELIGIBLE` contains
113,987 clean correct/substitution rows. `DELETION_ELIGIBLE` contains 3,418
rows. The clean observed real-phone inventory contains 40 canonical classes.

## Raw and derived field contract

`raw_label` is copied verbatim from the phones-tier interval. Identity fields
include `speaker_id`, `audio_path`, `utterance_id`, `textgrid_path`,
`interval_index`, `start_time`, and `end_time`.

All interpretations are stored separately:

- `relation` is `correct`, `substitution`, `deletion`, `addition`,
  `non_speech`, or `unresolved`.
- `expected_phone_raw` and `observed_phone_raw` preserve parsed phone tokens.
- `expected_phone_canonical` and `observed_phone_canonical` remove a terminal
  stress digit 0/1/2 only when the token is a valid ARPAbet vowel.
- `label_quality`, `exclusion_reason`, and `research_subset` make eligibility
  explicit. Boolean convenience fields do not replace those source fields.

Malformed and lowercase tokens are never corrected by inference. They remain
in `UNRESOLVED` with their exact raw annotation available for audit.

## Deletion and addition isolation

A clean deletion such as `T,sil,d` has expected phone `T`, observed canonical
target `<SIL>`, and `is_deletion=true`. `<SIL>` is not a member of the 40-class
real-phone target inventory; future work can exclude deletion directly via
`DELETION_ELIGIBLE`.

A clean addition such as `sil,T,a` is kept in `ADDITION_AUDIT` with
`label_quality=excluded_addition`. It is not part of the clean main set or the
phone-identification set.

## Unresolved accounting

All 1,797 unresolved intervals remain in the CSV:

| Exclusion reason | Rows |
|---|---:|
| substitution invalid observed phone | 1,737 |
| deletion invalid expected phone | 2 |
| addition invalid observed phone | 48 |
| unrecognized non-error label | 10 |

The companion audit JSON lists every malformed token by speaker, the ten
unknown non-error labels, and per-speaker retention rates.

## Speaker quality warning

Substitution annotation retention varies materially. With a predeclared 20%
exclusion-rate flag, these speakers require per-speaker reporting in later
research:

| Speaker | Tagged substitutions | Clean | Excluded | Exclusion % |
|---|---:|---:|---:|---:|
| ABA | 402 | 251 | 151 | 37.562 |
| ASI | 761 | 451 | 310 | 40.736 |
| ERMS | 855 | 621 | 234 | 27.368 |
| SKA | 571 | 455 | 116 | 20.315 |
| SVBI | 717 | 395 | 322 | 44.909 |
| YBAA | 331 | 220 | 111 | 33.535 |

No speaker is removed automatically.

## Reproduction

Run `build_l2_arctic_expected_observed_v4.py` with `--dataset-root` pointing to
`l2arctic_release_v5.0`. The builder validates all locked counts, unique
interval identity, audio lookup, 24-speaker coverage, clean-target inventory,
and exclusion isolation before writing. It refuses to overwrite an existing V4
CSV or audit JSON.
