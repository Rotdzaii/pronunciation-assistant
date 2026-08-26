# R4-4D1C Matched-Control Row-Identity Correction Freeze

Final: **R4_4D1C_IDENTITY_CORRECTION_FROZEN**

All frozen source identities and the authoritative v1 TRAIN threshold passed SHA verification. The threshold remains RAW `theta = 2.197946548461914` with SHA-256 `36F6FD5AB6B7E98A607D499445E455DCAB8C3DD4ACDD19F252DC472FCDD07E94`; it was not recalculated or modified.

The identity discrepancy is semantic bookkeeping. `csv.DictReader` data rows are enumerated from zero, while the matched-control `source_csv_row` stores the one-based physical CSV line including the header. Thus `source_index=0` is physical line 2 and the canonical identity is `int(source_index)+2`.

The old convention reproduces 1,201/1,434 mappings with 233 missing. The canonical convention maps 1,434/1,434 with zero missing identities, duplicates, collisions, ambiguities, or checked metadata-field mismatches. Support remains 717 deletion, 717 non-deletion, 32 phones, and all six validation speakers.

The new v2 driver is validation-only. It imports the frozen v1 acoustic/scoring/metric/classification functions, rejects TRAIN recalibration, verifies and loads the immutable v1 threshold, verifies the matched mapping before inference, and reserves `locked_execution_v2` without altering `locked_execution_v1`. Its only semantic changes are the canonical row identity, removal of TRAIN calibration, immutable threshold loading, and the v2 output directory.

Synthetic/static tests A-J: **10/10 PASS**.

No VALIDATION acoustic inference, hypothesis scoring, or metrics were performed. No neural training occurred. R4 TEST remained closed. `locked_execution_v2` was not run.
