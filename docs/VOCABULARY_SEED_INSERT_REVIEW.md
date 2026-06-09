# Vocabulary Seed Insert Review

## Purpose

This document reviews the DB1 seed insertion draft for the pronunciation-focused vocabulary feature.

The seed draft is documentation-only SQL. It is intended for later manual review and manual application through the Supabase SQL Editor. It does not create new tables, modify backend logic, modify frontend logic, insert audio files, or apply data automatically.

## Seed Source

Seed source file:

```text
docs/vocabulary_seed_items.sample.json
```

Seed insertion draft:

```text
docs/vocabulary_seed_insert.sql
```

The draft includes 46 seed vocabulary items from the sample JSON.

## Sets Created

The draft creates these public active vocabulary sets when a set with the same normalized title does not already exist:

- `Final Consonants Practice`
- `Vowel Length Practice`
- `Word Stress Practice`
- `Consonant Clusters Practice`
- `Common Vietnamese Learner Challenges`

Items are linked into sets through `public.vocabulary_set_items` based on their `common_mistake_tags`.

## Duplicate Prevention

Vocabulary items use the DB1 pronunciation-aware unique index:

```text
vocabulary_items_word_pronunciation_unique_idx
```

The draft inserts items with:

```sql
on conflict (
  lower(btrim(word)),
  coalesce(phonetic, ''),
  coalesce(stress_pattern, '')
) do nothing
```

This allows the same English word to appear multiple times when pronunciation or stress metadata differs, while preventing exact duplicate `word` + `phonetic` + `stress_pattern` entries.

Vocabulary sets are inserted only when no existing set has the same normalized title. Set-item links use:

```sql
on conflict on constraint vocabulary_set_items_set_item_unique do update
set sort_order = excluded.sort_order
```

This avoids duplicate set-item links and keeps ordering deterministic if the draft is reviewed and rerun.

## Manual Apply Notes

Do not apply this SQL automatically.

When approved, apply manually through the Supabase SQL Editor:

```text
docs/vocabulary_seed_insert.sql
```

After applying, run the read-only verification queries:

```text
docs/vocabulary_seed_insert_verification.sql
```

## Verification Queries

The verification file checks:

- Count of active seed vocabulary items.
- Count of active public seed vocabulary sets.
- Count of seed set-item links.
- Item counts by set.
- Sample items by topic and pronunciation focus tags.

All verification queries are read-only `SELECT` queries.

## Rollback Notes

If the seed draft must be rolled back before application usage, remove seed data in dependency order:

1. Delete matching rows from `public.vocabulary_set_items` for the five seed set titles.
2. Delete the five seed rows from `public.vocabulary_sets`.
3. Delete seed rows from `public.vocabulary_items` only after confirming they are not referenced by other reviewed content.

Do not run destructive rollback in a shared or production database without explicit approval.
