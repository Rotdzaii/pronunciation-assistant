# Vocabulary Core Tables Migration

## Purpose

This document describes the DB1 migration for the pronunciation-focused vocabulary feature.

DB1 creates only the core vocabulary content tables needed to store words, sets, and set membership. It does not add assignments, class relationships, practice-history links, backend logic, frontend logic, or seed data.

## Tables Created

Migration file:

```text
fastapi-backend/db/migrations/005_create_vocabulary_core_tables.sql
```

DB1 creates:

- `public.vocabulary_items`
- `public.vocabulary_sets`
- `public.vocabulary_set_items`

`vocabulary_items` stores pronunciation-focused word metadata such as `word`, `phonetic`, `meaning_vi`, `sample_sentence`, `target_phonemes`, `common_mistake_tags`, `stress_pattern`, topic, level, and difficulty.

`vocabulary_sets` stores public active word collections that can later represent topics, lessons, or pronunciation drills.

`vocabulary_set_items` links sets to items and preserves order with `sort_order`.

## Why DB1 Excludes Assignments, Class, And Practice History Links

Supabase confirmation found:

- No foreign keys on `profiles` or `practice_history` were confirmed.
- RLS is enabled on `profiles` and `practice_history`, but no active policies were confirmed.
- No class-related tables were confirmed.
- No SQL role helper functions were confirmed.
- No generic `updated_at` trigger function was confirmed.

Because of this, DB1 intentionally excludes:

- `vocabulary_assignments`
- `vocabulary_practice_history`
- `class_id`
- class tables
- foreign keys to `auth.users`
- foreign keys to `profiles`
- foreign keys to `practice_history`
- teacher/admin write policies
- role helper functions
- generic `updated_at` triggers

This keeps DB1 minimal, reversible, and focused on safe vocabulary content storage.

## RLS Policy Summary

RLS is enabled on all three DB1 tables.

Policies added:

- Authenticated users can select active `vocabulary_items`.
- Authenticated users can select active public `vocabulary_sets`.
- Authenticated users can select `vocabulary_set_items` only when the related set is active and public, and the related item is active.

No INSERT, UPDATE, or DELETE policies are added in DB1.

This means DB1 is read-only from the app side. Management, teacher, and admin write policies are deferred until the project has a reviewed RLS policy style and confirmed role-helper convention.

## Application Steps

Do not apply this migration until it has been reviewed.

When approved, apply through the project's normal Supabase migration process. Do not paste ad hoc SQL into production unless that is the agreed project workflow.

After applying, run the read-only verification queries:

```text
docs/vocabulary_core_tables_verification.sql
```

## Verification SQL

The verification file checks:

- DB1 tables exist.
- Expected columns exist.
- RLS is enabled.
- Expected SELECT policies exist.
- Expected indexes exist.

All verification queries are read-only `SELECT` queries.

## Rollback Notes

If DB1 must be rolled back before seed data or app usage exists, the core rollback concept is to drop the DB1 objects in dependency order:

- Drop `public.vocabulary_set_items`.
- Drop `public.vocabulary_sets`.
- Drop `public.vocabulary_items`.

If data has already been inserted, export or review the data first. Do not use destructive rollback in a shared or production database without approval.

## Next Phases

DB2: seed insertion.

Add reviewed seed vocabulary and initial public sets after DB1 schema is accepted.

DB3: `vocabulary_practice_history`.

Add vocabulary-to-practice linkage only after `practice_history` foreign key and RLS conventions are confirmed.

DB4: assignments.

Add `vocabulary_assignments` only after the class, teacher, and student relationship model is confirmed. Do not add `class_id` before class tables exist.
