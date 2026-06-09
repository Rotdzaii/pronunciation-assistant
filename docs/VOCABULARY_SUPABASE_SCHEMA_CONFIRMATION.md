# Vocabulary Supabase Schema Confirmation

## Purpose

This document defines the safe confirmation process that must happen before any vocabulary database migration is written for the Pronunciation Assistant project.

The vocabulary feature is pronunciation-focused. Its database design should add vocabulary context around the existing pronunciation practice flow, not replace or duplicate the existing `practice_history` source of truth.

## Why This Confirmation Is Needed

The repository contains database migration fragments and backend usage of `profiles` and `practice_history`, but it does not contain the complete deployed Supabase schema.

The previous database relationship review found:

- `profiles` is referenced in migrations and backend auth code.
- `practice_history` is referenced in migrations and backend practice code.
- `practice_history_legacy` appears in one RLS migration.
- No class-related tables were found in the repository.
- No active RLS policy definitions were found in the repository.
- No generic `updated_at` trigger function was found in the repository.
- No SQL role helper functions were found in the repository.

Because of these gaps, creating a vocabulary migration now could choose the wrong foreign key target, conflict with deployed RLS policy style, or introduce unsafe teacher/student visibility rules.

## What Must Be Confirmed Before Migration

Before creating any vocabulary migration, confirm:

- Whether `profiles.id` references `auth.users(id)`.
- Whether `practice_history.student_id` references `auth.users(id)`, `profiles(id)`, or neither.
- Whether RLS is enabled on `profiles`.
- Whether RLS is enabled on `practice_history`.
- Existing policy style for student-owned data, teacher access, and admin access.
- Whether class-related tables already exist in Supabase.
- Whether an `updated_at` trigger function already exists.
- Whether vocabulary assignments should support `class_id` now or defer it until a class model exists.
- Whether job helper functions such as `enqueue_practice_job` and `archive_practice_job` exist in the deployed database.
- Whether role helper functions already exist and should be reused in future RLS policies.

## How To Run Read-Only SQL In Supabase

Use the query file:

```text
docs/vocabulary_schema_confirmation_queries.sql
```

Steps:

1. Open the Supabase dashboard for the project.
2. Go to SQL Editor.
3. Open `docs/vocabulary_schema_confirmation_queries.sql` locally.
4. Copy one section at a time into SQL Editor.
5. Run the section.
6. Confirm the query starts with `select` or `with` before running it.
7. Do not run any `create`, `alter`, `drop`, `insert`, `update`, `delete`, `grant`, `revoke`, or migration SQL.

The provided SQL file is intended to be read-only and uses metadata views such as:

- `information_schema.columns`
- `information_schema.table_constraints`
- `information_schema.key_column_usage`
- `information_schema.constraint_column_usage`
- `pg_tables`
- `pg_policies`
- `information_schema.triggers`
- `information_schema.routines`
- `pg_proc`
- `information_schema.tables`

## How To Paste Results Back For Review

After running the queries, paste the results back into the review thread using this structure:

```text
## A. Existing Table Columns
<paste results>

## B. Foreign Key Constraints
<paste results>

## C. RLS Status
<paste results>

## D. Existing Policies
<paste results>

## E. Existing Triggers
<paste results>

## F. Existing Functions
<paste results>

## G. Class-Related Tables
<paste results>
```

If Supabase returns an empty result for a section, paste:

```text
No rows returned.
```

Do not paste secrets, access tokens, service role keys, signed audio URLs, or `.env` values.

## Recommended Checklist

Before creating the vocabulary migration, confirm and record:

- `profiles.id` relationship.
- `practice_history.student_id` relationship.
- RLS enabled status for `profiles`.
- RLS enabled status for `practice_history`.
- Existing policy style for user-owned rows.
- Existing policy style for teacher/admin access.
- Whether class tables exist.
- Whether an `updated_at` trigger function exists.
- Whether vocabulary assignments should support `class_id` now or later.
- Whether vocabulary seed content should be system-owned with `created_by null`.
- Whether future vocabulary tables should reference `auth.users(id)` or `profiles(id)` for `created_by`, `teacher_id`, and `student_id`.

## Recommendation

Do not create the vocabulary migration until the Supabase confirmation result is reviewed.

The safest next step is to run the read-only confirmation queries, paste the results back for review, and then decide the Phase DB1 migration shape based on the deployed schema rather than repository assumptions.
