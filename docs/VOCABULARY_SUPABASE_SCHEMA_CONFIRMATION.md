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

## Confirmed Supabase Results

These results were confirmed by running the read-only schema confirmation queries against the deployed Supabase project.

### Existing Columns

`public.profiles`:

- `id uuid not null`
- `email text nullable`
- `app_role text not null default 'student'`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

`public.practice_history`:

- `id uuid not null`
- `student_id uuid not null`
- `target_word text not null`
- `audio_url text not null`
- `status text not null default 'processing'`
- `score double precision nullable`
- `problem_phonemes jsonb not null default '[]'`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`
- `feedback jsonb not null default '{}'`

`public.practice_history_legacy`:

- `id bigint not null`
- `created_at timestamptz not null default now()`
- `target_word text nullable`
- `overall_score double precision nullable`
- `phoneme_details jsonb nullable`

### Foreign Keys

The foreign key query for `public.profiles` and `public.practice_history` returned no rows.

Confirmed interpretation:

- No foreign key constraint was confirmed for `profiles.id`.
- No foreign key constraint was confirmed for `practice_history.student_id`.
- The target identity relationship is still a convention, not a confirmed database constraint.

### RLS Status

Confirmed RLS status:

- `public.profiles`: `rls_enabled = true`
- `public.practice_history`: `rls_enabled = true`

### Existing Policies

The existing policies query for `public.profiles` and `public.practice_history` returned no rows.

Confirmed interpretation:

- No active policy was confirmed in `pg_policies` for `public.profiles`.
- No active policy was confirmed in `pg_policies` for `public.practice_history`.
- RLS is enabled, but no policy style was confirmed for these tables.

### Class-Related Tables

The class-related table query returned no rows.

No class-related table was confirmed for:

- `classes`
- `class_members`
- `classrooms`
- `teacher_students`
- `student_classes`
- `enrollments`

### Functions

Confirmed functions:

- `auth.role()`
- `public.archive_practice_job(p_msg_id bigint)`
- `public.enqueue_practice_job(p_job_id uuid, p_student_id uuid, p_target_word text, p_audio_url text)`

Not confirmed:

- Generic `updated_at` trigger function.
- SQL role helper function for app roles such as student, teacher, or admin.

## Recommendation

Do not create `vocabulary_assignments` yet because no class-related table exists.

Do not create `class_id` yet.

Do not create `vocabulary_practice_history` with a foreign key to `practice_history` yet. `practice_history.student_id` and `profiles.id` have no confirmed foreign key constraints, and RLS is enabled without confirmed active policies. The FK/RLS conventions need a separate review before linking vocabulary attempt history to existing pronunciation attempt history.

DB1 should only create the core vocabulary tables:

- `vocabulary_items`
- `vocabulary_sets`
- `vocabulary_set_items`

DB1 should avoid foreign keys to `auth.users`, `profiles`, or `practice_history`. Keep ownership and attempt-history links out of DB1 until identity and RLS conventions are confirmed.

DB1 should be minimal and reversible:

- Store pronunciation-focused vocabulary metadata.
- Store vocabulary set metadata.
- Store set-to-item membership.
- Use soft activation fields such as `is_active` instead of relying on destructive deletes.
- Avoid teacher assignment, class assignment, and practice-history linkage.

RLS policy design needs a separate review because RLS is enabled on existing tables, but no policies were confirmed. Any vocabulary RLS design should be aligned with the final project policy style before migration SQL is created.

Do not create the vocabulary migration until this confirmed schema result is reviewed and the DB1 table boundaries are accepted.
