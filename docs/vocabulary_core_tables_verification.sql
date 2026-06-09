-- Vocabulary core tables verification queries.
-- Read-only SELECT queries only.
-- Run after DB1 migration is applied.

-- Verify DB1 tables exist.
select
  table_schema,
  table_name,
  table_type
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'vocabulary_items',
    'vocabulary_sets',
    'vocabulary_set_items'
  )
order by
  table_name;

-- Verify DB1 columns exist.
select
  table_schema,
  table_name,
  ordinal_position,
  column_name,
  data_type,
  udt_name,
  is_nullable,
  column_default
from information_schema.columns
where table_schema = 'public'
  and table_name in (
    'vocabulary_items',
    'vocabulary_sets',
    'vocabulary_set_items'
  )
order by
  table_name,
  ordinal_position;

-- Verify RLS is enabled.
select
  schemaname,
  tablename,
  rowsecurity as rls_enabled
from pg_tables
where schemaname = 'public'
  and tablename in (
    'vocabulary_items',
    'vocabulary_sets',
    'vocabulary_set_items'
  )
order by
  tablename;

-- Verify DB1 policies exist.
select
  schemaname,
  tablename,
  policyname,
  permissive,
  roles,
  cmd,
  qual,
  with_check
from pg_policies
where schemaname = 'public'
  and tablename in (
    'vocabulary_items',
    'vocabulary_sets',
    'vocabulary_set_items'
  )
order by
  tablename,
  policyname;

-- Verify DB1 indexes exist.
select
  schemaname,
  tablename,
  indexname,
  indexdef
from pg_indexes
where schemaname = 'public'
  and tablename in (
    'vocabulary_items',
    'vocabulary_sets',
    'vocabulary_set_items'
  )
order by
  tablename,
  indexname;
