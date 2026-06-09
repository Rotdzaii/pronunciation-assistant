-- Vocabulary schema confirmation queries.
-- Read-only SELECT queries only.
-- Run one section at a time in the Supabase SQL Editor.

-- A. Existing table columns: profiles, practice_history, practice_history_legacy if present.
select
  c.table_schema,
  c.table_name,
  c.ordinal_position,
  c.column_name,
  c.data_type,
  c.udt_name,
  c.is_nullable,
  c.column_default
from information_schema.columns as c
where c.table_schema = 'public'
  and c.table_name in (
    'profiles',
    'practice_history',
    'practice_history_legacy'
  )
order by
  c.table_name,
  c.ordinal_position;

-- B. Foreign key constraints: profiles and practice_history.
select
  tc.table_schema,
  tc.table_name,
  tc.constraint_name,
  kcu.column_name,
  ccu.table_schema as foreign_table_schema,
  ccu.table_name as foreign_table_name,
  ccu.column_name as foreign_column_name
from information_schema.table_constraints as tc
join information_schema.key_column_usage as kcu
  on tc.constraint_schema = kcu.constraint_schema
  and tc.constraint_name = kcu.constraint_name
  and tc.table_schema = kcu.table_schema
  and tc.table_name = kcu.table_name
join information_schema.constraint_column_usage as ccu
  on tc.constraint_schema = ccu.constraint_schema
  and tc.constraint_name = ccu.constraint_name
where tc.constraint_type = 'FOREIGN KEY'
  and tc.table_schema = 'public'
  and tc.table_name in (
    'profiles',
    'practice_history'
  )
order by
  tc.table_name,
  tc.constraint_name,
  kcu.ordinal_position;

-- C. RLS status: profiles and practice_history.
select
  schemaname,
  tablename,
  rowsecurity as rls_enabled
from pg_tables
where schemaname = 'public'
  and tablename in (
    'profiles',
    'practice_history'
  )
order by
  tablename;

-- D. Existing policies: profiles and practice_history.
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
    'profiles',
    'practice_history'
  )
order by
  tablename,
  policyname;

-- E. Existing triggers: profiles and practice_history.
select
  trigger_schema,
  event_object_schema,
  event_object_table,
  trigger_name,
  event_manipulation,
  action_timing,
  action_statement
from information_schema.triggers
where event_object_schema = 'public'
  and event_object_table in (
    'profiles',
    'practice_history'
  )
order by
  event_object_table,
  trigger_name,
  event_manipulation;

-- F1. Existing routines related to job queueing, archiving, updated_at, or role helpers.
select
  routine_schema,
  routine_name,
  routine_type,
  data_type
from information_schema.routines
where routine_schema in ('public', 'auth')
  and (
    routine_name ilike '%enqueue_practice_job%'
    or routine_name ilike '%archive_practice_job%'
    or routine_name ilike '%updated_at%'
    or routine_name ilike '%update_updated_at%'
    or routine_name ilike '%handle_updated_at%'
    or routine_name ilike '%role%'
    or routine_name ilike '%teacher%'
    or routine_name ilike '%student%'
    or routine_name ilike '%admin%'
  )
order by
  routine_schema,
  routine_name;

-- F2. Existing pg_proc functions related to job queueing, archiving, updated_at, or role helpers.
select
  n.nspname as function_schema,
  p.proname as function_name,
  pg_get_function_arguments(p.oid) as arguments,
  pg_get_function_result(p.oid) as result_type
from pg_proc as p
join pg_namespace as n
  on n.oid = p.pronamespace
where n.nspname in ('public', 'auth')
  and (
    p.proname ilike '%enqueue_practice_job%'
    or p.proname ilike '%archive_practice_job%'
    or p.proname ilike '%updated_at%'
    or p.proname ilike '%update_updated_at%'
    or p.proname ilike '%handle_updated_at%'
    or p.proname ilike '%role%'
    or p.proname ilike '%teacher%'
    or p.proname ilike '%student%'
    or p.proname ilike '%admin%'
  )
order by
  n.nspname,
  p.proname;

-- G. Check whether class-related tables exist.
select
  table_schema,
  table_name,
  table_type
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'classes',
    'class_members',
    'classrooms',
    'teacher_students',
    'student_classes',
    'enrollments'
  )
order by
  table_name;
