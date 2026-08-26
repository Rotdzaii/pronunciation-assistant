-- Read-only verification for migration 013_assignment_recipients_grades.sql.
-- Run in the development Supabase SQL editor *after* migration 013.
-- Expected: every structural query returns its documented rows; every orphan
-- and duplicate query returns zero rows.

-- Tables, constraints, indexes, RLS, and policies should all be present.
SELECT c.relname AS table_name, c.relrowsecurity AS rls_enabled
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relname IN ('assignment_recipients', 'assignment_grades')
ORDER BY c.relname;

SELECT conrelid::regclass AS table_name, conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid IN ('public.assignment_recipients'::regclass, 'public.assignment_grades'::regclass, 'public.assessment_submissions'::regclass)
ORDER BY 1, conname;

SELECT tablename, policyname, cmd
FROM pg_policies
WHERE schemaname = 'public' AND tablename IN ('assignment_recipients', 'assignment_grades')
ORDER BY tablename, policyname;

SELECT indexrelid::regclass AS index_name, pg_get_indexdef(indexrelid) AS definition
FROM pg_index
WHERE indrelid IN ('public.assignment_recipients'::regclass, 'public.assignment_grades'::regclass, 'public.practice_history'::regclass)
  AND (pg_get_indexdef(indexrelid) ILIKE '%assignment%' OR pg_get_indexdef(indexrelid) ILIKE '%assessment_submission%')
ORDER BY 1;

-- Backfill counts; recipient_count should be at least the distinct legacy
-- progress/submission pairs after migration.
SELECT
  (SELECT count(*) FROM public.assignment_recipients) AS recipient_count,
  (SELECT count(*) FROM public.assignment_grades) AS grade_count,
  (SELECT count(DISTINCT (assignment_id, student_id)) FROM public.assignment_progress) AS legacy_progress_pairs,
  (SELECT count(DISTINCT (assignment_id, student_id)) FROM public.assessment_submissions) AS legacy_submission_pairs;

-- Expected zero rows: duplicates and FK-orphan checks.
SELECT assignment_id, student_id, count(*) AS duplicate_count
FROM public.assignment_recipients
GROUP BY assignment_id, student_id HAVING count(*) > 1;

SELECT assignment_id, student_id, count(*) AS duplicate_count
FROM public.assignment_grades
GROUP BY assignment_id, student_id HAVING count(*) > 1;

SELECT assignment_id, student_id, count(*) AS duplicate_count
FROM public.assessment_submissions
GROUP BY assignment_id, student_id HAVING count(*) > 1;

SELECT r.* FROM public.assignment_recipients r
LEFT JOIN public.assignments a ON a.id = r.assignment_id
LEFT JOIN public.profiles p ON p.id = r.student_id
WHERE a.id IS NULL OR p.id IS NULL;

SELECT g.* FROM public.assignment_grades g
LEFT JOIN public.assignments a ON a.id = g.assignment_id
LEFT JOIN public.profiles p ON p.id = g.student_id
LEFT JOIN public.assessment_submissions s ON s.id = g.assessment_submission_id
WHERE a.id IS NULL OR p.id IS NULL OR (g.assessment_submission_id IS NOT NULL AND s.id IS NULL);

SELECT ph.id, ph.assignment_id, ph.item_id, ph.assessment_submission_id
FROM public.practice_history ph
LEFT JOIN public.assignments a ON a.id = ph.assignment_id
LEFT JOIN public.vocabulary_items v ON v.id = ph.item_id
LEFT JOIN public.assessment_submissions s ON s.id = ph.assessment_submission_id
WHERE (ph.assignment_id IS NOT NULL AND a.id IS NULL)
   OR (ph.item_id IS NOT NULL AND v.id IS NULL)
   OR (ph.assessment_submission_id IS NOT NULL AND s.id IS NULL);

-- Expected zero rows: the database concurrency guard for assessment words.
SELECT assessment_submission_id, item_id, count(*) AS duplicate_count
FROM public.practice_history
WHERE assessment_submission_id IS NOT NULL
GROUP BY assessment_submission_id, item_id HAVING count(*) > 1;
