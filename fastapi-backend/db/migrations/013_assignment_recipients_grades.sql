-- Assignment recipients are the authoritative record of who may access an
-- assignment.  Progress and submissions remain execution/legacy records.

BEGIN;

CREATE TABLE IF NOT EXISTS public.assignment_recipients (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id uuid NOT NULL REFERENCES public.assignments(id) ON DELETE CASCADE,
  student_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  assigned_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (assignment_id, student_id)
);

CREATE INDEX IF NOT EXISTS assignment_recipients_assignment_id_idx
  ON public.assignment_recipients (assignment_id);
CREATE INDEX IF NOT EXISTS assignment_recipients_student_id_idx
  ON public.assignment_recipients (student_id);

-- Backfill all known historical recipients.  Do not delete the legacy rows;
-- they remain the source of execution state during this compatibility phase.
INSERT INTO public.assignment_recipients (assignment_id, student_id, assigned_at)
SELECT assignment_id, student_id, created_at FROM public.assignment_progress
ON CONFLICT (assignment_id, student_id) DO NOTHING;

INSERT INTO public.assignment_recipients (assignment_id, student_id, assigned_at)
SELECT assignment_id, student_id, created_at FROM public.assessment_submissions
ON CONFLICT (assignment_id, student_id) DO NOTHING;

INSERT INTO public.assignment_recipients (assignment_id, student_id, assigned_at)
SELECT id, student_id, created_at FROM public.assignments WHERE student_id IS NOT NULL
ON CONFLICT (assignment_id, student_id) DO NOTHING;

INSERT INTO public.assignment_recipients (assignment_id, student_id, assigned_at)
SELECT a.id, sc.student_id, a.created_at
FROM public.assignments a
JOIN public.student_classes sc ON sc.class_id = a.class_id AND sc.status = 'active'
WHERE a.class_id IS NOT NULL
ON CONFLICT (assignment_id, student_id) DO NOTHING;

ALTER TABLE public.assignment_progress
  DROP CONSTRAINT IF EXISTS assignment_progress_status_check;
ALTER TABLE public.assignment_progress
  ADD CONSTRAINT assignment_progress_status_check
  CHECK (status IN ('not_started','in_progress','completed','submitted','overdue','locked'));

CREATE TABLE IF NOT EXISTS public.assignment_grades (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id uuid NOT NULL REFERENCES public.assignments(id) ON DELETE CASCADE,
  student_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  assessment_submission_id uuid REFERENCES public.assessment_submissions(id) ON DELETE SET NULL,
  auto_score numeric(5,2) CHECK (auto_score IS NULL OR auto_score BETWEEN 0 AND 100),
  teacher_override_score numeric(5,2) CHECK (teacher_override_score IS NULL OR teacher_override_score BETWEEN 0 AND 100),
  final_score numeric(5,2) CHECK (final_score IS NULL OR final_score BETWEEN 0 AND 100),
  grading_status text NOT NULL DEFAULT 'pending'
    CHECK (grading_status IN ('pending','provisional','processing','graded','needs_review')),
  graded_at timestamptz,
  graded_by uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
  override_reason text,
  details jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(details) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (assignment_id, student_id),
  CHECK (teacher_override_score IS NULL OR override_reason IS NOT NULL),
  CHECK (final_score IS NULL OR final_score = COALESCE(teacher_override_score, auto_score))
);

CREATE INDEX IF NOT EXISTS assignment_grades_assignment_id_idx
  ON public.assignment_grades (assignment_id);
CREATE INDEX IF NOT EXISTS assignment_grades_student_id_idx
  ON public.assignment_grades (student_id);

INSERT INTO public.assignment_grades (assignment_id, student_id, assessment_submission_id, grading_status)
SELECT r.assignment_id, r.student_id, s.id,
  CASE WHEN a.is_assessment AND s.id IS NOT NULL AND s.submitted_at IS NOT NULL THEN 'processing' ELSE 'pending' END
FROM public.assignment_recipients r
JOIN public.assignments a ON a.id = r.assignment_id
LEFT JOIN public.assessment_submissions s
  ON s.assignment_id = r.assignment_id AND s.student_id = r.student_id
ON CONFLICT (assignment_id, student_id) DO NOTHING;

ALTER TABLE public.practice_history
  ADD COLUMN IF NOT EXISTS assignment_id uuid REFERENCES public.assignments(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS item_id uuid REFERENCES public.vocabulary_items(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS assessment_submission_id uuid REFERENCES public.assessment_submissions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS practice_history_assignment_student_idx
  ON public.practice_history (assignment_id, student_id, created_at);
CREATE INDEX IF NOT EXISTS practice_history_assignment_item_idx
  ON public.practice_history (assignment_id, item_id, status);
-- A submitted assessment counts at most one AI job per required word.  This
-- is also the database-level concurrency guard for double-taps/retries.
CREATE UNIQUE INDEX IF NOT EXISTS practice_history_assessment_submission_item_uidx
  ON public.practice_history (assessment_submission_id, item_id)
  WHERE assessment_submission_id IS NOT NULL;

ALTER TABLE public.assignment_recipients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.assignment_grades ENABLE ROW LEVEL SECURITY;

CREATE POLICY assignment_recipients_student_select ON public.assignment_recipients
  FOR SELECT TO authenticated USING (student_id = auth.uid());
CREATE POLICY assignment_recipients_teacher_select ON public.assignment_recipients
  FOR SELECT TO authenticated USING (
    EXISTS (SELECT 1 FROM public.assignments a WHERE a.id = assignment_recipients.assignment_id AND a.assigned_by = auth.uid())
  );
CREATE POLICY assignment_grades_student_select ON public.assignment_grades
  FOR SELECT TO authenticated USING (student_id = auth.uid());
CREATE POLICY assignment_grades_teacher_select ON public.assignment_grades
  FOR SELECT TO authenticated USING (
    EXISTS (SELECT 1 FROM public.assignments a WHERE a.id = assignment_grades.assignment_id AND a.assigned_by = auth.uid())
  );

COMMIT;
