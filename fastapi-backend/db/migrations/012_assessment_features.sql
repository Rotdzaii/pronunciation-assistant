ALTER TABLE public.assignments
  ADD COLUMN is_assessment boolean NOT NULL DEFAULT false,
  ADD COLUMN deadline timestamptz,
  ADD COLUMN timer_per_word_seconds int NOT NULL DEFAULT 60
    CHECK (timer_per_word_seconds > 0 AND timer_per_word_seconds <= 3600);

CREATE TABLE public.assessment_submissions (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id uuid        NOT NULL REFERENCES public.assignments(id) ON DELETE CASCADE,
  student_id    uuid        NOT NULL REFERENCES public.profiles(id),
  recordings    jsonb       NOT NULL DEFAULT '[]'::jsonb,
  started_at    timestamptz NOT NULL DEFAULT now(),
  submitted_at  timestamptz,
  is_locked     boolean     NOT NULL DEFAULT false,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE(assignment_id, student_id)
);

CREATE INDEX assessment_submissions_assignment_id_idx ON public.assessment_submissions (assignment_id);
CREATE INDEX assessment_submissions_student_id_idx   ON public.assessment_submissions (student_id);

ALTER TABLE public.assessment_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY assessment_submissions_student_select ON public.assessment_submissions
  FOR SELECT TO authenticated USING (student_id = auth.uid());

CREATE POLICY assessment_submissions_student_insert ON public.assessment_submissions
  FOR INSERT TO authenticated WITH CHECK (student_id = auth.uid());

CREATE POLICY assessment_submissions_student_update ON public.assessment_submissions
  FOR UPDATE TO authenticated
  USING (student_id = auth.uid() AND is_locked = false)
  WITH CHECK (student_id = auth.uid());

CREATE POLICY assessment_submissions_teacher_select ON public.assessment_submissions
  FOR SELECT TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.assignments a
      WHERE a.id = assessment_submissions.assignment_id
        AND a.assigned_by = auth.uid()
    )
  );
