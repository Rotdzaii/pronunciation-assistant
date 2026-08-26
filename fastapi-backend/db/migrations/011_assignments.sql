CREATE TABLE public.assignments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL,
  description text,
  content_type text NOT NULL CHECK (content_type IN ('vocabulary_set', 'sentence_set')),
  content_id uuid NOT NULL,
  class_id uuid REFERENCES public.classes(id),
  student_id uuid REFERENCES public.profiles(id),
  assigned_by uuid NOT NULL REFERENCES public.profiles(id),
  due_date timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT assignments_target_check CHECK (
    (class_id IS NOT NULL AND student_id IS NULL) OR
    (class_id IS NULL AND student_id IS NOT NULL)
  )
);

CREATE TABLE public.assignment_progress (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id uuid NOT NULL REFERENCES public.assignments(id) ON DELETE CASCADE,
  student_id uuid NOT NULL REFERENCES public.profiles(id),
  status text NOT NULL DEFAULT 'not_started' CHECK (status IN ('not_started','in_progress','completed','overdue')),
  completed_items int NOT NULL DEFAULT 0,
  total_items int NOT NULL DEFAULT 0,
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(assignment_id, student_id)
);

CREATE INDEX assignments_assigned_by_idx ON public.assignments (assigned_by);
CREATE INDEX assignments_class_id_idx ON public.assignments (class_id);
CREATE INDEX assignments_student_id_idx ON public.assignments (student_id);
CREATE INDEX assignment_progress_assignment_id_idx ON public.assignment_progress (assignment_id);
CREATE INDEX assignment_progress_student_id_idx ON public.assignment_progress (student_id);

-- Migrate existing rows from vocabulary_set_assignments into assignments
INSERT INTO public.assignments (
  title, content_type, content_id, class_id, student_id, assigned_by, created_at, updated_at
)
SELECT
  COALESCE(vs.title, 'Vocabulary Set'),
  'vocabulary_set',
  vsa.set_id,
  vsa.class_id,
  vsa.student_id,
  vsa.assigned_by,
  vsa.created_at,
  vsa.created_at
FROM public.vocabulary_set_assignments vsa
LEFT JOIN public.vocabulary_sets vs ON vs.id = vsa.set_id;

-- Create progress rows for class-based assignments
INSERT INTO public.assignment_progress (assignment_id, student_id, total_items)
SELECT
  a.id,
  sc.student_id,
  COALESCE((
    SELECT COUNT(*)::int FROM public.vocabulary_set_items vsi WHERE vsi.set_id = a.content_id
  ), 0)
FROM public.assignments a
JOIN public.student_classes sc ON sc.class_id = a.class_id AND sc.status = 'active'
WHERE a.class_id IS NOT NULL;

-- Create progress rows for individual student assignments
INSERT INTO public.assignment_progress (assignment_id, student_id, total_items)
SELECT
  a.id,
  a.student_id,
  COALESCE((
    SELECT COUNT(*)::int FROM public.vocabulary_set_items vsi WHERE vsi.set_id = a.content_id
  ), 0)
FROM public.assignments a
WHERE a.student_id IS NOT NULL;

DROP TABLE public.vocabulary_set_assignments;
