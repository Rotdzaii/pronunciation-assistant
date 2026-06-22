create table if not exists public.review_requests (
  id uuid primary key default gen_random_uuid(),
  practice_history_id uuid not null references public.practice_history(id) on delete cascade,
  student_id uuid not null references public.profiles(id) on delete cascade,
  class_id uuid references public.classes(id) on delete set null,
  source text not null default 'student_report'
    check (source in ('student_report', 'system_flag')),
  reason text not null
    check (reason in (
      'ai_scored_wrong',
      'audio_issue',
      'result_mismatch',
      'low_confidence',
      'other'
    )),
  student_note text,
  severity text not null default 'medium'
    check (severity in ('low', 'medium', 'high')),
  status text not null default 'pending'
    check (status in (
      'pending',
      'in_review',
      'resolved',
      'rejected',
      'reanalyzing'
    )),
  teacher_id uuid references public.profiles(id) on delete set null,
  teacher_note text,
  teacher_resolution text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  resolved_at timestamptz
);

create index if not exists idx_review_requests_student_id
on public.review_requests(student_id);

create index if not exists idx_review_requests_class_id
on public.review_requests(class_id);

create index if not exists idx_review_requests_status
on public.review_requests(status);

create index if not exists idx_review_requests_created_at
on public.review_requests(created_at desc);

create table if not exists public.review_audit_logs (
  id uuid primary key default gen_random_uuid(),
  review_request_id uuid not null references public.review_requests(id) on delete cascade,
  actor_id uuid not null references public.profiles(id) on delete cascade,
  actor_role text not null check (actor_role in ('student', 'teacher', 'admin')),
  action_type text not null,
  before_value jsonb,
  after_value jsonb,
  reason text,
  created_at timestamptz not null default now()
);

create index if not exists idx_review_audit_logs_review_request_id
on public.review_audit_logs(review_request_id);

create index if not exists idx_review_audit_logs_created_at
on public.review_audit_logs(created_at desc);
