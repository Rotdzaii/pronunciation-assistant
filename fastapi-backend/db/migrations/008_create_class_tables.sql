-- Migration 008: Create class management tables
--
-- Creates: classes, teacher_classes, student_classes, teacher_class_invitations
-- All tables use Supabase service-role access (RLS policies allow service role).

-- ============================================================
-- classes
-- ============================================================
create table if not exists public.classes (
    id          uuid        primary key default gen_random_uuid(),
    code        text        unique,
    name        text        not null,
    description text,
    status      text        not null default 'active',
    teacher_id  uuid        references public.profiles(id) on delete set null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists idx_classes_code      on public.classes(code);
create index if not exists idx_classes_teacher_id on public.classes(teacher_id);
create index if not exists idx_classes_status    on public.classes(status);

alter table public.classes enable row level security;

create policy "Service role full access on classes"
    on public.classes
    as permissive
    for all
    to service_role
    using (true)
    with check (true);

-- ============================================================
-- teacher_classes
-- ============================================================
create table if not exists public.teacher_classes (
    class_id     uuid        not null references public.classes(id)  on delete cascade,
    teacher_id   uuid        not null references public.profiles(id) on delete cascade,
    teacher_role text        not null default 'owner',
    status       text        not null default 'active',
    created_at   timestamptz not null default now(),
    primary key (class_id, teacher_id)
);

create index if not exists idx_teacher_classes_teacher_id on public.teacher_classes(teacher_id);
create index if not exists idx_teacher_classes_status     on public.teacher_classes(status);

alter table public.teacher_classes enable row level security;

create policy "Service role full access on teacher_classes"
    on public.teacher_classes
    as permissive
    for all
    to service_role
    using (true)
    with check (true);

-- ============================================================
-- student_classes
-- ============================================================
create table if not exists public.student_classes (
    class_id   uuid        not null references public.classes(id)  on delete cascade,
    student_id uuid        not null references public.profiles(id) on delete cascade,
    joined_at  timestamptz not null default now(),
    status     text        not null default 'active',
    created_at timestamptz not null default now(),
    primary key (class_id, student_id)
);

create index if not exists idx_student_classes_student_id on public.student_classes(student_id);
create index if not exists idx_student_classes_status     on public.student_classes(status);

alter table public.student_classes enable row level security;

create policy "Service role full access on student_classes"
    on public.student_classes
    as permissive
    for all
    to service_role
    using (true)
    with check (true);

-- ============================================================
-- teacher_class_invitations
-- ============================================================
create table if not exists public.teacher_class_invitations (
    id                uuid        primary key default gen_random_uuid(),
    class_id          uuid        not null references public.classes(id)  on delete cascade,
    teacher_id        uuid        references public.profiles(id) on delete set null,
    teacher_profile_id uuid       references public.profiles(id) on delete set null,
    invited_by        uuid        references public.profiles(id) on delete set null,
    teacher_role      text        not null default 'co_teacher',
    status            text        not null default 'pending',
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index if not exists idx_tci_class_id          on public.teacher_class_invitations(class_id);
create index if not exists idx_tci_teacher_id        on public.teacher_class_invitations(teacher_id);
create index if not exists idx_tci_teacher_profile_id on public.teacher_class_invitations(teacher_profile_id);
create index if not exists idx_tci_invited_by        on public.teacher_class_invitations(invited_by);

alter table public.teacher_class_invitations enable row level security;

create policy "Service role full access on teacher_class_invitations"
    on public.teacher_class_invitations
    as permissive
    for all
    to service_role
    using (true)
    with check (true);
