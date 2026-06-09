-- DB1 vocabulary core tables.
-- Vocabulary is a pronunciation-support feature, not a general vocabulary app.
-- This migration intentionally creates only core read-side vocabulary structures.
-- Deferred: assignments, class links, practice_history links, write policies, role helpers,
-- and updated_at trigger infrastructure.

create table public.vocabulary_items (
  id uuid primary key default gen_random_uuid(),
  word text not null,
  phonetic text,
  meaning_vi text,
  topic text,
  level text,
  difficulty smallint,
  sample_sentence text,
  target_phonemes jsonb not null default '[]'::jsonb,
  common_mistake_tags jsonb not null default '[]'::jsonb,
  stress_pattern text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint vocabulary_items_word_not_blank_check
    check (length(btrim(word)) > 0),
  constraint vocabulary_items_difficulty_check
    check (difficulty is null or difficulty between 1 and 5),
  constraint vocabulary_items_target_phonemes_jsonb_check
    check (jsonb_typeof(target_phonemes) = 'array'),
  constraint vocabulary_items_common_mistake_tags_jsonb_check
    check (jsonb_typeof(common_mistake_tags) = 'array')
);

comment on table public.vocabulary_items is
  'DB1 pronunciation-focused vocabulary words. App-side writes, ownership, assignments, and practice_history links are deferred.';

comment on column public.vocabulary_items.updated_at is
  'TODO: maintain with a shared updated_at trigger only after project-wide trigger conventions are confirmed.';

create unique index vocabulary_items_lower_word_unique_idx
on public.vocabulary_items (lower(word));

create index vocabulary_items_topic_idx
on public.vocabulary_items (topic);

create index vocabulary_items_level_idx
on public.vocabulary_items (level);

create index vocabulary_items_is_active_idx
on public.vocabulary_items (is_active);

create table public.vocabulary_sets (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text,
  topic text,
  level text,
  is_public boolean not null default true,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint vocabulary_sets_title_not_blank_check
    check (length(btrim(title)) > 0)
);

comment on table public.vocabulary_sets is
  'DB1 pronunciation vocabulary sets. Teacher/admin write policies and assignment links are deferred.';

comment on column public.vocabulary_sets.updated_at is
  'TODO: maintain with a shared updated_at trigger only after project-wide trigger conventions are confirmed.';

create index vocabulary_sets_topic_idx
on public.vocabulary_sets (topic);

create index vocabulary_sets_level_idx
on public.vocabulary_sets (level);

create index vocabulary_sets_is_public_idx
on public.vocabulary_sets (is_public);

create index vocabulary_sets_is_active_idx
on public.vocabulary_sets (is_active);

create table public.vocabulary_set_items (
  id uuid primary key default gen_random_uuid(),
  set_id uuid not null references public.vocabulary_sets(id) on delete cascade,
  item_id uuid not null references public.vocabulary_items(id) on delete cascade,
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  constraint vocabulary_set_items_set_item_unique unique (set_id, item_id)
);

comment on table public.vocabulary_set_items is
  'DB1 join table for public active vocabulary sets and items. Assignment and practice-history context is deferred.';

create index vocabulary_set_items_set_id_idx
on public.vocabulary_set_items (set_id);

create index vocabulary_set_items_item_id_idx
on public.vocabulary_set_items (item_id);

create index vocabulary_set_items_set_id_sort_order_idx
on public.vocabulary_set_items (set_id, sort_order);

alter table public.vocabulary_items enable row level security;
alter table public.vocabulary_sets enable row level security;
alter table public.vocabulary_set_items enable row level security;

-- DB1 is read-only from the app side.
-- Management, teacher, and admin INSERT/UPDATE/DELETE policies are deferred until
-- the project RLS style and role-helper conventions are reviewed.
-- Assignment and practice-history links are deferred until class/user/FK conventions
-- are confirmed.

create policy vocabulary_items_select_active_authenticated
on public.vocabulary_items
for select
to authenticated
using (is_active = true);

create policy vocabulary_sets_select_active_public_authenticated
on public.vocabulary_sets
for select
to authenticated
using (
  is_active = true
  and is_public = true
);

create policy vocabulary_set_items_select_active_public_authenticated
on public.vocabulary_set_items
for select
to authenticated
using (
  exists (
    select 1
    from public.vocabulary_sets as vs
    where vs.id = vocabulary_set_items.set_id
      and vs.is_active = true
      and vs.is_public = true
  )
);
