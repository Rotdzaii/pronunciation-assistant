alter table public.profiles
add column if not exists avatar_url text;

alter table public.profiles
add column if not exists avatar_path text;

alter table public.profiles
add column if not exists updated_at timestamptz default now();

insert into storage.buckets (id, name, public)
values ('avatars', 'avatars', true)
on conflict (id) do update
set public = excluded.public;
