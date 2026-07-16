begin;

alter table public.profiles
drop constraint if exists profiles_app_role_check;

alter table public.profiles
add constraint profiles_app_role_check
check (app_role in ('student', 'teacher', 'admin'));

create or replace function public.handle_new_auth_user_profile()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  requested_app_role text;
begin
  requested_app_role := new.raw_user_meta_data->>'app_role';

  insert into public.profiles (id, email, app_role)
  values (
    new.id,
    new.email,
    case
      when requested_app_role in ('student', 'teacher') then requested_app_role
      else 'student'
    end
  )
  on conflict (id) do nothing;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created_create_profile on auth.users;

create trigger on_auth_user_created_create_profile
after insert on auth.users
for each row execute function public.handle_new_auth_user_profile();

-- Repair only missing rows. Existing rows (including administrators) are never
-- updated by this backfill.
insert into public.profiles (id, email, app_role)
select
  users.id,
  users.email,
  case
    when users.raw_user_meta_data->>'app_role' in ('student', 'teacher')
      then users.raw_user_meta_data->>'app_role'
    else 'student'
  end
from auth.users as users
left join public.profiles as profiles on profiles.id = users.id
where profiles.id is null
on conflict (id) do nothing;

commit;
