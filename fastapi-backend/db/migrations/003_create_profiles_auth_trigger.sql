alter table public.profiles
drop constraint if exists profiles_app_role_check;

alter table public.profiles
add constraint profiles_app_role_check
check (app_role in ('student', 'teacher'));

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
  on conflict (id) do update
  set
    email = excluded.email,
    app_role = excluded.app_role,
    updated_at = now();

  return new;
end;
$$;

drop trigger if exists on_auth_user_created_create_profile on auth.users;

create trigger on_auth_user_created_create_profile
after insert on auth.users
for each row execute function public.handle_new_auth_user_profile();
