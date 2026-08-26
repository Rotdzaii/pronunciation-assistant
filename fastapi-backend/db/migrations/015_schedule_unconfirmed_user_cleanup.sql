begin;

-- Keep the profile FK safe for Auth Admin API hard-deletes. This changes the
-- current constraint only when it does not already cascade; old migrations are
-- intentionally left untouched.
do $$
declare
  profile_fk record;
begin
  select c.conname, c.confdeltype
  into profile_fk
  from pg_constraint as c
  join pg_attribute as a
    on a.attrelid = c.conrelid
    and a.attnum = any(c.conkey)
  where c.contype = 'f'
    and c.conrelid = 'public.profiles'::regclass
    and c.confrelid = 'auth.users'::regclass
    and a.attname = 'id';

  if not found then
    raise exception 'public.profiles.id must reference auth.users.id before enabling cleanup';
  end if;

  if profile_fk.confdeltype <> 'c' then
    execute format('alter table public.profiles drop constraint %I', profile_fk.conname);
    execute format(
      'alter table public.profiles add constraint %I foreign key (id) references auth.users(id) on delete cascade',
      profile_fk.conname
    );
  end if;
end;
$$;

create extension if not exists pg_cron;
create extension if not exists pg_net;

create schema if not exists private;
revoke all on schema private from public;

create or replace function private.invoke_cleanup_unconfirmed_users(dry_run boolean default true)
returns bigint
language plpgsql
security definer
set search_path = private, public, extensions, vault
as $$
declare
  request_id bigint;
begin
  select net.http_post(
    url := 'https://wlvmuuktpzficbaugwqw.supabase.co/functions/v1/cleanup-unconfirmed-users',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (
        select decrypted_secret from vault.decrypted_secrets
        where name = 'cleanup_unconfirmed_users_service_role_key'
      )
    ),
    body := jsonb_build_object('dryRun', dry_run)
  ) into request_id;

  return request_id;
end;
$$;

revoke all on function private.invoke_cleanup_unconfirmed_users(boolean) from public;

-- Store the service-role key in Supabase Vault before applying this migration:
-- cleanup_unconfirmed_users_service_role_key. It is never exposed to frontend
-- code or written into this migration.
do $$
declare
  existing_job record;
begin
  if not exists (
    select 1 from vault.decrypted_secrets
    where name = 'cleanup_unconfirmed_users_service_role_key'
  ) then
    raise exception 'Missing Vault secret cleanup_unconfirmed_users_service_role_key';
  end if;

  for existing_job in select jobid from cron.job where jobname = 'cleanup-unconfirmed-users-hourly' loop
    perform cron.unschedule(existing_job.jobid);
  end loop;

  perform cron.schedule(
    'cleanup-unconfirmed-users-hourly',
    '0 * * * *',
    $cron$
      select private.invoke_cleanup_unconfirmed_users(true);
    $cron$
  );
end;
$$;

commit;
