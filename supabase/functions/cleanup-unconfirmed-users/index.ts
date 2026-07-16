import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import {
  isEligibleForUnconfirmedCleanup,
  isProtectedAdmin,
  type CleanupUser,
} from './policy.ts';

const PAGE_SIZE = 200;
const RETENTION_HOURS = 24;

function json(body: Record<string, unknown>, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function chunks<T>(items: T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < items.length; index += size) result.push(items.slice(index, index + size));
  return result;
}

Deno.serve(async (request) => {
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405);

  const supabaseUrl = Deno.env.get('SUPABASE_URL');
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!supabaseUrl || !serviceRoleKey) {
    console.error('[cleanup-unconfirmed-users] missing server configuration');
    return json({ error: 'Server configuration is incomplete' }, 500);
  }

  // The function is callable only by the Cron job using the service-role key.
  if (request.headers.get('authorization') !== `Bearer ${serviceRoleKey}`) {
    return json({ error: 'Unauthorized' }, 401);
  }

  let dryRun = true;
  try {
    const body = await request.json() as { dryRun?: unknown };
    dryRun = body.dryRun !== false;
  } catch {
    // An empty body remains a safe dry run.
  }

  const supabase = createClient(supabaseUrl, serviceRoleKey, {
    auth: { autoRefreshToken: false, persistSession: false },
  });
  const now = new Date();
  const users: CleanupUser[] = [];
  let page = 1;

  try {
    while (true) {
      const { data, error } = await supabase.auth.admin.listUsers({ page, perPage: PAGE_SIZE });
      if (error) throw error;
      const currentPage = (data.users ?? []) as CleanupUser[];
      users.push(...currentPage);
      if (currentPage.length < PAGE_SIZE) break;
      page += 1;
    }

    const eligibleByAge = users.filter((user) => isEligibleForUnconfirmedCleanup(user, now, RETENTION_HOURS));
    const profileRoles = new Map<string, string | null>();
    for (const ids of chunks(eligibleByAge.map((user) => user.id), 200)) {
      if (ids.length === 0) continue;
      const { data, error } = await supabase.from('profiles').select('id, app_role').in('id', ids);
      if (error) throw error;
      for (const profile of data ?? []) profileRoles.set(profile.id, profile.app_role);
    }

    const candidates = eligibleByAge.filter((user) => !isProtectedAdmin(user, profileRoles.get(user.id)));
    let deleted = 0;
    let errors = 0;
    if (!dryRun) {
      for (const user of candidates) {
        const { error } = await supabase.auth.admin.deleteUser(user.id);
        if (error) {
          errors += 1;
          console.warn('[cleanup-unconfirmed-users] delete failed', { error: error.message });
        } else {
          deleted += 1;
        }
      }
    }

    const summary = { dryRun, checked: users.length, eligible: candidates.length, deleted, errors };
    console.info('[cleanup-unconfirmed-users] completed', summary);
    return json(summary);
  } catch (error) {
    console.error('[cleanup-unconfirmed-users] failed', {
      message: error instanceof Error ? error.message : 'Unknown cleanup error',
    });
    return json({ error: 'Cleanup could not be completed' }, 500);
  }
});
