import {
  isEligibleForUnconfirmedCleanup,
  isProtectedAdmin,
} from './policy.ts';

const now = new Date('2026-07-15T12:00:00.000Z');
const oldUnconfirmed = {
  id: 'old-user',
  email: 'old@example.com',
  created_at: '2026-07-14T11:59:59.000Z',
  email_confirmed_at: null,
  last_sign_in_at: null,
};

Deno.test('selects only an unconfirmed, never-signed-in user older than 24 hours', () => {
  if (!isEligibleForUnconfirmedCleanup(oldUnconfirmed, now)) throw new Error('expected old user to be eligible');
  if (isEligibleForUnconfirmedCleanup({ ...oldUnconfirmed, id: 'new-user', created_at: '2026-07-15T11:59:59.000Z' }, now)) throw new Error('new user must be retained');
  if (isEligibleForUnconfirmedCleanup({ ...oldUnconfirmed, id: 'confirmed', email_confirmed_at: '2026-07-14T12:00:00.000Z' }, now)) throw new Error('confirmed user must be retained');
  if (isEligibleForUnconfirmedCleanup({ ...oldUnconfirmed, id: 'signed-in', last_sign_in_at: '2026-07-14T12:00:00.000Z' }, now)) throw new Error('signed-in user must be retained');
});

Deno.test('protects admins even when their email is unconfirmed', () => {
  if (!isProtectedAdmin({ ...oldUnconfirmed, user_metadata: { app_role: 'admin' } }, null)) throw new Error('metadata admin must be protected');
  if (!isProtectedAdmin(oldUnconfirmed, 'admin')) throw new Error('profile admin must be protected');
});
