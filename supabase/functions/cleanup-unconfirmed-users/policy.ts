export type CleanupUser = {
  id: string;
  email?: string | null;
  created_at?: string | null;
  email_confirmed_at?: string | null;
  last_sign_in_at?: string | null;
  user_metadata?: Record<string, unknown> | null;
  app_metadata?: Record<string, unknown> | null;
};

export function isProtectedAdmin(user: CleanupUser, profileRole: string | null | undefined): boolean {
  return profileRole === 'admin'
    || user.user_metadata?.app_role === 'admin'
    || user.app_metadata?.app_role === 'admin';
}

export function isEligibleForUnconfirmedCleanup(
  user: CleanupUser,
  now: Date,
  retentionHours = 24,
): boolean {
  if (!user.email || user.email_confirmed_at || user.last_sign_in_at || !user.created_at) return false;
  const createdAt = new Date(user.created_at);
  if (Number.isNaN(createdAt.getTime())) return false;
  return createdAt.getTime() < now.getTime() - retentionHours * 60 * 60 * 1000;
}
