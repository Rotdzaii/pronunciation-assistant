import { createContext, PropsWithChildren, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'expo-router';
import type { Session } from '@supabase/supabase-js';
import { ApiError, getMe } from './api';
import { supabase } from './supabase';
import type { AppRole, CurrentUser } from '../types';

type DemoAppRole = 'student' | 'teacher' | 'admin';
export type RoleLoadFailure = 'profile_pending' | 'backend_unavailable' | 'session_expired' | 'access_denied' | null;
export type RoleLoadResult = { user: CurrentUser | null; failure: RoleLoadFailure };

const SESSION_EXPIRED_MESSAGE = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';
const ACCESS_DENIED_MESSAGE = 'Tài khoản không có quyền truy cập khu vực này.';
const PROFILE_LOAD_MESSAGE = 'Hồ sơ tài khoản chưa sẵn sàng. Vui lòng thử lại hoặc đăng nhập lại.';

function normalizeAppRole(role: AppRole | null | undefined): DemoAppRole | null {
  return role === 'student' || role === 'teacher' || role === 'admin' ? role : null;
}

type AuthState = {
  session: Session | null;
  loading: boolean;
  roleLoading: boolean;
  accessToken: string | null;
  currentUser: CurrentUser | null;
  appRole: DemoAppRole | null;
  roleError: string | null;
  refreshCurrentUser: (accessToken?: string | null) => Promise<CurrentUser | null>;
  refreshCurrentUserWithRetry: (accessToken?: string | null, attempts?: number) => Promise<RoleLoadResult>;
  signOut: () => Promise<void>;
};

type CachedAuthSnapshot = {
  session: Session | null;
  currentUser: CurrentUser | null;
  appRole: DemoAppRole | null;
};

let cachedAuthSnapshot: CachedAuthSnapshot = {
  session: null,
  currentUser: null,
  appRole: null,
};

const AuthContext = createContext<AuthState>({
  session: null,
  loading: true,
  roleLoading: false,
  accessToken: null,
  currentUser: null,
  appRole: null,
  roleError: null,
  refreshCurrentUser: async () => null,
  refreshCurrentUserWithRetry: async () => ({ user: null, failure: 'backend_unavailable' }),
  signOut: async () => {},
});

export function AuthProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const [session, setSession] = useState<Session | null>(cachedAuthSnapshot.session);
  const [loading, setLoading] = useState(!cachedAuthSnapshot.session);
  const [roleLoading, setRoleLoading] = useState(false);
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(cachedAuthSnapshot.currentUser);
  const [appRole, setAppRole] = useState<DemoAppRole | null>(cachedAuthSnapshot.appRole);
  const [roleError, setRoleError] = useState<string | null>(null);
  const sessionRef = useRef<Session | null>(cachedAuthSnapshot.session);
  const currentUserRef = useRef<CurrentUser | null>(cachedAuthSnapshot.currentUser);
  const appRoleRef = useRef<DemoAppRole | null>(cachedAuthSnapshot.appRole);
  const profileRequestRef = useRef(0);
  const roleFailureRef = useRef<RoleLoadFailure>(null);

  const cacheAuthState = useCallback((
    nextSession: Session | null,
    nextUser: CurrentUser | null,
    nextRole: DemoAppRole | null,
  ) => {
    cachedAuthSnapshot = { session: nextSession, currentUser: nextUser, appRole: nextRole };
    sessionRef.current = nextSession;
    currentUserRef.current = nextUser;
    appRoleRef.current = nextRole;
  }, []);

  const clearBackendUser = useCallback((nextSession = sessionRef.current) => {
    setCurrentUser(null);
    setAppRole(null);
    setRoleError(null);
    setRoleLoading(false);
    roleFailureRef.current = null;
    cacheAuthState(nextSession, null, null);
  }, [cacheAuthState]);

  const updateSession = useCallback((nextSession: Session | null) => {
    const previousUserId = sessionRef.current?.user.id;
    const nextUserId = nextSession?.user.id;
    setSession(nextSession);

    if (!nextSession || previousUserId !== nextUserId) {
      clearBackendUser(nextSession);
      return;
    }

    cacheAuthState(nextSession, currentUserRef.current, appRoleRef.current);
  }, [cacheAuthState, clearBackendUser]);

  const refreshCurrentUser = useCallback(async (accessToken?: string | null) => {
    const requestId = ++profileRequestRef.current;
    const token = accessToken ?? sessionRef.current?.access_token ?? null;

    if (!token) {
      if (requestId === profileRequestRef.current) {
        clearBackendUser();
      }
      return null;
    }

    setRoleLoading(true);
    setRoleError(null);
    roleFailureRef.current = null;

    try {
      const user = await getMe(token);
      const backendRole = normalizeAppRole(user.app_role);
      if (requestId !== profileRequestRef.current) {
        return user;
      }

      if (!backendRole) {
        setCurrentUser(user);
        setAppRole(null);
        setRoleError(PROFILE_LOAD_MESSAGE);
        roleFailureRef.current = 'profile_pending';
        cacheAuthState(sessionRef.current, user, null);
        return null;
      }

      setCurrentUser(user);
      setAppRole(backendRole);
      cacheAuthState(sessionRef.current, user, backendRole);
      return user;
    } catch (err) {
      if (requestId !== profileRequestRef.current) {
        return null;
      }

      if (err instanceof ApiError && err.status === 401) {
        roleFailureRef.current = 'session_expired';
        setRoleError(SESSION_EXPIRED_MESSAGE);
        await supabase.auth.signOut();
        setSession(null);
        cacheAuthState(null, null, null);
        router.replace('/(auth)/login');
      } else if (err instanceof ApiError && err.status === 403) {
        roleFailureRef.current = 'access_denied';
        setCurrentUser(null);
        setAppRole(null);
        setRoleError(ACCESS_DENIED_MESSAGE);
        cacheAuthState(sessionRef.current, null, null);
      } else {
        roleFailureRef.current = err instanceof ApiError && err.status === 503 && /profile.*prepar/i.test(err.message)
          ? 'profile_pending'
          : 'backend_unavailable';
        setCurrentUser(null);
        setAppRole(null);
        setRoleError(PROFILE_LOAD_MESSAGE);
        cacheAuthState(sessionRef.current, null, null);
      }
      return null;
    } finally {
      if (requestId === profileRequestRef.current) {
        setRoleLoading(false);
      }
    }
  }, [cacheAuthState, clearBackendUser, router]);

  const refreshCurrentUserWithRetry = useCallback(async (
    accessToken?: string | null,
    attempts = 3,
  ): Promise<RoleLoadResult> => {
    const safeAttempts = Math.max(1, Math.min(attempts, 4));
    for (let attempt = 0; attempt < safeAttempts; attempt += 1) {
      const user = await refreshCurrentUser(accessToken);
      if (user) return { user, failure: null };

      const failure = roleFailureRef.current;
      if (failure !== 'profile_pending' || attempt === safeAttempts - 1) {
        return { user: null, failure };
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
    }
    return { user: null, failure: roleFailureRef.current };
  }, [refreshCurrentUser]);

  const signOut = useCallback(async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
    profileRequestRef.current += 1;
    setSession(null);
    clearBackendUser(null);
    router.replace('/welcome');
  }, [clearBackendUser, router]);

  useEffect(() => {
    let mounted = true;

    const loadSession = async () => {
      const { data } = await supabase.auth.getSession();
      if (!mounted) return;

      const restoredSession = data.session ?? null;
      updateSession(restoredSession);
      if (restoredSession) {
        await refreshCurrentUser(restoredSession.access_token);
      }
      if (mounted) setLoading(false);
    };

    void loadSession();

    const { data } = supabase.auth.onAuthStateChange((event, nextSession) => {
      updateSession(nextSession);

      if (event === 'SIGNED_OUT' || !nextSession) {
        profileRequestRef.current += 1;
        clearBackendUser(null);
        setLoading(false);
        return;
      }

      if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
        // Pass the event's token directly: AsyncStorage may still contain the
        // previous session during a signup/signin callback.
        void refreshCurrentUser(nextSession.access_token);
      }
      setLoading(false);
    });

    return () => {
      mounted = false;
      data.subscription.unsubscribe();
    };
  }, [clearBackendUser, refreshCurrentUser, updateSession]);

  const value = useMemo<AuthState>(() => ({
    session,
    loading,
    roleLoading,
    accessToken: session?.access_token ?? null,
    currentUser,
    appRole,
    roleError,
    refreshCurrentUser,
    refreshCurrentUserWithRetry,
    signOut,
  }), [session, loading, roleLoading, currentUser, appRole, roleError, refreshCurrentUser, refreshCurrentUserWithRetry, signOut]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
