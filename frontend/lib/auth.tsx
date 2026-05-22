import { createContext, PropsWithChildren, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'expo-router';
import type { Session } from '@supabase/supabase-js';
import { ApiError, getMe } from './api';
import { supabase } from './supabase';
import type { AppRole, CurrentUser } from '../types';

type DemoAppRole = 'student' | 'teacher' | 'admin';

const PROFILE_MISSING_MESSAGE = 'Không tìm thấy hồ sơ người dùng. Vui lòng kiểm tra profile trong Supabase.';
const SESSION_EXPIRED_MESSAGE = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';
const ACCESS_DENIED_MESSAGE = 'Tài khoản không có quyền truy cập khu vực này.';
const PROFILE_LOAD_MESSAGE = 'Không thể tải hồ sơ người dùng.';

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
  refreshCurrentUser: () => Promise<CurrentUser | null>;
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

  const cacheAuthState = useCallback((
    nextSession: Session | null,
    nextUser: CurrentUser | null,
    nextRole: DemoAppRole | null,
  ) => {
    cachedAuthSnapshot = {
      session: nextSession,
      currentUser: nextUser,
      appRole: nextRole,
    };
    sessionRef.current = nextSession;
    currentUserRef.current = nextUser;
    appRoleRef.current = nextRole;
  }, []);

  const updateSession = useCallback((nextSession: Session | null) => {
    setSession(nextSession);
    cacheAuthState(nextSession, currentUserRef.current, appRoleRef.current);
  }, [cacheAuthState]);

  const clearBackendUser = useCallback(() => {
    setCurrentUser(null);
    setAppRole(null);
    setRoleError(null);
    setRoleLoading(false);
    cacheAuthState(sessionRef.current, null, null);
  }, [cacheAuthState]);

  const refreshCurrentUser = useCallback(async () => {
    setRoleLoading(true);
    setRoleError(null);

    try {
      const user = await getMe(sessionRef.current?.access_token ?? null);
      const backendRole = normalizeAppRole(user.app_role);

      if (!backendRole) {
        setCurrentUser(user);
        setAppRole(null);
        setRoleError(PROFILE_MISSING_MESSAGE);
        cacheAuthState(sessionRef.current, user, null);
        return null;
      }

      setCurrentUser(user);
      setAppRole(backendRole);
      cacheAuthState(sessionRef.current, user, backendRole);
      return user;
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setRoleError(SESSION_EXPIRED_MESSAGE);
        await supabase.auth.signOut();
        setSession(null);
        cacheAuthState(null, null, null);
        router.replace('/(auth)/login');
      } else if (err instanceof ApiError && err.status === 403) {
        setCurrentUser(null);
        setAppRole(null);
        setRoleError(ACCESS_DENIED_MESSAGE);
        cacheAuthState(sessionRef.current, null, null);
      } else if (err instanceof ApiError && err.status === 404) {
        setCurrentUser(null);
        setAppRole(null);
        setRoleError(PROFILE_MISSING_MESSAGE);
        cacheAuthState(sessionRef.current, null, null);
      } else {
        setRoleError(err instanceof Error ? err.message : PROFILE_LOAD_MESSAGE);
      }

      return null;
    } finally {
      setRoleLoading(false);
    }
  }, [cacheAuthState, router]);

  const signOut = useCallback(async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      throw error;
    }
    setSession(null);
    cacheAuthState(null, null, null);
    clearBackendUser();
    router.replace('/welcome');
  }, [cacheAuthState, clearBackendUser, router]);

  useEffect(() => {
    let mounted = true;

    const loadSession = async () => {
      const { data } = await supabase.auth.getSession();
      if (!mounted) {
        return;
      }

      const restoredSession = data.session ?? null;
      updateSession(restoredSession);
      if (restoredSession) {
        await refreshCurrentUser();
      } else {
        clearBackendUser();
      }

      if (mounted) {
        setLoading(false);
      }
    };

    void loadSession();

    const { data } = supabase.auth.onAuthStateChange((event, currentSession) => {
      updateSession(currentSession);

      if (event === 'SIGNED_OUT' || !currentSession) {
        clearBackendUser();
        setLoading(false);
        return;
      }

      if (event === 'SIGNED_IN') {
        if (!appRoleRef.current || currentUserRef.current?.id !== currentSession.user.id) {
          void refreshCurrentUser();
        }
        setLoading(false);
        return;
      }

      if (event === 'TOKEN_REFRESHED') {
        setLoading(false);
      }
    });

    return () => {
      mounted = false;
      data.subscription.unsubscribe();
    };
  }, [clearBackendUser, refreshCurrentUser, updateSession]);

  const value = useMemo<AuthState>(
    () => ({
      session,
      loading,
      roleLoading,
      accessToken: session?.access_token ?? null,
      currentUser,
      appRole,
      roleError,
      refreshCurrentUser,
      signOut,
    }),
    [session, loading, roleLoading, currentUser, appRole, roleError, refreshCurrentUser, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
