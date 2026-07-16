import { useCallback, useEffect, useRef, useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { Link, useLocalSearchParams, useRouter } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AppButton, LoadingState, colors } from '../../components/AppUI';
import { type RoleLoadFailure, useAuth } from '../../lib/auth';
import {
  isPasswordRecoveryEvent,
  isRecoveryFlow,
  markPasswordRecoverySession,
  RESET_PASSWORD_ROUTE,
} from '../../lib/passwordRecovery';
import { supabase } from '../../lib/supabase';

type CallbackState = 'working' | 'profile_pending' | 'sign_in_required' | 'backend_unavailable' | 'invalid_recovery';

function routeForRole(role: string | null | undefined): '/(tabs)' | '/(tabs)/teacher' | '/(tabs)/admin' | null {
  if (role === 'student') return '/(tabs)';
  if (role === 'teacher') return '/(tabs)/teacher';
  if (role === 'admin') return '/(tabs)/admin';
  return null;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('callback timeout')), timeoutMs);
    promise.then(
      (value) => { clearTimeout(timeout); resolve(value); },
      (error) => { clearTimeout(timeout); reject(error); },
    );
  });
}

export default function ConfirmationCallbackScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ code?: string | string[]; flow?: string | string[] }>();
  const code = typeof params.code === 'string' ? params.code : null;
  const recoveryFlow = isRecoveryFlow(params.flow);
  const { refreshCurrentUserWithRetry } = useAuth();
  const [state, setState] = useState<CallbackState>('working');
  const processingRef = useRef(false);
  const sessionRef = useRef<Session | null>(null);
  const exchangedCodeRef = useRef<string | null>(null);
  const recoveryEventRef = useRef(false);

  const complete = useCallback(async () => {
    if (processingRef.current) return;
    processingRef.current = true;
    setState('working');

    const { data: listener } = supabase.auth.onAuthStateChange((event, nextSession) => {
      if (isPasswordRecoveryEvent(event)) {
        recoveryEventRef.current = true;
        if (nextSession) sessionRef.current = nextSession;
      }
    });

    try {
      let session = sessionRef.current;

      // PKCE codes must be exchanged before reading a stored session so an old
      // browser session cannot mask the purpose of the incoming link.
      if (code && exchangedCodeRef.current !== code) {
        exchangedCodeRef.current = code;
        const { data, error } = await withTimeout(supabase.auth.exchangeCodeForSession(code), 12_000);
        if (error || !data.session) {
          setState(recoveryFlow ? 'invalid_recovery' : 'sign_in_required');
          return;
        }
        session = data.session;
      } else if (!session) {
        const { data } = await withTimeout(supabase.auth.getSession(), 4_000);
        session = data.session;
      }

      if (recoveryFlow || recoveryEventRef.current) {
        if (!session) {
          setState('invalid_recovery');
          return;
        }
        sessionRef.current = session;
        markPasswordRecoverySession(session);
        router.replace(RESET_PASSWORD_ROUTE);
        return;
      }

      if (!session) {
        setState('sign_in_required');
        return;
      }

      sessionRef.current = session;
      const result = await refreshCurrentUserWithRetry(session.access_token, 3);
      const destination = routeForRole(result.user?.app_role);
      if (destination) {
        await new Promise<void>((resolve) => setTimeout(resolve, 350));
        router.replace(destination);
        return;
      }

      setState(result.failure === 'profile_pending' ? 'profile_pending' : failureToState(result.failure));
    } catch {
      setState(recoveryFlow ? 'invalid_recovery' : 'backend_unavailable');
    } finally {
      listener.subscription.unsubscribe();
      processingRef.current = false;
    }
  }, [code, recoveryFlow, refreshCurrentUserWithRetry, router]);

  useEffect(() => {
    void complete();
  }, [complete]);

  if (state === 'working') {
    return <SafeAreaView style={styles.safeArea}><LoadingState title="Đang xác nhận email và đăng nhập..." message="Vui lòng chờ trong giây lát." /></SafeAreaView>;
  }

  if (state === 'invalid_recovery') {
    return <CallbackMessage title="Liên kết không hợp lệ" message="Liên kết đặt lại mật khẩu không hợp lệ hoặc đã hết hạn." primaryTitle="Gửi lại liên kết" onPrimary={() => router.replace('/(auth)/forgot-password')} />;
  }

  if (state === 'profile_pending') {
    return <CallbackMessage title="Email đã được xác nhận" message="Email đã được xác nhận, nhưng hồ sơ đang được chuẩn bị. Hệ thống sẽ thử lại trong giây lát." primaryTitle="Thử lại" onPrimary={() => void complete()} />;
  }

  if (state === 'sign_in_required') {
    return <CallbackMessage title="Email đã được xác nhận thành công" message="Vui lòng đăng nhập để tiếp tục." />;
  }

  return <CallbackMessage title="Không thể kết nối đến máy chủ" message="Vui lòng thử lại." primaryTitle="Thử lại" onPrimary={() => void complete()} />;
}

function failureToState(failure: RoleLoadFailure): CallbackState {
  return failure === 'profile_pending' ? 'profile_pending' : 'backend_unavailable';
}

function CallbackMessage({ title, message, primaryTitle, onPrimary }: { title: string; message: string; primaryTitle?: string; onPrimary?: () => void }) {
  return <SafeAreaView style={styles.safeArea}><View style={styles.card}>
    <Text style={styles.title}>{title}</Text><Text style={styles.message}>{message}</Text>
    {primaryTitle && onPrimary ? <AppButton title={primaryTitle} onPress={onPrimary} /> : null}
    <Link href="/(auth)/login" style={styles.loginLink}>Đi đến trang đăng nhập</Link>
  </View></SafeAreaView>;
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background, alignItems: 'center', justifyContent: 'center', padding: 20 },
  card: { width: '100%', maxWidth: 460, backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: 20, padding: 24, gap: 14, alignItems: 'center' },
  title: { color: colors.text, fontSize: 22, fontWeight: '900', textAlign: 'center' },
  message: { color: colors.muted, fontSize: 15, lineHeight: 22, textAlign: 'center' },
  loginLink: { color: colors.primary, fontSize: 14, fontWeight: '800', marginTop: 4 },
});
