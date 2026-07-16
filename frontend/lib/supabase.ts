import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient } from '@supabase/supabase-js';
import { Platform } from 'react-native';
import * as Linking from 'expo-linking';

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    storage: AsyncStorage,
    persistSession: true,
    autoRefreshToken: true,
    flowType: 'pkce',
    // The callback screen exchanges the PKCE code exactly once. Letting the
    // SDK also parse it would make a Strict Mode render look like a duplicate.
    detectSessionInUrl: false,
  },
});

function getAuthCallbackUrl(): string {
  if (Platform.OS === 'web' && typeof window !== 'undefined' && window.location.origin) {
    return `${window.location.origin}/callback`;
  }

  return Linking.createURL('callback');
}

export function getEmailConfirmationRedirectUrl(): string {
  return getAuthCallbackUrl();
}

export function getPasswordRecoveryRedirectUrl(): string {
  const callbackUrl = getAuthCallbackUrl();
  return `${callbackUrl}${callbackUrl.includes('?') ? '&' : '?'}flow=recovery`;
}
