import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'expo-router';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors } from '../../components/AppUI';
import {
  requestPasswordRecovery,
  type PasswordRecoveryFeedback,
  type PasswordRecoveryFieldErrors,
  validateRecoveryEmail,
} from '../../lib/passwordRecovery';
import { getPasswordRecoveryRedirectUrl, supabase } from '../../lib/supabase';

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [fieldErrors, setFieldErrors] = useState<PasswordRecoveryFieldErrors>({});
  const [feedback, setFeedback] = useState<PasswordRecoveryFeedback | null>(null);
  const requestInProgress = useRef(false);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setInterval(() => setCooldown((seconds) => Math.max(0, seconds - 1)), 1_000);
    return () => clearInterval(timer);
  }, [cooldown]);

  const handleSubmit = async () => {
    if (loading || cooldown > 0 || requestInProgress.current) return;

    const errors = validateRecoveryEmail(email);
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setFeedback(null);
      return;
    }

    requestInProgress.current = true;
    setLoading(true);
    setFieldErrors({});
    const result = await requestPasswordRecovery(
      email,
      getPasswordRecoveryRedirectUrl(),
      (normalizedEmail, redirectTo) => supabase.auth.resetPasswordForEmail(normalizedEmail, { redirectTo }),
    );
    setFeedback(result);
    if (result.tone === 'success') setCooldown(60);
    setLoading(false);
    requestInProgress.current = false;
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView style={styles.keyboard} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.card}>
            <Text style={styles.title}>Quên mật khẩu?</Text>
            <Text style={styles.subtitle}>
              Nhập email của bạn để nhận liên kết đặt lại mật khẩu.
            </Text>

            <View style={styles.field}>
              <Text style={styles.label}>Địa chỉ email</Text>
              <TextInput
                accessibilityLabel="Địa chỉ email để đặt lại mật khẩu"
                autoCapitalize="none"
                autoComplete="email"
                autoCorrect={false}
                keyboardType="email-address"
                placeholder="Ví dụ: tenban@gmail.com"
                placeholderTextColor="#94A3B8"
                returnKeyType="send"
                value={email}
                onChangeText={(value) => {
                  setEmail(value);
                  setFieldErrors({});
                  setFeedback(null);
                }}
                onSubmitEditing={() => void handleSubmit()}
                style={[styles.input, fieldErrors.email ? styles.inputError : null]}
              />
              {fieldErrors.email ? <Text style={styles.fieldError}>{fieldErrors.email}</Text> : null}
            </View>

            {feedback ? <Text style={feedback.tone === 'success' ? styles.success : styles.error}>{feedback.message}</Text> : null}

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Gửi liên kết đặt lại mật khẩu"
              disabled={loading || cooldown > 0}
              onPress={() => void handleSubmit()}
              style={[styles.primaryButton, (loading || cooldown > 0) ? styles.disabledButton : null]}
            >
              <Text style={styles.primaryButtonText}>
                {loading ? 'Đang gửi...' : cooldown > 0 ? `Gửi lại sau ${cooldown}s` : 'Gửi liên kết đặt lại mật khẩu'}
              </Text>
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Quay lại đăng nhập"
              onPress={() => router.replace('/(auth)/login')}
              style={styles.secondaryButton}
            >
              <Text style={styles.secondaryButtonText}>Quay lại đăng nhập</Text>
            </Pressable>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  keyboard: { flex: 1 },
  scroll: { flexGrow: 1, justifyContent: 'center', padding: 20 },
  card: { width: '100%', maxWidth: 460, alignSelf: 'center', backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: 24, padding: 24, gap: 18 },
  title: { color: colors.text, fontSize: 28, fontWeight: '900', lineHeight: 34 },
  subtitle: { color: colors.muted, fontSize: 15, lineHeight: 22 },
  field: { gap: 8 },
  label: { color: colors.text, fontSize: 13, fontWeight: '800' },
  input: { minHeight: 56, borderColor: colors.border, borderWidth: 1, borderRadius: 16, paddingHorizontal: 16, color: colors.text, fontSize: 16, backgroundColor: '#FFFFFF' },
  inputError: { borderColor: colors.error },
  fieldError: { color: colors.error, fontSize: 13, fontWeight: '700' },
  success: { color: '#166534', fontSize: 14, lineHeight: 21, fontWeight: '700', backgroundColor: '#DCFCE7', borderRadius: 12, padding: 12 },
  error: { color: '#B91C1C', fontSize: 14, lineHeight: 21, fontWeight: '700', backgroundColor: '#FEE2E2', borderRadius: 12, padding: 12 },
  primaryButton: { minHeight: 56, borderRadius: 16, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.primary, paddingHorizontal: 16 },
  disabledButton: { opacity: 0.62 },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '900', textAlign: 'center' },
  secondaryButton: { minHeight: 48, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 16 },
  secondaryButtonText: { color: colors.primary, fontSize: 14, fontWeight: '800' },
});
