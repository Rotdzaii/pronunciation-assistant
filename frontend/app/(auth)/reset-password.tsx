import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
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
  clearPasswordRecoverySession,
  hasPasswordRecoverySession,
  type PasswordRecoveryFeedback,
  type PasswordRecoveryFieldErrors,
  updateRecoveredPassword,
} from '../../lib/passwordRecovery';
import { getPasswordRequirements } from '../../lib/register';
import { supabase } from '../../lib/supabase';

type RecoveryState = 'checking' | 'ready' | 'invalid';

export default function ResetPasswordScreen() {
  const router = useRouter();
  const [recoveryState, setRecoveryState] = useState<RecoveryState>('checking');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [confirmVisible, setConfirmVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<PasswordRecoveryFieldErrors>({});
  const [feedback, setFeedback] = useState<PasswordRecoveryFeedback | null>(null);
  const requestInProgress = useRef(false);
  const confirmInputRef = useRef<TextInput>(null);
  const requirements = getPasswordRequirements(password);

  useEffect(() => {
    let active = true;
    const verifySession = async () => {
      const { data } = await supabase.auth.getSession();
      if (!active) return;
      setRecoveryState(hasPasswordRecoverySession(data.session) ? 'ready' : 'invalid');
    };
    void verifySession();
    return () => { active = false; };
  }, []);

  const handleUpdate = async () => {
    if (loading || requestInProgress.current || recoveryState !== 'ready') return;
    requestInProgress.current = true;
    setLoading(true);
    setFieldErrors({});
    setFeedback(null);

    const result = await updateRecoveredPassword(
      password,
      confirmPassword,
      (nextPassword) => supabase.auth.updateUser({ password: nextPassword }),
    );
    if (result.fieldErrors) {
      setFieldErrors(result.fieldErrors);
    } else if (result.feedback) {
      setFeedback(result.feedback);
    } else if (result.success) {
      clearPasswordRecoverySession();
      await supabase.auth.signOut();
      router.replace('/(auth)/login?passwordReset=1');
    }

    setLoading(false);
    requestInProgress.current = false;
  };

  if (recoveryState === 'checking') {
    return <SafeAreaView style={styles.safeArea}><View style={styles.center}><Text style={styles.subtitle}>Đang kiểm tra liên kết đặt lại mật khẩu...</Text></View></SafeAreaView>;
  }

  if (recoveryState === 'invalid') {
    return (
      <SafeAreaView style={[styles.safeArea, styles.invalidSafeArea]}>
        <View style={styles.card}>
          <Text style={styles.title}>Liên kết không hợp lệ</Text>
          <Text style={styles.subtitle}>Liên kết đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.</Text>
          <Pressable accessibilityRole="button" onPress={() => router.replace('/(auth)/forgot-password')} style={styles.primaryButton}>
            <Text style={styles.primaryButtonText}>Gửi lại liên kết</Text>
          </Pressable>
          <Pressable accessibilityRole="button" onPress={() => router.replace('/(auth)/login')} style={styles.secondaryButton}>
            <Text style={styles.secondaryButtonText}>Quay lại đăng nhập</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView style={styles.keyboard} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          <View style={styles.card}>
            <Text style={styles.title}>Đặt lại mật khẩu</Text>
            <Text style={styles.subtitle}>Tạo mật khẩu mới cho tài khoản của bạn.</Text>

            <PasswordField label="Mật khẩu mới" value={password} visible={passwordVisible} onChangeText={(value) => { setPassword(value); setFieldErrors((errors) => ({ ...errors, password: undefined })); }} onToggle={() => setPasswordVisible((visible) => !visible)} onSubmitEditing={() => confirmInputRef.current?.focus()} error={fieldErrors.password} />
            <PasswordField inputRef={confirmInputRef} label="Xác nhận mật khẩu mới" value={confirmPassword} visible={confirmVisible} onChangeText={(value) => { setConfirmPassword(value); setFieldErrors((errors) => ({ ...errors, confirmPassword: undefined })); }} onToggle={() => setConfirmVisible((visible) => !visible)} onSubmitEditing={() => void handleUpdate()} error={fieldErrors.confirmPassword} returnKeyType="done" />

            <View style={styles.requirements}>
              <Requirement met={requirements.minLength} text="Ít nhất 8 ký tự" />
              <Requirement met={requirements.uppercase} text="Có chữ hoa" />
              <Requirement met={requirements.lowercase} text="Có chữ thường" />
              <Requirement met={requirements.number} text="Có ít nhất một chữ số" />
            </View>

            {feedback ? <Text style={styles.error}>{feedback.message}</Text> : null}

            <Pressable accessibilityRole="button" disabled={loading} onPress={() => void handleUpdate()} style={[styles.primaryButton, loading ? styles.disabledButton : null]}>
              <Text style={styles.primaryButtonText}>{loading ? 'Đang cập nhật...' : 'Cập nhật mật khẩu'}</Text>
            </Pressable>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function PasswordField({ inputRef, label, value, visible, onChangeText, onToggle, onSubmitEditing, error, returnKeyType = 'next' }: { inputRef?: RefObject<TextInput>; label: string; value: string; visible: boolean; onChangeText: (value: string) => void; onToggle: () => void; onSubmitEditing: () => void; error?: string; returnKeyType?: 'next' | 'done' }) {
  return <View style={styles.field}>
    <Text style={styles.label}>{label}</Text>
    <View style={styles.inputWrap}>
      <TextInput ref={inputRef} accessibilityLabel={label} autoCapitalize="none" autoCorrect={false} secureTextEntry={!visible} value={value} onChangeText={onChangeText} onSubmitEditing={onSubmitEditing} returnKeyType={returnKeyType} style={[styles.input, styles.inputWithAction, error ? styles.inputError : null]} placeholder="Tạo mật khẩu mới" placeholderTextColor="#94A3B8" />
      <Pressable accessibilityRole="button" accessibilityLabel={visible ? `Ẩn ${label.toLowerCase()}` : `Hiển thị ${label.toLowerCase()}`} onPress={onToggle} style={styles.visibilityButton}>
        <MaterialCommunityIcons name={visible ? 'eye-off-outline' : 'eye-outline'} size={22} color={colors.muted} />
      </Pressable>
    </View>
    {error ? <Text style={styles.fieldError}>{error}</Text> : null}
  </View>;
}

function Requirement({ met, text }: { met: boolean; text: string }) {
  return <Text style={[styles.requirement, met ? styles.requirementMet : null]}>{met ? '✓' : '○'} {text}</Text>;
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background },
  keyboard: { flex: 1 },
  scroll: { flexGrow: 1, justifyContent: 'center', padding: 20 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  card: { width: '100%', maxWidth: 460, alignSelf: 'center', backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: 24, padding: 24, gap: 18 },
  invalidSafeArea: { alignItems: 'center', justifyContent: 'center', padding: 20 },
  title: { color: colors.text, fontSize: 28, fontWeight: '900', lineHeight: 34 },
  subtitle: { color: colors.muted, fontSize: 15, lineHeight: 22 },
  field: { gap: 8 },
  label: { color: colors.text, fontSize: 13, fontWeight: '800' },
  inputWrap: { position: 'relative' },
  input: { minHeight: 56, borderColor: colors.border, borderWidth: 1, borderRadius: 16, paddingHorizontal: 16, color: colors.text, fontSize: 16, backgroundColor: '#FFFFFF' },
  inputWithAction: { paddingRight: 56 },
  inputError: { borderColor: colors.error },
  visibilityButton: { position: 'absolute', right: 7, top: 7, width: 42, height: 42, borderRadius: 21, alignItems: 'center', justifyContent: 'center' },
  fieldError: { color: colors.error, fontSize: 13, fontWeight: '700' },
  requirements: { gap: 5, backgroundColor: '#F8FAFC', borderRadius: 14, padding: 14 },
  requirement: { color: colors.muted, fontSize: 13, lineHeight: 19 },
  requirementMet: { color: '#166534', fontWeight: '700' },
  error: { color: '#B91C1C', fontSize: 14, lineHeight: 21, fontWeight: '700', backgroundColor: '#FEE2E2', borderRadius: 12, padding: 12 },
  primaryButton: { minHeight: 56, borderRadius: 16, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.primary, paddingHorizontal: 16 },
  disabledButton: { opacity: 0.62 },
  primaryButtonText: { color: '#FFFFFF', fontSize: 15, fontWeight: '900', textAlign: 'center' },
  secondaryButton: { minHeight: 48, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 16 },
  secondaryButtonText: { color: colors.primary, fontSize: 14, fontWeight: '800' },
});
