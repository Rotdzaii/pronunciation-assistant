import { forwardRef, useEffect, useRef, useState } from 'react';
import type { ComponentProps, ReactNode } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Link, useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AppButton, ErrorState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';
import { resendSignupConfirmation, type ConfirmationFeedback } from '../../lib/emailConfirmation';
import {
  attemptSignup,
  createSignupRequestGate,
  getPasswordRequirements,
  getSignupRoute,
  type SignupFeedback,
  type SignupFieldErrors,
  type SignupRole,
} from '../../lib/register';
import { getEmailConfirmationRedirectUrl, supabase } from '../../lib/supabase';

export default function RegisterScreen() {
  const router = useRouter();
  const { refreshCurrentUserWithRetry } = useAuth();
  const { width } = useWindowDimensions();
  const isWide = width >= 820;
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState<SignupRole>('student');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [confirmPasswordVisible, setConfirmPasswordVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<SignupFieldErrors>({});
  const [feedback, setFeedback] = useState<SignupFeedback | null>(null);
  const [resendFeedback, setResendFeedback] = useState<ConfirmationFeedback | null>(null);
  const [resending, setResending] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [pendingVerificationEmail, setPendingVerificationEmail] = useState<string | null>(null);
  const [checkingVerification, setCheckingVerification] = useState(false);
  const [verificationMessage, setVerificationMessage] = useState<string | null>(null);
  const passwordInputRef = useRef<TextInput>(null);
  const confirmPasswordInputRef = useRef<TextInput>(null);
  const requestGateRef = useRef(createSignupRequestGate());
  const requirements = getPasswordRequirements(password);

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setInterval(() => setResendCooldown((seconds) => Math.max(0, seconds - 1)), 1000);
    return () => clearInterval(timer);
  }, [resendCooldown]);

  const clearFieldError = (field: keyof SignupFieldErrors) => {
    setFieldErrors((current) => (current[field] ? { ...current, [field]: undefined } : current));
  };

  const handleRegister = async () => {
    if (loading || !requestGateRef.current.start()) return;

    setLoading(true);
    setFeedback(null);
    setResendFeedback(null);
    setVerificationMessage(null);
    try {
      const result = await attemptSignup(
        { email, password, confirmPassword, role },
        ({ email: normalizedEmail, password: enteredPassword, role: selectedRole }) =>
          supabase.auth.signUp({
            email: normalizedEmail,
            password: enteredPassword,
            options: {
              data: { app_role: selectedRole },
              emailRedirectTo: getEmailConfirmationRedirectUrl(),
            },
          }),
      );

      if (result.status === 'validation_error') {
        setFieldErrors(result.fieldErrors);
        return;
      }
      if (result.status === 'error') {
        setFeedback(result.feedback);
        return;
      }
      if (result.status === 'confirmation_required') {
        const normalizedEmail = email.trim().toLowerCase();
        setFeedback({
          title: 'Tài khoản đã được tạo',
          message: `Chúng tôi đã gửi liên kết xác nhận đến ${normalizedEmail}. Hãy mở email và bấm Xác nhận email để hoàn tất đăng ký. Nếu chưa thấy email, hãy kiểm tra Spam, Thư rác hoặc Quảng cáo.`,
          tone: 'success',
        });
        setPendingVerificationEmail(normalizedEmail);
        setResendCooldown(60);
        return;
      }

      // Do not route based on the selected form value. The backend profile is
      // authoritative, and the new token avoids the persisted-session race.
      const profileResult = await refreshCurrentUserWithRetry(result.accessToken, 3);
      const destination = getSignupRoute(profileResult.user?.app_role);
      if (destination) {
        router.replace(destination);
      } else {
        setFeedback({
          title: 'Hồ sơ đang được chuẩn bị',
          message: 'Hồ sơ tài khoản chưa sẵn sàng. Vui lòng thử lại hoặc đăng nhập lại.',
          tone: 'error',
        });
      }
    } finally {
      setLoading(false);
      requestGateRef.current.finish();
    }
  };

  const handleResendConfirmation = async () => {
    if (!pendingVerificationEmail || resending || resendCooldown > 0) return;
    setResending(true);
    setResendFeedback(null);
    const result = await resendSignupConfirmation(
      pendingVerificationEmail,
      getEmailConfirmationRedirectUrl(),
      (normalizedEmail, redirectTo) => supabase.auth.resend({
        type: 'signup',
        email: normalizedEmail,
        options: { emailRedirectTo: redirectTo },
      }),
    );
    setResendFeedback(result);
    if (result.tone === 'success') setResendCooldown(60);
    setResending(false);
  };

  const handleVerifiedContinue = async () => {
    if (checkingVerification) return;
    setCheckingVerification(true);
    setVerificationMessage(null);
    try {
      let { data } = await supabase.auth.getSession();
      if (!data.session) {
        const refreshed = await supabase.auth.refreshSession();
        data = refreshed.data;
      }
      if (!data.session?.access_token) {
        setVerificationMessage('Chưa tìm thấy phiên đăng nhập đã xác nhận. Hãy mở liên kết mới nhất trong email hoặc đăng nhập lại.');
        return;
      }

      const profileResult = await refreshCurrentUserWithRetry(data.session.access_token, 3);
      const destination = getSignupRoute(profileResult.user?.app_role);
      if (destination) {
        router.replace(destination);
      } else {
        setVerificationMessage('Email đã được xác nhận, nhưng hồ sơ đang được chuẩn bị. Vui lòng thử lại trong giây lát hoặc đăng nhập lại.');
      }
    } catch {
      setVerificationMessage('Chưa thể kiểm tra trạng thái xác nhận. Vui lòng thử lại hoặc đăng nhập lại.');
    } finally {
      setCheckingVerification(false);
    }
  };

  const handleUseDifferentEmail = () => {
    setFeedback(null);
    setPendingVerificationEmail(null);
    setResendFeedback(null);
    setVerificationMessage(null);
    setResendCooldown(0);
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setFieldErrors({});
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.backgroundCircleOne} />
      <View style={styles.backgroundCircleTwo} />
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        <View style={[styles.authCard, isWide ? styles.authCardWide : null]}>
          <BrandPanel compact={!isWide} />
          <View style={[styles.formPanel, isWide ? styles.formPanelWide : null]}>
            <View style={styles.header}>
              <Text style={styles.title}>Tạo tài khoản mới</Text>
              <Text style={styles.subtitle}>Bắt đầu luyện phát âm với phản hồi bằng tiếng Việt.</Text>
            </View>

            <View style={styles.roleSection}>
              <Text style={styles.roleTitle}>Bạn đăng ký với vai trò</Text>
              <Text style={styles.roleDescription}>Chọn Học viên để luyện tập hoặc Giáo viên để quản lý lớp học.</Text>
              <RoleSelector value={role} onChange={setRole} />
            </View>

            <View style={styles.form}>
              <LabeledInput
                label="Địa chỉ email"
                placeholder="Ví dụ: tenban@gmail.com"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="email-address"
                textContentType="emailAddress"
                value={email}
                onChangeText={(value) => { setEmail(value); clearFieldError('email'); }}
                error={fieldErrors.email}
                returnKeyType="next"
                blurOnSubmit={false}
                onSubmitEditing={() => passwordInputRef.current?.focus()}
              />
              <LabeledInput
                ref={passwordInputRef}
                label="Mật khẩu"
                placeholder="Tạo mật khẩu"
                secureTextEntry={!passwordVisible}
                textContentType="newPassword"
                value={password}
                onChangeText={(value) => { setPassword(value); clearFieldError('password'); }}
                error={fieldErrors.password}
                returnKeyType="next"
                blurOnSubmit={false}
                onSubmitEditing={() => confirmPasswordInputRef.current?.focus()}
                rightElement={<VisibilityButton visible={passwordVisible} onPress={() => setPasswordVisible((visible) => !visible)} label="mật khẩu" />}
              />
              <PasswordRequirements requirements={requirements} />
              <LabeledInput
                ref={confirmPasswordInputRef}
                label="Xác nhận mật khẩu"
                placeholder="Nhập lại mật khẩu"
                secureTextEntry={!confirmPasswordVisible}
                textContentType="newPassword"
                value={confirmPassword}
                onChangeText={(value) => { setConfirmPassword(value); clearFieldError('confirmPassword'); }}
                error={fieldErrors.confirmPassword}
                returnKeyType="done"
                onSubmitEditing={handleRegister}
                rightElement={<VisibilityButton visible={confirmPasswordVisible} onPress={() => setConfirmPasswordVisible((visible) => !visible)} label="xác nhận mật khẩu" />}
              />
            </View>

            {feedback?.tone === 'error' ? <ErrorState title={feedback.title} message={feedback.message} /> : null}
            {feedback?.tone === 'success' ? <View style={styles.successNotice}>
              <Text style={styles.successTitle}>{feedback.title}</Text>
              <Text style={styles.successMessage}>{feedback.message}</Text>
              <AppButton
                title={resending ? 'Đang gửi...' : resendCooldown > 0 ? `Gửi lại email xác nhận (${resendCooldown}s)` : 'Gửi lại email xác nhận'}
                variant="secondary"
                onPress={handleResendConfirmation}
                loading={resending}
                disabled={resending || resendCooldown > 0}
              />
              {resendFeedback ? <Text style={resendFeedback.tone === 'success' ? styles.resendSuccess : styles.resendError}>{resendFeedback.message}</Text> : null}
              <AppButton title="Tôi đã xác nhận – tiếp tục" onPress={handleVerifiedContinue} loading={checkingVerification} disabled={checkingVerification} />
              {verificationMessage ? <Text style={styles.resendError}>{verificationMessage}</Text> : null}
              <AppButton title="Dùng email khác" variant="secondary" onPress={handleUseDifferentEmail} />
              <AppButton title="Đi đến trang đăng nhập" variant="secondary" onPress={() => router.replace('/(auth)/login')} />
            </View> : null}

            <AppButton title="Đăng ký" onPress={handleRegister} loading={loading} disabled={loading || feedback?.tone === 'success'} />

            <View style={styles.footerRow}>
              <Text style={styles.footerText}>Đã có tài khoản?</Text>
              <Link href="/(auth)/login" style={styles.footerLink}>Đăng nhập</Link>
            </View>
            <View style={styles.footerRow}><Link href="/welcome" style={styles.footerLink}>Quay lại trang giới thiệu</Link></View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function BrandPanel({ compact }: { compact: boolean }) {
  return <View style={[styles.brandPanel, compact ? styles.brandPanelCompact : styles.brandPanelWide]}>
    <View style={styles.illustrationGlowOne} /><View style={styles.illustrationGlowTwo} />
    <View style={styles.brandContent}>
      <View style={styles.audioIcon}><View style={styles.audioBars}>{[18, 30, 42, 26, 34].map((height, index) => <View key={`${height}-${index}`} style={[styles.audioBar, { height }]} />)}</View></View>
      <Text style={styles.brandName}>Pronunciation Assistant</Text>
      <Text style={styles.brandSubtitle}>Trợ lý AI giúp bạn luyện phát âm tiếng Anh tự tin hơn.</Text>
    </View>
  </View>;
}

function RoleSelector({ value, onChange }: { value: SignupRole; onChange: (role: SignupRole) => void }) {
  return <View style={styles.roleGroup}>
    <RoleOption label="Học viên" active={value === 'student'} onPress={() => onChange('student')} />
    <RoleOption label="Giáo viên" active={value === 'teacher'} onPress={() => onChange('teacher')} />
  </View>;
}

function RoleOption({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return <Pressable accessibilityRole="button" accessibilityState={{ selected: active }} onPress={onPress} style={[styles.roleOption, active ? styles.roleOptionActive : null]}><Text style={[styles.roleText, active ? styles.roleTextActive : null]}>{label}</Text></Pressable>;
}

function VisibilityButton({ visible, onPress, label }: { visible: boolean; onPress: () => void; label: string }) {
  return <Pressable accessibilityRole="button" accessibilityLabel={visible ? `Ẩn ${label}` : `Hiển thị ${label}`} onPress={onPress} style={styles.visibilityButton} hitSlop={8}>
    <MaterialCommunityIcons name={visible ? 'eye-off-outline' : 'eye-outline'} size={22} color={colors.muted} />
  </Pressable>;
}

function PasswordRequirements({ requirements }: { requirements: ReturnType<typeof getPasswordRequirements> }) {
  const items: Array<[keyof typeof requirements, string]> = [['minLength', 'Ít nhất 8 ký tự'], ['uppercase', 'Có chữ hoa'], ['lowercase', 'Có chữ thường'], ['number', 'Có ít nhất một chữ số']];
  return <View accessibilityLabel="Yêu cầu mật khẩu" style={styles.requirements}>{items.map(([key, label]) => <View key={key} style={styles.requirementRow}><MaterialCommunityIcons name={requirements[key] ? 'check-circle' : 'circle-outline'} size={16} color={requirements[key] ? colors.success : colors.muted} /><Text style={[styles.requirementText, requirements[key] ? styles.requirementMet : null]}>{label}</Text></View>)}</View>;
}

const LabeledInput = forwardRef<TextInput, ComponentProps<typeof TextInput> & { label: string; error?: string; rightElement?: ReactNode }>(({ label, error, rightElement, style, ...inputProps }, ref) => <View style={styles.field}>
  <Text style={styles.inputLabel}>{label}</Text>
  <View><TextInput ref={ref} {...inputProps} style={[styles.input, rightElement ? styles.inputWithAction : null, error ? styles.inputError : null, style]} placeholderTextColor="#94A3B8" />{rightElement ? <View style={styles.inputAction}>{rightElement}</View> : null}</View>
  {error ? <Text style={styles.fieldError}>{error}</Text> : null}
</View>);
LabeledInput.displayName = 'LabeledInput';

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: colors.background }, backgroundCircleOne: { position: 'absolute', width: 260, height: 260, borderRadius: 130, backgroundColor: '#DBEAFE', opacity: 0.55, top: -90, left: -80 }, backgroundCircleTwo: { position: 'absolute', width: 280, height: 280, borderRadius: 140, backgroundColor: '#CCFBF1', opacity: 0.48, right: -90, bottom: -110 },
  scrollContent: { flexGrow: 1, alignItems: 'center', justifyContent: 'center', padding: 20 }, authCard: { width: '100%', maxWidth: 1100, backgroundColor: colors.surface, borderColor: colors.border, borderRadius: 24, borderWidth: 1, overflow: 'hidden', shadowColor: colors.text, shadowOffset: { width: 0, height: 18 }, shadowOpacity: 0.08, shadowRadius: 30, elevation: 8 }, authCardWide: { minHeight: 700, flexDirection: 'row' },
  brandPanel: { minHeight: 250, backgroundColor: '#EEF2FF', overflow: 'hidden', alignItems: 'center', justifyContent: 'center', padding: 30 }, brandPanelWide: { width: '42%', minHeight: '100%' }, brandPanelCompact: { minHeight: 220 }, illustrationGlowOne: { position: 'absolute', width: 220, height: 220, borderRadius: 110, backgroundColor: '#DBEAFE', top: -70, right: -70 }, illustrationGlowTwo: { position: 'absolute', width: 180, height: 180, borderRadius: 90, backgroundColor: '#CCFBF1', left: -48, bottom: -60 }, brandContent: { alignItems: 'center', maxWidth: 320, gap: 14 }, audioIcon: { width: 86, height: 86, borderRadius: 26, backgroundColor: colors.surface, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: '#BFDBFE' }, audioBars: { height: 46, flexDirection: 'row', alignItems: 'center', gap: 5 }, audioBar: { width: 7, borderRadius: 999, backgroundColor: colors.primary }, brandName: { color: colors.text, fontSize: 26, fontWeight: '900', textAlign: 'center', lineHeight: 32 }, brandSubtitle: { color: colors.muted, fontSize: 15, lineHeight: 22, textAlign: 'center' },
  formPanel: { padding: 24, gap: 18 }, formPanelWide: { flex: 1, paddingHorizontal: 48, paddingVertical: 56, justifyContent: 'center' }, header: { gap: 8 }, title: { color: colors.text, fontSize: 30, fontWeight: '900', lineHeight: 36 }, subtitle: { color: colors.muted, fontSize: 15, lineHeight: 22 }, roleSection: { gap: 7 }, roleTitle: { color: colors.text, fontSize: 14, fontWeight: '800' }, roleDescription: { color: colors.muted, fontSize: 13, lineHeight: 19 },
  roleGroup: { flexDirection: 'row', backgroundColor: colors.background, borderColor: colors.border, borderRadius: 16, borderWidth: 1, padding: 4, gap: 4 }, roleOption: { flex: 1, minHeight: 46, borderRadius: 13, alignItems: 'center', justifyContent: 'center' }, roleOptionActive: { backgroundColor: colors.primary }, roleText: { color: colors.muted, fontSize: 14, fontWeight: '800' }, roleTextActive: { color: '#FFFFFF' }, form: { gap: 14 }, field: { gap: 8 }, inputLabel: { color: colors.text, fontSize: 13, fontWeight: '800' }, input: { minHeight: 56, borderWidth: 1, borderColor: colors.border, borderRadius: 16, paddingHorizontal: 16, backgroundColor: '#FFFFFF', color: colors.text, fontSize: 15 }, inputWithAction: { paddingRight: 58 }, inputError: { borderColor: colors.error }, inputAction: { position: 'absolute', right: 8, top: 0, bottom: 0, justifyContent: 'center' }, visibilityButton: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' }, fieldError: { color: colors.error, fontSize: 12, fontWeight: '700' },
  requirements: { gap: 5, marginTop: -4 }, requirementRow: { flexDirection: 'row', alignItems: 'center', gap: 7 }, requirementText: { color: colors.muted, fontSize: 12 }, requirementMet: { color: '#15803D', fontWeight: '700' }, successNotice: { borderWidth: 1, borderColor: '#86EFAC', backgroundColor: '#F0FDF4', borderRadius: 14, padding: 14, gap: 4 }, successTitle: { color: '#166534', fontSize: 14, fontWeight: '900' }, successMessage: { color: '#166534', fontSize: 13, lineHeight: 19 },
  resendSuccess: { color: '#166534', fontSize: 13, fontWeight: '700' }, resendError: { color: '#B91C1C', fontSize: 13, fontWeight: '700' },
  footerRow: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 5 }, footerText: { color: colors.muted, fontSize: 14, fontWeight: '700' }, footerLink: { color: colors.primary, fontSize: 14, fontWeight: '900' },
});
