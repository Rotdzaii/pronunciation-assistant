import { forwardRef, useEffect, useRef, useState } from 'react';
import type { ComponentProps, ReactNode } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Link, useLocalSearchParams, useRouter } from 'expo-router';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AppButton, ErrorState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';
import { resendSignupConfirmation, type ConfirmationFeedback } from '../../lib/emailConfirmation';
import {
  attemptLogin,
  createLoginRequestGate,
  isEmailNotConfirmedError,
  type LoginFeedback,
  type LoginFieldErrors,
  validateLoginCredentials,
} from '../../lib/login';
import { FORGOT_PASSWORD_ROUTE } from '../../lib/passwordRecovery';
import { getEmailConfirmationRedirectUrl, supabase } from '../../lib/supabase';

export default function LoginScreen() {
  const router = useRouter();
  const { passwordReset } = useLocalSearchParams<{ passwordReset?: string | string[] }>();
  const { refreshCurrentUserWithRetry } = useAuth();
  const { width } = useWindowDimensions();
  const isWide = width >= 860;
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [remember, setRemember] = useState(true);
  const [loading, setLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<LoginFieldErrors>({});
  const [loginFeedback, setLoginFeedback] = useState<LoginFeedback | null>(null);
  const [profilePreparing, setProfilePreparing] = useState(false);
  const [unconfirmedEmail, setUnconfirmedEmail] = useState<string | null>(null);
  const [resending, setResending] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendFeedback, setResendFeedback] = useState<ConfirmationFeedback | null>(null);
  const passwordInputRef = useRef<TextInput>(null);
  const requestGateRef = useRef(createLoginRequestGate());

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setInterval(() => setResendCooldown((seconds) => Math.max(0, seconds - 1)), 1000);
    return () => clearInterval(timer);
  }, [resendCooldown]);

  const clearFieldError = (field: keyof LoginFieldErrors) => {
    setFieldErrors((currentErrors) => {
      if (!currentErrors[field]) {
        return currentErrors;
      }

      return { ...currentErrors, [field]: undefined };
    });
  };

  const handleEmailChange = (value: string) => {
    setEmail(value);
    setUnconfirmedEmail(null);
    setResendFeedback(null);
    clearFieldError('email');
  };

  const handlePasswordChange = (value: string) => {
    setPassword(value);
    clearFieldError('password');
  };

  const handleLogin = async () => {
    if (loading || !requestGateRef.current.start()) {
      return;
    }

    const validationErrors = validateLoginCredentials({ email, password });
    if (Object.keys(validationErrors).length > 0) {
      setFieldErrors(validationErrors);
      setLoginFeedback(null);
      requestGateRef.current.finish();
      return;
    }

    setLoading(true);
    setFieldErrors({});
    setLoginFeedback(null);
    setProfilePreparing(false);
    setUnconfirmedEmail(null);
    setResendFeedback(null);
    try {
      let signInAccessToken: string | null = null;
      let emailRequiresConfirmation = false;
      const result = await attemptLogin(
        { email, password },
        (credentials) => supabase.auth.signInWithPassword(credentials),
        ({ data, error }) => {
          signInAccessToken = data?.session?.access_token ?? null;
          emailRequiresConfirmation = isEmailNotConfirmedError(error);
          if (__DEV__) {
            const authError = error as { name?: unknown; code?: unknown; status?: unknown; message?: unknown } | null | undefined;
            console.debug('[Signin] result', {
              hasSession: Boolean(data?.session),
              hasUser: Boolean(data?.user),
              errorName: typeof authError?.name === 'string' ? authError.name : undefined,
              errorCode: typeof authError?.code === 'string' ? authError.code : undefined,
              errorStatus: typeof authError?.status === 'number' ? authError.status : undefined,
              errorMessage: typeof authError?.message === 'string' ? authError.message : undefined,
            });
          }
        },
      );

      if (result.status === 'error') {
        setLoginFeedback(result.feedback);
        if (emailRequiresConfirmation) {
          setUnconfirmedEmail(email.trim().toLowerCase());
        }
        return;
      }

      if (result.status === 'validation_error') {
        setFieldErrors(result.fieldErrors);
        return;
      }

      setProfilePreparing(true);
      const { data } = await supabase.auth.getSession();
      const profileResult = await refreshCurrentUserWithRetry(
        signInAccessToken ?? data.session?.access_token ?? null,
        3,
      );
      setProfilePreparing(false);
      const backendRole = profileResult.user?.app_role;
      if (backendRole === 'admin') {
        router.replace('/(tabs)/admin');
      } else if (backendRole === 'teacher') {
        router.replace('/(tabs)/teacher');
      } else if (backendRole === 'student') {
        router.replace('/(tabs)');
      } else if (profileResult.failure === 'profile_pending') {
        setLoginFeedback({
          title: 'Đăng nhập thành công',
          message: 'Hồ sơ đang được chuẩn bị, vui lòng chờ trong giây lát.',
        });
      } else if (profileResult.failure === 'backend_unavailable') {
        setLoginFeedback({
          title: 'Đã xác thực tài khoản',
          message: 'Đã xác thực tài khoản nhưng chưa thể kết nối đến máy chủ. Vui lòng thử lại.',
        });
      } else if (profileResult.failure === 'session_expired') {
        setLoginFeedback({
          title: 'Phiên đăng nhập đã hết hạn',
          message: 'Vui lòng đăng nhập lại.',
        });
      } else {
        setLoginFeedback({
          title: 'Không thể đăng nhập',
          message: 'Đã xảy ra lỗi. Vui lòng thử lại sau.',
        });
      }
    } finally {
      setProfilePreparing(false);
      setLoading(false);
      requestGateRef.current.finish();
    }
  };

  const handleResendConfirmation = async () => {
    if (!unconfirmedEmail || resending || resendCooldown > 0) return;
    setResending(true);
    setResendFeedback(null);
    const result = await resendSignupConfirmation(
      unconfirmedEmail,
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

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.pageGlowBlue} />
      <View style={styles.pageGlowTeal} />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={[styles.authShell, isWide ? styles.authShellWide : null]}>
          <BrandPanel wide={isWide} />

          <View style={[styles.formPanel, isWide ? styles.formPanelWide : null]}>
            <View style={styles.formInner}>
              <View style={styles.header}>
                <Text style={styles.title}>Chào mừng trở lại</Text>
                <Text style={styles.subtitle}>Vui lòng nhập thông tin để đăng nhập.</Text>
              </View>

              <View style={styles.form}>
                <LabeledInput
                  label="Địa chỉ Email"
                  placeholder="Nhập email của bạn"
                  autoCapitalize="none"
                  keyboardType="email-address"
                  value={email}
                  onChangeText={handleEmailChange}
                  error={fieldErrors.email}
                  returnKeyType="next"
                  blurOnSubmit={false}
                  onSubmitEditing={() => passwordInputRef.current?.focus()}
                />
                <LabeledInput
                  ref={passwordInputRef}
                  label="Mật khẩu"
                  placeholder="Nhập mật khẩu"
                  secureTextEntry={!passwordVisible}
                  value={password}
                  onChangeText={handlePasswordChange}
                  error={fieldErrors.password}
                  returnKeyType="done"
                  onSubmitEditing={handleLogin}
                  rightElement={
                    <Pressable
                      accessibilityRole="button"
                      accessibilityLabel={passwordVisible ? 'Ẩn mật khẩu' : 'Hiển thị mật khẩu'}
                      onPress={() => setPasswordVisible((visible) => !visible)}
                      style={styles.visibilityButton}
                    >
                      <MaterialCommunityIcons
                        name={passwordVisible ? 'eye-off-outline' : 'eye-outline'}
                        size={22}
                        color={colors.muted}
                      />
                    </Pressable>
                  }
                />
              </View>

              <View style={styles.metaRow}>
                <Pressable style={styles.rememberRow} onPress={() => setRemember((value) => !value)}>
                  <View style={[styles.checkbox, remember ? styles.checkboxActive : null]}>
                    {remember ? <View style={styles.checkboxMark} /> : null}
                  </View>
                  <Text style={styles.metaText}>Ghi nhớ đăng nhập</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Quên mật khẩu"
                  onPress={() => router.push(FORGOT_PASSWORD_ROUTE)}
                  style={styles.forgotButton}
                >
                  <Text style={styles.forgotLink}>Quên mật khẩu?</Text>
                </Pressable>
              </View>

              {passwordReset === '1' ? (
                <View style={styles.passwordResetNotice}>
                  <Text style={styles.passwordResetNoticeText}>Mật khẩu đã được cập nhật. Vui lòng đăng nhập bằng mật khẩu mới.</Text>
                </View>
              ) : null}
              {loginFeedback ? (
                <ErrorState title={loginFeedback.title} message={loginFeedback.message} />
              ) : null}
              {unconfirmedEmail ? (
                <View style={styles.confirmationNotice}>
                  <Text style={styles.confirmationTitle}>Email chưa được xác nhận</Text>
                  <Text style={styles.confirmationText}>Tài khoản đã được tạo nhưng email chưa được xác nhận. Vui lòng kiểm tra hộp thư hoặc gửi lại email xác nhận.</Text>
                  <AppButton
                    title={resending ? 'Đang gửi...' : resendCooldown > 0 ? `Gửi lại email xác nhận (${resendCooldown}s)` : 'Gửi lại email xác nhận'}
                    variant="secondary"
                    onPress={handleResendConfirmation}
                    loading={resending}
                    disabled={resending || resendCooldown > 0}
                  />
                  {resendFeedback ? <Text style={resendFeedback.tone === 'success' ? styles.confirmationSuccess : styles.confirmationError}>{resendFeedback.message}</Text> : null}
                </View>
              ) : null}
              {profilePreparing ? (
                <View style={styles.profilePreparingNotice}>
                  <Text style={styles.profilePreparingTitle}>Đăng nhập thành công</Text>
                  <Text style={styles.profilePreparingText}>Hồ sơ đang được chuẩn bị, vui lòng chờ trong giây lát.</Text>
                </View>
              ) : null}

              <Pressable
                accessibilityRole="button"
                style={[styles.submitButton, loading ? styles.submitButtonDisabled : null]}
                onPress={handleLogin}
                disabled={loading}
              >
                <Text style={styles.submitText}>{loading ? 'Đang đăng nhập...' : 'Đăng nhập'}</Text>
              </Pressable>

              <View style={styles.footerRow}>
                <Text style={styles.footerText}>Chưa có tài khoản?</Text>
                <Link href="/(auth)/register" style={styles.footerLink}>
                  Đăng ký ngay
                </Link>
              </View>
              <View style={styles.footerRow}>
                <Link href="/welcome" style={styles.footerLink}>
                  Quay lại trang giới thiệu
                </Link>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function BrandPanel({ wide }: { wide: boolean }) {
  return (
    <View style={[styles.brandPanel, wide ? styles.brandPanelWide : null]}>
      <View style={styles.panelCircleTop} />
      <View style={styles.panelCircleBottom} />
      <View style={styles.floatCardTop}>
        <View style={styles.floatLineLong} />
        <View style={styles.floatLineShort} />
      </View>
      <View style={styles.floatCardBottom}>
        <View style={styles.miniDot} />
        <View style={styles.miniDotMuted} />
        <View style={styles.miniDotTeal} />
      </View>

      <View style={styles.illustrationCenter}>
        <View style={styles.iconTile}>
          <View style={styles.waveBars}>
            {[18, 30, 44, 28, 36].map((height, index) => (
              <View key={`${height}-${index}`} style={[styles.waveBar, { height }]} />
            ))}
          </View>
        </View>
        <Text style={styles.brandName}>Pronunciation Assistant</Text>
        <Text style={styles.brandSubtitle}>
          Trợ lý AI giúp bạn luyện phát âm tiếng Anh tự tin hơn.
        </Text>
      </View>
    </View>
  );
}

const LabeledInput = forwardRef<TextInput, ComponentProps<typeof TextInput> & {
  label: string;
  error?: string;
  rightElement?: ReactNode;
}>(
  function LabeledInput(props, ref) {
  const { label, error, rightElement, style, ...inputProps } = props;

  return (
    <View style={styles.field}>
      <Text style={styles.inputLabel}>{label}</Text>
      <View style={styles.inputWrap}>
        <TextInput
          ref={ref}
          {...inputProps}
          style={[styles.input, error ? styles.inputError : null, rightElement ? styles.inputWithAction : null, style]}
          placeholderTextColor="#94A3B8"
        />
        {rightElement}
      </View>
      {error ? <Text style={styles.fieldError}>{error}</Text> : null}
    </View>
  );
  },
);

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: colors.background,
  },
  pageGlowBlue: {
    position: 'absolute',
    width: 340,
    height: 340,
    borderRadius: 170,
    backgroundColor: '#DBEAFE',
    opacity: 0.5,
    top: -130,
    left: -120,
  },
  pageGlowTeal: {
    position: 'absolute',
    width: 320,
    height: 320,
    borderRadius: 160,
    backgroundColor: '#CCFBF1',
    opacity: 0.45,
    right: -110,
    bottom: -130,
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  authShell: {
    width: '100%',
    maxWidth: 1088,
    backgroundColor: '#FFFFFF',
    borderColor: colors.border,
    borderRadius: 24,
    borderWidth: 1,
    overflow: 'hidden',
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 0.08,
    shadowRadius: 34,
    elevation: 8,
  },
  authShellWide: {
    minHeight: 720,
    flexDirection: 'row',
  },
  brandPanel: {
    minHeight: 245,
    backgroundColor: '#F1F5FF',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 28,
  },
  brandPanelWide: {
    width: '42%',
    minHeight: '100%',
  },
  panelCircleTop: {
    position: 'absolute',
    width: 250,
    height: 250,
    borderRadius: 125,
    backgroundColor: '#DBEAFE',
    top: -76,
    right: -80,
  },
  panelCircleBottom: {
    position: 'absolute',
    width: 230,
    height: 230,
    borderRadius: 115,
    backgroundColor: '#CCFBF1',
    left: -82,
    bottom: -84,
  },
  floatCardTop: {
    position: 'absolute',
    top: 78,
    right: 38,
    width: 152,
    height: 86,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.68)',
    borderColor: '#BFDBFE',
    borderWidth: 1,
    padding: 16,
    justifyContent: 'center',
    gap: 10,
  },
  floatLineLong: {
    height: 9,
    width: '72%',
    borderRadius: 999,
    backgroundColor: '#BFDBFE',
  },
  floatLineShort: {
    height: 9,
    width: '46%',
    borderRadius: 999,
    backgroundColor: '#99F6E4',
  },
  floatCardBottom: {
    position: 'absolute',
    left: 46,
    bottom: 104,
    width: 116,
    height: 74,
    borderRadius: 22,
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderColor: '#CCFBF1',
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  miniDot: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: colors.primary,
  },
  miniDotMuted: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#BFDBFE',
  },
  miniDotTeal: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: colors.secondary,
  },
  illustrationCenter: {
    alignItems: 'center',
    maxWidth: 314,
    gap: 14,
  },
  iconTile: {
    width: 88,
    height: 88,
    borderRadius: 28,
    backgroundColor: '#FFFFFF',
    borderColor: '#BFDBFE',
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.1,
    shadowRadius: 18,
    elevation: 4,
  },
  waveBars: {
    height: 48,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  waveBar: {
    width: 7,
    borderRadius: 999,
    backgroundColor: colors.primary,
  },
  brandName: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '900',
    lineHeight: 32,
    textAlign: 'center',
  },
  brandSubtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  formPanel: {
    backgroundColor: '#FFFFFF',
    padding: 24,
  },
  formPanelWide: {
    flex: 1,
    paddingHorizontal: 56,
    paddingVertical: 58,
    justifyContent: 'center',
  },
  formInner: {
    width: '100%',
    maxWidth: 430,
    alignSelf: 'center',
    gap: 20,
  },
  header: {
    gap: 8,
  },
  title: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 38,
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
  },
  form: {
    gap: 15,
  },
  field: {
    gap: 8,
  },
  inputLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '800',
  },
  input: {
    minHeight: 56,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    paddingHorizontal: 16,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 15,
  },
  inputError: {
    borderColor: colors.error,
  },
  fieldError: {
    color: '#B91C1C',
    fontSize: 13,
    fontWeight: '700',
    lineHeight: 18,
  },
  profilePreparingNotice: {
    borderWidth: 1,
    borderColor: '#86EFAC',
    backgroundColor: '#F0FDF4',
    borderRadius: 14,
    padding: 14,
    gap: 4,
  },
  profilePreparingTitle: {
    color: '#166534',
    fontSize: 14,
    fontWeight: '900',
  },
  profilePreparingText: {
    color: '#166534',
    fontSize: 13,
    lineHeight: 19,
  },
  confirmationNotice: {
    borderWidth: 1,
    borderColor: '#FDE68A',
    backgroundColor: '#FFFBEB',
    borderRadius: 14,
    padding: 14,
    gap: 8,
  },
  confirmationTitle: {
    color: '#92400E',
    fontSize: 14,
    fontWeight: '900',
  },
  confirmationText: {
    color: '#92400E',
    fontSize: 13,
    lineHeight: 19,
  },
  confirmationSuccess: {
    color: '#166534',
    fontSize: 13,
    fontWeight: '700',
  },
  confirmationError: {
    color: '#B91C1C',
    fontSize: 13,
    fontWeight: '700',
  },
  passwordResetNotice: {
    backgroundColor: '#DCFCE7',
    borderRadius: 12,
    padding: 12,
  },
  passwordResetNoticeText: {
    color: '#166534',
    fontSize: 14,
    fontWeight: '700',
    lineHeight: 20,
  },
  inputWrap: {
    position: 'relative',
  },
  inputWithAction: {
    paddingRight: 52,
  },
  visibilityButton: {
    position: 'absolute',
    right: 8,
    top: 7,
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  rememberRow: {
    minHeight: 36,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  checkboxMark: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FFFFFF',
  },
  metaText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '700',
  },
  forgotLink: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '800',
  },
  forgotButton: {
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  submitButton: {
    minHeight: 56,
    borderRadius: 16,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.18,
    shadowRadius: 18,
    elevation: 5,
  },
  submitButtonDisabled: {
    opacity: 0.65,
    shadowOpacity: 0,
    elevation: 0,
  },
  submitText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  footerRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 5,
  },
  footerText: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '700',
  },
  footerLink: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
});
