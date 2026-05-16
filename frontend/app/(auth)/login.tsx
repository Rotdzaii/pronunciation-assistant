import { useEffect, useState } from 'react';
import type { ComponentProps } from 'react';
import { Link, useRouter } from 'expo-router';
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
import { ErrorState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';
import { supabase } from '../../lib/supabase';

type Role = 'student' | 'teacher';

export default function LoginScreen() {
  const router = useRouter();
  const { session } = useAuth();
  const { width } = useWindowDimensions();
  const isWide = width >= 860;
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('student');
  const [remember, setRemember] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async () => {
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setError(error.message);
    } else {
      router.replace(role === 'teacher' ? '/(tabs)/teacher' : '/(tabs)');
    }
    setLoading(false);
  };

  useEffect(() => {
    if (session) {
      router.replace('/(tabs)');
    }
  }, [session, router]);

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

              <RoleSelector value={role} onChange={setRole} />

              <View style={styles.form}>
                <LabeledInput
                  label="Địa chỉ Email"
                  placeholder="Nhập email của bạn"
                  autoCapitalize="none"
                  keyboardType="email-address"
                  value={email}
                  onChangeText={setEmail}
                />
                <LabeledInput
                  label="Mật khẩu"
                  placeholder="Nhập mật khẩu"
                  secureTextEntry
                  value={password}
                  onChangeText={setPassword}
                />
              </View>

              <View style={styles.metaRow}>
                <Pressable style={styles.rememberRow} onPress={() => setRemember((value) => !value)}>
                  <View style={[styles.checkbox, remember ? styles.checkboxActive : null]}>
                    {remember ? <View style={styles.checkboxMark} /> : null}
                  </View>
                  <Text style={styles.metaText}>Ghi nhớ đăng nhập</Text>
                </Pressable>
                <Text style={styles.forgotLink}>Quên mật khẩu?</Text>
              </View>

              {error ? <ErrorState title="Không thể đăng nhập" message={error} /> : null}

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

function RoleSelector({ value, onChange }: { value: Role; onChange: (role: Role) => void }) {
  return (
    <View style={styles.roleGroup}>
      <RoleOption label="Học viên" active={value === 'student'} onPress={() => onChange('student')} />
      <RoleOption label="Giáo viên" active={value === 'teacher'} onPress={() => onChange('teacher')} />
    </View>
  );
}

function RoleOption({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={[styles.roleOption, active ? styles.roleOptionActive : null]}>
      <Text style={[styles.roleText, active ? styles.roleTextActive : null]}>{label}</Text>
    </Pressable>
  );
}

function LabeledInput(props: ComponentProps<typeof TextInput> & { label: string }) {
  const { label, style, ...inputProps } = props;

  return (
    <View style={styles.field}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        {...inputProps}
        style={[styles.input, style]}
        placeholderTextColor="#94A3B8"
      />
    </View>
  );
}

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
  roleGroup: {
    flexDirection: 'row',
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderRadius: 16,
    borderWidth: 1,
    padding: 4,
    gap: 4,
  },
  roleOption: {
    flex: 1,
    minHeight: 46,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
  },
  roleOptionActive: {
    backgroundColor: colors.primary,
  },
  roleText: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '800',
  },
  roleTextActive: {
    color: '#FFFFFF',
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
