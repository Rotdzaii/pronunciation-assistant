import { useState } from 'react';
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
import { AppButton, ErrorState, colors } from '../../components/AppUI';
import { supabase } from '../../lib/supabase';

type Role = 'student' | 'teacher';

export default function RegisterScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const isWide = width >= 820;
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('student');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRegister = async () => {
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      setError(error.message);
    } else {
      router.replace('/(tabs)/practice');
    }
    setLoading(false);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.backgroundCircleOne} />
      <View style={styles.backgroundCircleTwo} />
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={[styles.authCard, isWide ? styles.authCardWide : null]}>
          <BrandPanel compact={!isWide} />

          <View style={[styles.formPanel, isWide ? styles.formPanelWide : null]}>
            <View style={styles.header}>
              <Text style={styles.title}>Tạo tài khoản mới</Text>
              <Text style={styles.subtitle}>
                Bắt đầu luyện phát âm với phản hồi bằng tiếng Việt.
              </Text>
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

            {error ? <ErrorState title="Không thể tạo tài khoản" message={error} /> : null}

            <AppButton
              title={loading ? 'Đang đăng ký...' : 'Đăng ký'}
              onPress={handleRegister}
              loading={loading}
            />

            <View style={styles.footerRow}>
              <Text style={styles.footerText}>Đã có tài khoản?</Text>
              <Link href="/(auth)/login" style={styles.footerLink}>
                Đăng nhập
              </Link>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function BrandPanel({ compact }: { compact: boolean }) {
  return (
    <View style={[styles.brandPanel, compact ? styles.brandPanelCompact : styles.brandPanelWide]}>
      <View style={styles.illustrationGlowOne} />
      <View style={styles.illustrationGlowTwo} />
      <View style={styles.illustrationBlockLarge} />
      <View style={styles.illustrationBlockSmall} />

      <View style={styles.brandContent}>
        <View style={styles.audioIcon}>
          <View style={styles.audioBars}>
            {[18, 30, 42, 26, 34].map((height, index) => (
              <View key={`${height}-${index}`} style={[styles.audioBar, { height }]} />
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
  backgroundCircleOne: {
    position: 'absolute',
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: '#DBEAFE',
    opacity: 0.55,
    top: -90,
    left: -80,
  },
  backgroundCircleTwo: {
    position: 'absolute',
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: '#CCFBF1',
    opacity: 0.48,
    right: -90,
    bottom: -110,
  },
  scrollContent: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  authCard: {
    width: '100%',
    maxWidth: 1100,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 24,
    borderWidth: 1,
    overflow: 'hidden',
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.08,
    shadowRadius: 30,
    elevation: 8,
  },
  authCardWide: {
    minHeight: 700,
    flexDirection: 'row',
  },
  brandPanel: {
    minHeight: 250,
    backgroundColor: '#EEF2FF',
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
  },
  brandPanelWide: {
    width: '42%',
    minHeight: '100%',
  },
  brandPanelCompact: {
    minHeight: 230,
  },
  illustrationGlowOne: {
    position: 'absolute',
    width: 220,
    height: 220,
    borderRadius: 110,
    backgroundColor: '#DBEAFE',
    top: -70,
    right: -70,
  },
  illustrationGlowTwo: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: '#CCFBF1',
    left: -48,
    bottom: -60,
  },
  illustrationBlockLarge: {
    position: 'absolute',
    width: 160,
    height: 90,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.62)',
    right: 34,
    top: 80,
    borderWidth: 1,
    borderColor: 'rgba(191, 219, 254, 0.9)',
  },
  illustrationBlockSmall: {
    position: 'absolute',
    width: 96,
    height: 96,
    borderRadius: 28,
    backgroundColor: 'rgba(20, 184, 166, 0.12)',
    left: 44,
    bottom: 92,
    borderWidth: 1,
    borderColor: 'rgba(20, 184, 166, 0.22)',
  },
  brandContent: {
    alignItems: 'center',
    maxWidth: 320,
    gap: 14,
  },
  audioIcon: {
    width: 86,
    height: 86,
    borderRadius: 26,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#BFDBFE',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
    elevation: 4,
  },
  audioBars: {
    height: 46,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  audioBar: {
    width: 7,
    borderRadius: 999,
    backgroundColor: colors.primary,
  },
  brandName: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '900',
    textAlign: 'center',
    lineHeight: 32,
  },
  brandSubtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  formPanel: {
    padding: 24,
    gap: 18,
  },
  formPanelWide: {
    flex: 1,
    paddingHorizontal: 48,
    paddingVertical: 56,
    justifyContent: 'center',
  },
  header: {
    gap: 8,
  },
  title: {
    color: colors.text,
    fontSize: 30,
    fontWeight: '900',
    lineHeight: 36,
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
    gap: 14,
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
