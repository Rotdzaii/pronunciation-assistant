import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { AppButton, AppCard, AppScreen, ErrorState, SectionHeader, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';

const settings = [
  ['Vai trò', 'Người học'],
  ['Microphone', 'Sẵn sàng kiểm tra quyền truy cập'],
  ['Giao diện', 'Sáng'],
  ['Ngôn ngữ giao diện', 'Tiếng Việt'],
];

export default function ProfileScreen() {
  const { session, signOut } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSignOut = async () => {
    setLoading(true);
    setError(null);
    try {
      await signOut();
    } catch (signOutError) {
      setError(signOutError instanceof Error ? signOutError.message : 'Không thể đăng xuất.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AppScreen>
      <SectionHeader
        eyebrow="Tài khoản"
        title="Hồ sơ"
        subtitle="Quản lý thông tin tài khoản và cài đặt học tập."
      />

      {error ? <ErrorState title="Không thể đăng xuất" message={error} /> : null}

      <AppCard>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{session?.user?.email?.slice(0, 1).toUpperCase() ?? 'U'}</Text>
        </View>
        <Text style={styles.email}>{session?.user?.email ?? 'Chưa có email'}</Text>
        <Text style={styles.helper}>Dữ liệu hồ sơ mở rộng sẽ được hiển thị khi backend hỗ trợ.</Text>
      </AppCard>

      <AppCard>
        {settings.map(([label, value]) => (
          <View key={label} style={styles.settingRow}>
            <Text style={styles.settingLabel}>{label}</Text>
            <Text style={styles.settingValue}>{value}</Text>
          </View>
        ))}
        <AppButton
          title={loading ? 'Đang đăng xuất...' : 'Đăng xuất'}
          onPress={handleSignOut}
          loading={loading}
          variant="secondary"
        />
      </AppCard>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  avatar: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 24,
    fontWeight: '900',
  },
  email: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  helper: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20,
  },
  settingRow: {
    minHeight: 52,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  settingLabel: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '800',
  },
  settingValue: {
    flex: 1,
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'right',
  },
});
