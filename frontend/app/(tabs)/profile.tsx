import { useEffect, useMemo, useState } from 'react';
import { Audio } from 'expo-av';
import { Image, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { AppButton, AppCard, AppScreen, ErrorState, SectionHeader, StatusBadge, colors } from '../../components/AppUI';
import { fetchMyProfile, updateMyProfile, uploadMyAvatar } from '../../lib/api';
import { formatUserRole } from '../../lib/format';
import { useAuth } from '../../lib/auth';
import { useTheme, type ThemeMode } from '../../lib/theme';
import type { UserProfile } from '../../types';

const MAX_AVATAR_BYTES = 2 * 1024 * 1024;

export default function ProfileScreen() {
  const { appRole, currentUser, session, signOut } = useAuth();
  const email = currentUser?.email ?? session?.user?.email ?? '';
  const { mode: themeMode, setThemeMode } = useTheme();
  const fallbackName = useMemo(() => email.split('@')[0] || 'Phoenix User', [email]);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [displayName, setDisplayName] = useState(fallbackName);
  const [microStatus, setMicroStatus] = useState('Chưa kiểm tra');
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [savingProfile, setSavingProfile] = useState(false);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resolvedEmail = profile?.email ?? email;
  const resolvedName = profile?.display_name || displayName || fallbackName;
  const avatarSource = avatarPreviewUrl ?? profile?.avatar_url ?? null;
  const initials = getInitials(resolvedName || resolvedEmail);
  const roleLabel = formatUserRole(profile?.app_role ?? appRole);
  const isDark = themeMode === 'dark';

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      setLoadingProfile(true);
      setError(null);
      try {
        const nextProfile = await fetchMyProfile();
        if (cancelled) return;
        setProfile(nextProfile);
        setDisplayName(nextProfile.display_name);
      } catch (loadError) {
        if (cancelled) return;
        setError(loadError instanceof Error ? loadError.message : 'Không thể tải hồ sơ.');
      } finally {
        if (!cancelled) {
          setLoadingProfile(false);
        }
      }
    }

    loadProfile();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (avatarPreviewUrl) {
        URL.revokeObjectURL(avatarPreviewUrl);
      }
    };
  }, [avatarPreviewUrl]);

  const handleSaveProfile = async () => {
    const trimmedName = displayName.trim();
    setNotice(null);
    setError(null);

    if (trimmedName.length < 2 || trimmedName.length > 80) {
      setError('Tên hiển thị phải có từ 2 đến 80 ký tự.');
      return;
    }

    setSavingProfile(true);
    try {
      const nextProfile = await updateMyProfile({ display_name: trimmedName });
      setProfile(nextProfile);
      setDisplayName(nextProfile.display_name);
      setNotice('Đã lưu hồ sơ.');
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Không thể lưu hồ sơ.');
    } finally {
      setSavingProfile(false);
    }
  };

  const handlePickAvatar = () => {
    setNotice(null);
    setError(null);

    if (Platform.OS !== 'web' || typeof document === 'undefined') {
      setNotice('Chọn ảnh hiện chỉ hỗ trợ trên web trong phiên bản này.');
      return;
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/jpeg,image/png,image/webp';
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return;

      if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
        setError('Avatar phải là ảnh JPEG, PNG hoặc WebP.');
        return;
      }

      if (file.size > MAX_AVATAR_BYTES) {
        setError('Avatar phải nhỏ hơn hoặc bằng 2MB.');
        return;
      }

      setAvatarFile(file);
      setAvatarPreviewUrl((previousUrl) => {
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
        return URL.createObjectURL(file);
      });
    };
    input.click();
  };

  const handleUploadAvatar = async () => {
    if (!avatarFile) {
      setNotice('Chọn ảnh trước khi cập nhật avatar.');
      return;
    }

    setNotice(null);
    setError(null);
    setUploadingAvatar(true);
    try {
      const response = await uploadMyAvatar(avatarFile);
      setProfile(response.profile);
      setAvatarFile(null);
      setAvatarPreviewUrl((previousUrl) => {
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
        return null;
      });
      setNotice('Đã cập nhật avatar.');
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Không thể cập nhật avatar.');
    } finally {
      setUploadingAvatar(false);
    }
  };

  const handleCheckMicro = async () => {
    setNotice(null);
    setError(null);
    try {
      const permission = await Audio.requestPermissionsAsync();
      setMicroStatus(permission.granted ? 'Đã cấp quyền' : 'Bị từ chối');
      if (!permission.granted) {
        setNotice('Trình duyệt hoặc thiết bị chưa cấp quyền micro.');
      }
    } catch (microError) {
      setMicroStatus('Trình duyệt không hỗ trợ');
      setError(microError instanceof Error ? microError.message : 'Không thể kiểm tra quyền micro.');
    }
  };

  const handleChangeTheme = async (nextThemeMode: ThemeMode) => {
    setThemeMode(nextThemeMode);
    setNotice(null);
    setError(null);
    try {
      await setThemeMode(nextThemeMode);
    } catch (themeError) {
      setError(themeError instanceof Error ? themeError.message : 'Không thể lưu lựa chọn giao diện.');
    }
  };

  const handleSignOut = async () => {
    setSigningOut(true);
    setError(null);
    try {
      await signOut();
    } catch (signOutError) {
      setError(signOutError instanceof Error ? signOutError.message : 'Không thể đăng xuất.');
    } finally {
      setSigningOut(false);
    }
  };

  return (
    <AppScreen maxWidth={900}>
      <SectionHeader
        eyebrow="Tài khoản"
        title="Hồ sơ"
        subtitle="Quản lý thông tin cá nhân, quyền micro và tùy chọn giao diện."
      />

      {error ? <ErrorState title="Không thể xử lý hồ sơ" message={error} /> : null}
      {notice ? <Notice message={notice} /> : null}

      <View style={styles.profileGrid}>
        <AppCard style={[styles.profileCard, isDark ? styles.darkCard : null]}>
          <View style={styles.avatar}>
            {avatarSource ? (
              <Image source={{ uri: avatarSource }} style={styles.avatarImage} />
            ) : (
              <Text style={styles.avatarText}>{initials}</Text>
            )}
          </View>
          <View style={styles.identity}>
            <Text style={[styles.displayName, isDark ? styles.darkText : null]}>
              {loadingProfile ? 'Đang tải hồ sơ...' : resolvedName}
            </Text>
            <Text style={[styles.email, isDark ? styles.darkMutedText : null]}>
              {resolvedEmail || 'Chưa có email'}
            </Text>
            <StatusBadge label={roleLabel} tone="primary" style={styles.roleBadge} />
          </View>
        </AppCard>

        <AppCard style={[styles.card, isDark ? styles.darkCard : null]}>
          <Text style={[styles.cardTitle, isDark ? styles.darkText : null]}>Thông tin hiển thị</Text>
          <View style={styles.inputGroup}>
            <Text style={[styles.inputLabel, isDark ? styles.darkText : null]}>Tên hiển thị</Text>
            <TextInput
              value={displayName}
              onChangeText={setDisplayName}
              placeholder="Nhập tên hiển thị"
              placeholderTextColor={isDark ? '#94A3B8' : '#64748B'}
              style={[styles.input, isDark ? styles.darkInput : null]}
            />
          </View>
          <View style={styles.avatarActions}>
            <Pressable accessibilityRole="button" onPress={handlePickAvatar} style={styles.outlineButton}>
              <MaterialCommunityIcons name="image-plus" size={17} color={colors.primary} />
              <Text style={styles.outlineButtonText}>Chọn ảnh</Text>
            </Pressable>
            <AppButton
              title={uploadingAvatar ? 'Đang cập nhật...' : 'Cập nhật avatar'}
              onPress={handleUploadAvatar}
              loading={uploadingAvatar}
              disabled={!avatarFile}
              variant="secondary"
              style={styles.avatarUploadButton}
            />
          </View>
          <Text style={[styles.mutedText, isDark ? styles.darkMutedText : null]}>
            Hỗ trợ JPEG, PNG hoặc WebP, tối đa 2MB.
          </Text>
          <AppButton
            title={savingProfile ? 'Đang lưu...' : 'Lưu thay đổi'}
            onPress={handleSaveProfile}
            loading={savingProfile}
            disabled={loadingProfile}
          />
        </AppCard>

        <AppCard style={[styles.card, isDark ? styles.darkCard : null]}>
          <Text style={[styles.cardTitle, isDark ? styles.darkText : null]}>Thiết bị và giao diện</Text>
          <View style={[styles.settingRow, isDark ? styles.darkSettingRow : null]}>
            <View style={styles.settingCopy}>
              <Text style={[styles.settingLabel, isDark ? styles.darkText : null]}>Quyền micro</Text>
              <Text style={[styles.mutedText, isDark ? styles.darkMutedText : null]}>{microStatus}</Text>
            </View>
            <Pressable accessibilityRole="button" onPress={handleCheckMicro} style={styles.outlineButton}>
              <MaterialCommunityIcons name="microphone-outline" size={17} color={colors.primary} />
              <Text style={styles.outlineButtonText}>Kiểm tra micro</Text>
            </Pressable>
          </View>
          <View style={[styles.settingRow, isDark ? styles.darkSettingRow : null]}>
            <View style={styles.settingCopy}>
              <Text style={[styles.settingLabel, isDark ? styles.darkText : null]}>Giao diện</Text>
              <Text style={[styles.mutedText, isDark ? styles.darkMutedText : null]}>
                Lựa chọn được lưu trên thiết bị này.
              </Text>
            </View>
            <View style={styles.themeToggle}>
              <ThemeOption
                icon="white-balance-sunny"
                label="Sáng"
                selected={themeMode === 'light'}
                onPress={() => handleChangeTheme('light')}
              />
              <ThemeOption
                icon="moon-waning-crescent"
                label="Tối"
                selected={themeMode === 'dark'}
                onPress={() => handleChangeTheme('dark')}
              />
            </View>
          </View>
          <View style={[styles.settingRow, styles.lastSettingRow]}>
            <View style={styles.settingCopy}>
              <Text style={[styles.settingLabel, isDark ? styles.darkText : null]}>Vai trò</Text>
              <Text style={[styles.mutedText, isDark ? styles.darkMutedText : null]}>
                Không thể đổi role trong Hồ sơ.
              </Text>
            </View>
            <StatusBadge label={roleLabel} tone="primary" />
          </View>
        </AppCard>

        <AppCard style={[styles.card, isDark ? styles.darkCard : null]}>
          <Text style={[styles.cardTitle, isDark ? styles.darkText : null]}>Phiên đăng nhập</Text>
          <AppButton
            title={signingOut ? 'Đang đăng xuất...' : 'Đăng xuất'}
            onPress={handleSignOut}
            loading={signingOut}
            variant="secondary"
          />
        </AppCard>
      </View>
    </AppScreen>
  );
}

function ThemeOption({
  icon,
  label,
  selected,
  onPress,
}: {
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={[styles.themeOption, selected ? styles.themeOptionSelected : null]}
    >
      <MaterialCommunityIcons name={icon} size={17} color={selected ? '#FFFFFF' : colors.primary} />
      <Text style={[styles.themeOptionText, selected ? styles.themeOptionTextSelected : null]}>{label}</Text>
    </Pressable>
  );
}

function Notice({ message }: { message: string }) {
  return (
    <View style={styles.notice}>
      <MaterialCommunityIcons name="information-outline" size={18} color={colors.primary} />
      <Text style={styles.noticeText}>{message}</Text>
    </View>
  );
}

function getInitials(value: string): string {
  const parts = value.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }
  return value.slice(0, 2).toUpperCase() || 'PA';
}

const styles = StyleSheet.create({
  profileGrid: {
    gap: 16,
  },
  profileCard: {
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  avatar: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: {
    width: 84,
    height: 84,
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 26,
    fontWeight: '900',
  },
  identity: {
    flex: 1,
    gap: 5,
  },
  displayName: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '900',
  },
  email: {
    color: colors.muted,
    fontSize: 14,
  },
  roleBadge: {
    alignSelf: 'flex-start',
  },
  card: {
    borderRadius: 12,
  },
  darkCard: {
    backgroundColor: '#111827',
    borderColor: '#334155',
  },
  cardTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  inputGroup: {
    gap: 7,
  },
  inputLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  input: {
    minHeight: 48,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 15,
    paddingHorizontal: 12,
  },
  darkInput: {
    backgroundColor: '#0F172A',
    borderColor: '#334155',
    color: '#E5E7EB',
  },
  avatarActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 10,
  },
  avatarUploadButton: {
    minWidth: 170,
  },
  mutedText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  darkText: {
    color: '#E5E7EB',
  },
  darkMutedText: {
    color: '#CBD5E1',
  },
  settingRow: {
    minHeight: 62,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    paddingVertical: 10,
  },
  darkSettingRow: {
    borderBottomColor: '#334155',
  },
  lastSettingRow: {
    borderBottomWidth: 0,
  },
  settingCopy: {
    flex: 1,
    gap: 3,
  },
  settingLabel: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  outlineButton: {
    minHeight: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 12,
  },
  outlineButtonText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  themeToggle: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    gap: 4,
    padding: 4,
  },
  themeOption: {
    minHeight: 34,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
  },
  themeOptionSelected: {
    backgroundColor: colors.primary,
  },
  themeOptionText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  themeOptionTextSelected: {
    color: '#FFFFFF',
  },
  notice: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
  },
  noticeText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '800',
  },
});
