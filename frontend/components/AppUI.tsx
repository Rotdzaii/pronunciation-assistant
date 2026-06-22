import { PropsWithChildren, useEffect, useMemo, useState } from 'react';
import { type Href, useGlobalSearchParams, usePathname, useRouter } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import {
  ActivityIndicator,
  Image,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleProp,
  StyleSheet,
  Text,
  TextInput,
  TextStyle,
  useWindowDimensions,
  View,
  ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { fetchMyProfile, updateMyProfile, uploadMyAvatar } from '../lib/api';
import { useAuth } from '../lib/auth';
import { formatUserRole } from '../lib/format';
import { useTheme, type ThemeMode } from '../lib/theme';
import type { UserProfile } from '../types';

export const colors = {
  background: '#F8FAFC',
  surface: '#FFFFFF',
  primary: '#2563EB',
  secondary: '#14B8A6',
  accent: '#F97316',
  success: '#22C55E',
  warning: '#F59E0B',
  error: '#EF4444',
  text: '#0F172A',
  muted: '#64748B',
  border: '#E2E8F0',
  softBlue: '#EFF6FF',
  softTeal: '#F0FDFA',
  softOrange: '#FFF7ED',
  softRed: '#FEF2F2',
};

type AppCardProps = PropsWithChildren<{
  style?: StyleProp<ViewStyle>;
  tone?: 'default' | 'blue' | 'teal' | 'orange';
}>;

type AppButtonProps = {
  title: string;
  onPress?: () => void;
  disabled?: boolean;
  loading?: boolean;
  variant?: 'primary' | 'secondary' | 'ghost';
  style?: StyleProp<ViewStyle>;
};

type SectionHeaderProps = {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  align?: 'left' | 'center';
};

type StateProps = {
  title: string;
  message?: string;
};

type StatusBadgeProps = {
  label: string;
  tone?: 'idle' | 'processing' | 'success' | 'warning' | 'error' | 'primary';
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
};

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type SidebarItemConfig = {
  label: string;
  href?: Href<string>;
  icon: IconName;
  match: string[];
  section?: string;
};

type TeacherSidebarNotifications = {
  classes: number;
};

type SidebarVariant = 'student' | 'teacher' | 'admin';

const MAX_AVATAR_BYTES = 2 * 1024 * 1024;

const studentSidebarItems: SidebarItemConfig[] = [
  { label: 'Trang chủ', href: '/(tabs)', icon: 'home-outline', match: ['/', '/(tabs)'] },
  { label: 'Luyện tập', href: '/(tabs)/practice', icon: 'microphone-outline', match: ['/practice', '/(tabs)/practice'] },
  { label: 'Lịch sử', href: '/(tabs)/history', icon: 'history', match: ['/history', '/(tabs)/history'] },
  { label: 'Lỗi phổ biến', href: '/(tabs)/mistakes', icon: 'alert-circle-outline', match: ['/mistakes', '/(tabs)/mistakes'] },
  {
    label: 'Ôn tập',
    href: '/(tabs)/vocabulary',
    icon: 'cards-outline',
    match: ['/vocabulary', '/(tabs)/vocabulary', '/quiz', '/(tabs)/quiz', '/quiz-results', '/(tabs)/quiz-results'],
  },
  { label: 'Hỗ trợ', href: '/(tabs)', icon: 'lifebuoy', match: [], section: 'support' },
];

const teacherSidebarItems: SidebarItemConfig[] = [
  {
    label: 'Tổng quan',
    href: '/(tabs)/teacher',
    icon: 'view-dashboard-outline',
    match: ['/teacher', '/(tabs)/teacher'],
    section: 'overview',
  },
  {
    label: 'Lớp học',
    href: '/(tabs)/teacher',
    icon: 'account-group-outline',
    match: ['/teacher', '/(tabs)/teacher'],
    section: 'classes',
  },
  {
    label: 'Báo cáo',
    href: '/(tabs)/teacher',
    icon: 'file-chart-outline',
    match: ['/teacher', '/(tabs)/teacher'],
    section: 'reports',
  },
  {
    label: 'Hỗ trợ',
    href: '/(tabs)/teacher',
    icon: 'lifebuoy',
    match: ['/teacher', '/(tabs)/teacher'],
    section: 'support',
  },
];

const adminSidebarItems: SidebarItemConfig[] = [
  {
    label: 'Tổng quan',
    href: '/(tabs)/admin',
    icon: 'view-dashboard-outline',
    match: ['/admin', '/(tabs)/admin'],
    section: 'overview',
  },
  {
    label: 'Người dùng',
    href: '/(tabs)/admin',
    icon: 'account-group-outline',
    match: ['/admin', '/(tabs)/admin'],
    section: 'users',
  },
  {
    label: 'Lớp học',
    href: '/(tabs)/admin',
    icon: 'school-outline',
    match: ['/admin', '/(tabs)/admin'],
    section: 'classes',
  },
  {
    label: 'Xuất báo cáo',
    href: '/(tabs)/admin',
    icon: 'download-outline',
    match: ['/admin', '/(tabs)/admin'],
    section: 'exports',
  },
  {
    label: 'Demo readiness',
    href: '/(tabs)/admin',
    icon: 'clipboard-check-outline',
    match: ['/admin', '/(tabs)/admin'],
    section: 'readiness',
  },
  { label: 'Hỗ trợ', href: '/(tabs)/admin', icon: 'lifebuoy', match: [], section: 'support' },
];

export function AppCard({ children, style, tone = 'default' }: AppCardProps) {
  const { theme, mode } = useTheme();
  const toneStyle = mode === 'dark' ? darkToneStyles[tone] : toneStyles[tone];

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: theme.card,
          borderColor: theme.border,
          shadowColor: theme.shadow,
        },
        toneStyle,
        style,
      ]}
    >
      {children}
    </View>
  );
}

export function AppButton({
  title,
  onPress,
  disabled = false,
  loading = false,
  variant = 'primary',
  style,
}: AppButtonProps) {
  const { theme, mode } = useTheme();
  const isDisabled = disabled || loading;
  const dynamicButtonStyle =
    variant === 'primary'
      ? {
          backgroundColor: theme.primary,
          borderColor: theme.primary,
          shadowColor: theme.primary,
        }
      : variant === 'ghost'
        ? {
            backgroundColor: theme.softBlue,
            borderColor: mode === 'dark' ? theme.border : '#BFDBFE',
          }
        : {
            backgroundColor: theme.surface,
            borderColor: theme.border,
          };

  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={isDisabled}
      style={[
        styles.button,
        buttonStyles[variant],
        dynamicButtonStyle,
        isDisabled ? styles.buttonDisabled : null,
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? theme.primaryText : theme.primary} />
      ) : (
        <Text
          style={[
            styles.buttonText,
            { color: variant === 'primary' ? theme.primaryText : theme.text },
            isDisabled && variant !== 'primary' ? styles.disabledButtonText : null,
          ]}
        >
          {title}
        </Text>
      )}
    </Pressable>
  );
}

export function SectionHeader({ eyebrow, title, subtitle, align = 'left' }: SectionHeaderProps) {
  const { theme } = useTheme();
  const centered = align === 'center';

  return (
    <View style={[styles.sectionHeader, centered ? styles.centered : null]}>
      {eyebrow ? <Text style={[styles.eyebrow, { color: theme.primary }]}>{eyebrow}</Text> : null}
      <Text style={[styles.sectionTitle, { color: theme.text }, centered ? styles.centeredText : null]}>{title}</Text>
      {subtitle ? (
        <Text style={[styles.sectionSubtitle, { color: theme.textMuted }, centered ? styles.centeredText : null]}>
          {subtitle}
        </Text>
      ) : null}
    </View>
  );
}

export function EmptyState({ title, message }: StateProps) {
  const { theme } = useTheme();
  return (
    <AppCard style={styles.stateCard}>
      <View style={styles.stateMark}>
        <View style={styles.stateMarkDot} />
      </View>
      <Text style={[styles.stateTitle, { color: theme.text }]}>{title}</Text>
      {message ? <Text style={[styles.stateMessage, { color: theme.textMuted }]}>{message}</Text> : null}
    </AppCard>
  );
}

export function LoadingState({ title, message }: StateProps) {
  const { theme } = useTheme();
  return (
    <AppCard style={styles.stateCard}>
      <ActivityIndicator color={theme.primary} />
      <Text style={[styles.stateTitle, { color: theme.text }]}>{title}</Text>
      {message ? <Text style={[styles.stateMessage, { color: theme.textMuted }]}>{message}</Text> : null}
    </AppCard>
  );
}

export function ErrorState({ title, message }: StateProps) {
  const { theme } = useTheme();
  return (
    <View style={[styles.errorState, { backgroundColor: theme.softRed, borderColor: theme.danger }]}>
      <Text style={[styles.errorTitle, { color: theme.danger }]}>{title}</Text>
      {message ? <Text style={[styles.errorMessage, { color: theme.text }]}>{message}</Text> : null}
    </View>
  );
}

export function StatusBadge({
  label,
  tone = 'idle',
  style,
  textStyle,
}: StatusBadgeProps) {
  const { theme } = useTheme();
  const dynamicBadgeStyles = {
    idle: { backgroundColor: theme.textMuted },
    processing: { backgroundColor: theme.warning },
    success: { backgroundColor: theme.success },
    warning: { backgroundColor: colors.accent },
    error: { backgroundColor: theme.danger },
    primary: { backgroundColor: theme.primary },
  };

  return (
    <View style={[styles.badge, badgeStyles[tone], dynamicBadgeStyles[tone], style]}>
      <Text style={[styles.badgeText, tone === 'primary' ? { color: theme.primaryText } : null, textStyle]}>{label}</Text>
    </View>
  );
}

export function AppSidebar({ variant = 'student' }: { variant?: SidebarVariant }) {
  const { theme, mode } = useTheme();
  const { currentUser, appRole, signOut, refreshCurrentUser } = useAuth();
  const pathname = usePathname();
  const params = useGlobalSearchParams<{ section?: string | string[] }>();
  const router = useRouter();
  const [activeTeacherSection, setActiveTeacherSection] = useState('overview');
  const [profileOpen, setProfileOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [teacherNotifications, setTeacherNotifications] = useState<TeacherSidebarNotifications>({ classes: 0 });
  const items = variant === 'admin' ? adminSidebarItems : variant === 'teacher' ? teacherSidebarItems : studentSidebarItems;
  const isTeacher = variant === 'teacher';
  const mainItems = items.filter((item) => item.section !== 'support');
  const supportItems = items.filter((item) => item.section === 'support');
  const section = Array.isArray(params.section) ? params.section[0] : params.section;
  const displayName = profile?.display_name || currentUser?.email?.split('@')[0] || 'Phoenix User';
  const email = profile?.email ?? currentUser?.email ?? '';
  const roleLabel = formatUserRole(profile?.app_role ?? appRole);
  const initials = getInitials(displayName || email);
  // TODO: wire this to backend/app version metadata when available.
  const profileNotificationCount = 0;

  useEffect(() => {
    if (variant !== 'teacher' || typeof window === 'undefined') {
      return undefined;
    }

    const onTeacherSection = (event: Event) => {
      const rawSection = (event as CustomEvent<string>).detail;
      const nextSection = rawSection === 'actions' ? 'classes' : rawSection;
      if (nextSection) {
        setActiveTeacherSection(nextSection);
      }
    };

    window.addEventListener('phoenix:teacher-section', onTeacherSection);
    return () => window.removeEventListener('phoenix:teacher-section', onTeacherSection);
  }, [variant]);

  useEffect(() => {
    if (variant !== 'teacher' || typeof window === 'undefined') {
      return undefined;
    }

    const onTeacherNotifications = (event: Event) => {
      const detail = (event as CustomEvent<Partial<TeacherSidebarNotifications>>).detail;
      setTeacherNotifications((current) => ({
        classes: Math.max(0, Number(detail?.classes ?? current.classes) || 0),
      }));
    };

    window.addEventListener('phoenix:teacher-notifications', onTeacherNotifications);
    return () => window.removeEventListener('phoenix:teacher-notifications', onTeacherNotifications);
  }, [variant]);

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      try {
        const nextProfile = await fetchMyProfile();
        if (!cancelled) {
          setProfile(nextProfile);
        }
      } catch {
        if (!cancelled) {
          setProfile(null);
        }
      }
    }

    void loadProfile();

    return () => {
      cancelled = true;
    };
  }, [currentUser?.id]);

  const renderItem = (item: SidebarItemConfig, utility = false) => {
    const matchesPath = item.match.includes(pathname);
    const matchesTeacherPath = pathname === '/teacher' || pathname === '/(tabs)/teacher';
    const isTeacherOverview = item.match.includes('/teacher') || item.match.includes('/(tabs)/teacher');
    const isActive = isTeacher && item.section
      ? matchesPath && activeTeacherSection === item.section
      : variant === 'admin' && item.section === 'overview'
      ? matchesPath && !section
      : item.section
      ? matchesPath && section === item.section
      : matchesPath && (!isTeacherOverview || !section);
    const href = item.href;
    const isTeacherInternalItem = isTeacher && Boolean(item.section);
    const canPress = Boolean(href || isTeacherInternalItem);
    const notificationCount = isTeacher && item.section === 'classes' ? teacherNotifications.classes : 0;
    const activeColor = mode === 'dark' ? '#F8FAFC' : '#0F766E';
    const inactiveColor = mode === 'dark' ? '#94A3B8' : '#64748B';
    const activeBackground = mode === 'dark' ? '#153447' : '#ECFDF5';

    return (
      <Pressable
        key={item.label}
        accessibilityRole="link"
        onPress={
          isTeacherInternalItem
            ? () => {
                const nextSection = item.section ?? 'overview';
                setActiveTeacherSection(nextSection);
                if (typeof window !== 'undefined') {
                  (window as Window & { __phoenixTeacherSection?: string }).__phoenixTeacherSection = nextSection;
                  window.dispatchEvent(new CustomEvent('phoenix:teacher-section', { detail: nextSection }));
                }
                if (!matchesTeacherPath) {
                  router.push('/(tabs)/teacher');
                }
              }
            : href
              ? () => router.push(href)
              : undefined
        }
        disabled={!canPress}
        style={[
          styles.sidebarItem,
          styles.roleSidebarItem,
          utility ? styles.roleUtilityItem : null,
          isActive ? [styles.roleSidebarItemActive, { backgroundColor: activeBackground }] : null,
          !canPress ? styles.sidebarItemDisabled : null,
        ]}
      >
        <View style={[styles.roleNavBar, isActive ? styles.roleNavBarActive : null]} />
        <MaterialCommunityIcons
          name={item.icon}
          size={utility ? 21 : 24}
          color={isActive ? activeColor : inactiveColor}
        />
        <Text
          style={[
            styles.sidebarItemText,
            { color: inactiveColor },
            utility ? styles.roleUtilityText : null,
            isActive ? [styles.roleSidebarItemTextActive, { color: activeColor }] : null,
            !canPress ? styles.sidebarItemTextDisabled : null,
          ]}
        >
          {item.label}
        </Text>
        {notificationCount > 0 ? <MenuNotificationBadge count={notificationCount} /> : null}
      </Pressable>
    );
  };

  return (
    <View
      style={[
        styles.sidebar,
        styles.roleSidebar,
        {
          backgroundColor: mode === 'dark' ? '#0F172A' : '#F3F7FF',
          borderRightColor: mode === 'dark' ? '#334155' : '#E2E8F0',
        },
      ]}
    >
      <View style={styles.sidebarBrand}>
        <Text style={[styles.sidebarTitle, { color: theme.primary }]}>Trợ lý Phát âm</Text>
        <Text style={[styles.sidebarSubtitle, { color: theme.textMuted }]}>Học cùng AI</Text>
      </View>

      <View style={styles.sidebarNav}>
        {mainItems.map((item) => renderItem(item))}
      </View>

      <View style={styles.roleSidebarSpacer} />

      <View style={[styles.roleUtilityNav, { borderTopColor: mode === 'dark' ? '#334155' : '#DCE3F0' }]}>
        {supportItems.map((item) => renderItem(item, true))}
        <UserMiniCard
          initials={initials}
          displayName={displayName}
          roleLabel={roleLabel}
          avatarUrl={profile?.avatar_url ?? null}
          notificationCount={profileNotificationCount}
          onOpenProfile={() => setProfileOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </View>

      <ProfileQuickView
        visible={profileOpen}
        profile={profile}
        displayName={displayName}
        email={email}
        roleLabel={roleLabel}
        initials={initials}
        onClose={() => setProfileOpen(false)}
        onEditProfile={() => {
          setProfileOpen(false);
          setSettingsOpen(true);
        }}
        onSignOut={signOut}
      />
      <AccountSettingsModal
        visible={settingsOpen}
        profile={profile}
        fallbackEmail={email}
        roleLabel={roleLabel}
        onClose={() => setSettingsOpen(false)}
        onProfileChange={(nextProfile) => {
          setProfile(nextProfile);
          void refreshCurrentUser();
        }}
        onSignOut={signOut}
      />
    </View>
  );
}

function UserMiniCard({
  initials,
  displayName,
  roleLabel,
  avatarUrl,
  notificationCount,
  onOpenProfile,
  onOpenSettings,
}: {
  initials: string;
  displayName: string;
  roleLabel: string;
  avatarUrl: string | null;
  notificationCount: number;
  onOpenProfile: () => void;
  onOpenSettings: () => void;
}) {
  const { theme } = useTheme();

  return (
    <View style={[styles.userMiniCard, { backgroundColor: theme.cardMuted, borderColor: theme.border }]}>
      <Pressable accessibilityRole="button" onPress={onOpenProfile} style={styles.userMiniIdentity}>
        <View style={styles.userMiniAvatarWrap}>
          <Avatar initials={initials} avatarUrl={avatarUrl} size={36} />
          {notificationCount > 0 ? <View style={styles.profileNotificationDot} /> : null}
        </View>
        <View style={styles.userMiniText}>
          <Text numberOfLines={1} style={[styles.userMiniName, { color: theme.text }]}>{displayName}</Text>
          <Text numberOfLines={1} style={[styles.userMiniRole, { color: theme.textMuted }]}>{roleLabel}</Text>
        </View>
      </Pressable>
      <Pressable accessibilityRole="button" onPress={onOpenSettings} style={styles.userMiniIconButton}>
        <MaterialCommunityIcons name="cog-outline" size={20} color={theme.textMuted} />
      </Pressable>
    </View>
  );
}

function MenuNotificationBadge({ count }: { count: number }) {
  return (
    <View style={styles.menuNotificationBadge}>
      <Text style={styles.menuNotificationText}>{count > 99 ? '99+' : String(count)}</Text>
    </View>
  );
}

function Avatar({
  initials,
  avatarUrl,
  size,
}: {
  initials: string;
  avatarUrl: string | null;
  size: number;
}) {
  return (
    <View style={[styles.avatarShell, { width: size, height: size, borderRadius: size / 2 }]}>
      {avatarUrl ? (
        <Image source={{ uri: avatarUrl }} style={[styles.avatarImage, { width: size, height: size, borderRadius: size / 2 }]} />
      ) : (
        <Text style={[styles.avatarInitials, { fontSize: Math.max(13, size * 0.34) }]}>{initials}</Text>
      )}
    </View>
  );
}

function ProfileQuickView({
  visible,
  profile,
  displayName,
  email,
  roleLabel,
  initials,
  onClose,
  onEditProfile,
  onSignOut,
}: {
  visible: boolean;
  profile: UserProfile | null;
  displayName: string;
  email: string;
  roleLabel: string;
  initials: string;
  onClose: () => void;
  onEditProfile: () => void;
  onSignOut: () => Promise<void>;
}) {
  const { theme } = useTheme();
  const [signingOut, setSigningOut] = useState(false);

  const handleSignOut = async () => {
    setSigningOut(true);
    try {
      await onSignOut();
    } finally {
      setSigningOut(false);
    }
  };

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={styles.modalOverlay} onPress={onClose}>
        <Pressable style={[styles.quickView, { backgroundColor: theme.card, borderColor: theme.border }]} onPress={(event) => event.stopPropagation()}>
          <View style={styles.quickHeader}>
            <Avatar initials={initials} avatarUrl={profile?.avatar_url ?? null} size={72} />
            <View style={styles.quickIdentity}>
              <Text style={[styles.quickName, { color: theme.text }]}>{displayName}</Text>
              <Text style={[styles.quickEmail, { color: theme.textMuted }]}>{email || 'Chưa có email'}</Text>
              <StatusBadge label={roleLabel} tone="primary" style={styles.quickBadge} />
            </View>
          </View>
          <View style={styles.quickActions}>
            <AppButton title="Chỉnh sửa hồ sơ" onPress={onEditProfile} variant="secondary" />
            <AppButton title={signingOut ? 'Đang đăng xuất...' : 'Đăng xuất'} onPress={handleSignOut} loading={signingOut} variant="secondary" />
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

function AccountSettingsModal({
  visible,
  profile,
  fallbackEmail,
  roleLabel,
  onClose,
  onProfileChange,
  onSignOut,
}: {
  visible: boolean;
  profile: UserProfile | null;
  fallbackEmail: string;
  roleLabel: string;
  onClose: () => void;
  onProfileChange: (profile: UserProfile) => void;
  onSignOut: () => Promise<void>;
}) {
  const { theme, mode, setThemeMode } = useTheme();
  const [activeTab, setActiveTab] = useState('account');
  const [displayName, setDisplayName] = useState(profile?.display_name ?? '');
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState<string | null>(null);
  const [microStatus, setMicroStatus] = useState('Chưa kiểm tra');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const email = profile?.email ?? fallbackEmail;
  const name = displayName || profile?.display_name || email.split('@')[0] || 'Phoenix User';
  const avatarUrl = avatarPreviewUrl ?? profile?.avatar_url ?? null;
  const initials = getInitials(name || email);

  useEffect(() => {
    if (visible) {
      setDisplayName(profile?.display_name ?? email.split('@')[0] ?? '');
      setNotice(null);
      setError(null);
    }
  }, [email, profile?.display_name, visible]);

  useEffect(() => () => {
    if (avatarPreviewUrl) {
      URL.revokeObjectURL(avatarPreviewUrl);
    }
  }, [avatarPreviewUrl]);

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
        if (previousUrl) URL.revokeObjectURL(previousUrl);
        return URL.createObjectURL(file);
      });
    };
    input.click();
  };

  const handleSaveProfile = async () => {
    const trimmed = displayName.trim();
    setNotice(null);
    setError(null);
    if (trimmed.length < 2 || trimmed.length > 80) {
      setError('Tên hiển thị phải có từ 2 đến 80 ký tự.');
      return;
    }
    setSaving(true);
    try {
      const nextProfile = await updateMyProfile({ display_name: trimmed });
      onProfileChange(nextProfile);
      setNotice('Đã lưu hồ sơ.');
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Không thể lưu hồ sơ.');
    } finally {
      setSaving(false);
    }
  };

  const handleUploadAvatar = async () => {
    if (!avatarFile) {
      setNotice('Chọn ảnh trước khi cập nhật avatar.');
      return;
    }
    setNotice(null);
    setError(null);
    setUploading(true);
    try {
      const response = await uploadMyAvatar(avatarFile);
      onProfileChange(response.profile);
      setAvatarFile(null);
      setAvatarPreviewUrl((previousUrl) => {
        if (previousUrl) URL.revokeObjectURL(previousUrl);
        return null;
      });
      setNotice('Đã cập nhật avatar.');
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Không thể cập nhật avatar.');
    } finally {
      setUploading(false);
    }
  };

  const handleCheckMicro = async () => {
    setNotice(null);
    setError(null);
    try {
      const permission = await Audio.requestPermissionsAsync();
      setMicroStatus(permission.granted ? 'Đã cấp quyền' : 'Bị từ chối');
    } catch (microError) {
      setMicroStatus('Trình duyệt không hỗ trợ');
      setError(microError instanceof Error ? microError.message : 'Không thể kiểm tra quyền micro.');
    }
  };

  const handleTheme = async (nextMode: ThemeMode) => {
    try {
      await setThemeMode(nextMode);
      setNotice('Đã lưu lựa chọn giao diện.');
    } catch (themeError) {
      setError(themeError instanceof Error ? themeError.message : 'Không thể lưu giao diện.');
    }
  };

  const handleSignOut = async () => {
    setSigningOut(true);
    try {
      await onSignOut();
    } finally {
      setSigningOut(false);
    }
  };

  const navItems = [
    { key: 'account', label: 'Tài khoản', icon: 'account-circle-outline' as IconName },
    { key: 'profile', label: 'Hồ sơ', icon: 'card-account-details-outline' as IconName },
    { key: 'device', label: 'Thiết bị & Micro', icon: 'microphone-outline' as IconName },
    { key: 'appearance', label: 'Giao diện', icon: 'palette-outline' as IconName },
    { key: 'notifications', label: 'Thông báo', icon: 'bell-outline' as IconName },
    { key: 'privacy', label: 'Quyền riêng tư', icon: 'shield-lock-outline' as IconName },
    { key: 'logout', label: 'Đăng xuất', icon: 'logout' as IconName },
  ];

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.settingsOverlay}>
        <View style={[styles.settingsModal, { backgroundColor: theme.card, borderColor: theme.border }]}>
          <View style={[styles.settingsNav, { borderRightColor: theme.border }]}>
            <Text style={[styles.settingsNavTitle, { color: theme.textMuted }]}>Cài đặt Phoenix</Text>
            {navItems.map((item) => {
              const selected = activeTab === item.key;
              return (
                <Pressable
                  key={item.key}
                  accessibilityRole="button"
                  onPress={() => setActiveTab(item.key)}
                  style={[styles.settingsNavItem, selected ? { backgroundColor: mode === 'dark' ? '#1E293B' : '#ECFDF5' } : null]}
                >
                  <MaterialCommunityIcons name={item.icon} size={19} color={selected ? '#0F766E' : theme.textMuted} />
                  <Text style={[styles.settingsNavText, { color: selected ? '#0F766E' : theme.textMuted }]}>{item.label}</Text>
                </Pressable>
              );
            })}
          </View>
          <View style={styles.settingsContent}>
            <View style={styles.settingsHeader}>
              <Text style={[styles.settingsTitle, { color: theme.text }]}>{navItems.find((item) => item.key === activeTab)?.label}</Text>
              <Pressable accessibilityRole="button" onPress={onClose} style={styles.settingsCloseButton}>
                <MaterialCommunityIcons name="close" size={24} color={theme.textMuted} />
              </Pressable>
            </View>
            {error ? <ErrorState title="Không thể xử lý cài đặt" message={error} /> : null}
            {notice ? <NoticeInline message={notice} /> : null}
            {activeTab === 'account' ? (
              <SettingsPanel>
                <SettingsRow label="Email" value={email || 'Chưa có email'} />
                <SettingsRow label="Vai trò" value={roleLabel} />
                <SettingsRow label="Trạng thái" value="Đang hoạt động" />
              </SettingsPanel>
            ) : null}
            {activeTab === 'profile' ? (
              <SettingsPanel>
                <View style={styles.settingsAvatarRow}>
                  <Avatar initials={initials} avatarUrl={avatarUrl} size={82} />
                  <View style={styles.settingsAvatarActions}>
                    <AppButton title="Chọn ảnh" onPress={handlePickAvatar} variant="secondary" />
                    <AppButton title={uploading ? 'Đang cập nhật...' : 'Cập nhật avatar'} onPress={handleUploadAvatar} loading={uploading} disabled={!avatarFile} variant="secondary" />
                  </View>
                </View>
                <Text style={[styles.settingsLabel, { color: theme.text }]}>Tên hiển thị</Text>
                <TextInput
                  value={displayName}
                  onChangeText={setDisplayName}
                  placeholder="Nhập tên hiển thị"
                  placeholderTextColor={theme.textMuted}
                  style={[styles.settingsInput, { color: theme.inputText, backgroundColor: theme.inputBackground, borderColor: theme.border }]}
                />
                <AppButton title={saving ? 'Đang lưu...' : 'Lưu thay đổi'} onPress={handleSaveProfile} loading={saving} />
              </SettingsPanel>
            ) : null}
            {activeTab === 'device' ? (
              <SettingsPanel>
                <SettingsRow label="Quyền micro" value={microStatus} />
                <AppButton title="Kiểm tra micro" onPress={handleCheckMicro} variant="secondary" />
              </SettingsPanel>
            ) : null}
            {activeTab === 'appearance' ? (
              <SettingsPanel>
                <View style={styles.themeSegment}>
                  <ThemeSegmentButton icon="white-balance-sunny" label="Sáng" selected={mode === 'light'} onPress={() => handleTheme('light')} />
                  <ThemeSegmentButton icon="moon-waning-crescent" label="Tối" selected={mode === 'dark'} onPress={() => handleTheme('dark')} />
                </View>
              </SettingsPanel>
            ) : null}
            {activeTab === 'notifications' ? (
              <SettingsPanel>
                <EmptyState title="Thông báo đang được hoàn thiện" message="Chưa có backend thông báo trong batch này. Các tuỳ chọn sẽ được nối khi API sẵn sàng." />
              </SettingsPanel>
            ) : null}
            {activeTab === 'privacy' ? (
              <SettingsPanel>
                <Text style={[styles.settingsParagraph, { color: theme.textMuted }]}>Audio được dùng cho chẩn đoán phát âm Phoenix.</Text>
                <Text style={[styles.settingsParagraph, { color: theme.textMuted }]}>Giáo viên chỉ xem dữ liệu thuộc lớp mình phụ trách.</Text>
                <Text style={[styles.settingsParagraph, { color: theme.textMuted }]}>Quản trị viên quản lý dữ liệu hệ thống phục vụ vận hành.</Text>
              </SettingsPanel>
            ) : null}
            {activeTab === 'logout' ? (
              <SettingsPanel>
                <Text style={[styles.settingsParagraph, { color: theme.textMuted }]}>Đăng xuất khỏi phiên Phoenix hiện tại.</Text>
                <AppButton title={signingOut ? 'Đang đăng xuất...' : 'Đăng xuất'} onPress={handleSignOut} loading={signingOut} variant="secondary" />
              </SettingsPanel>
            ) : null}
          </View>
        </View>
      </View>
    </Modal>
  );
}

function SettingsPanel({ children }: PropsWithChildren) {
  return <View style={styles.settingsPanel}>{children}</View>;
}

function SettingsRow({ label, value }: { label: string; value: string }) {
  const { theme } = useTheme();
  return (
    <View style={[styles.settingsRow, { borderColor: theme.border }]}>
      <Text style={[styles.settingsRowLabel, { color: theme.textMuted }]}>{label}</Text>
      <Text style={[styles.settingsRowValue, { color: theme.text }]}>{value}</Text>
    </View>
  );
}

function ThemeSegmentButton({
  icon,
  label,
  selected,
  onPress,
}: {
  icon: IconName;
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  const { theme } = useTheme();
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={[styles.themeSegmentButton, selected ? { backgroundColor: theme.primary, borderColor: theme.primary } : { borderColor: theme.border }]}>
      <MaterialCommunityIcons name={icon} size={18} color={selected ? theme.primaryText : theme.primary} />
      <Text style={[styles.themeSegmentText, { color: selected ? theme.primaryText : theme.text }]}>{label}</Text>
    </Pressable>
  );
}

function NoticeInline({ message }: { message: string }) {
  return (
    <View style={styles.noticeInline}>
      <MaterialCommunityIcons name="information-outline" size={18} color={colors.primary} />
      <Text style={styles.noticeInlineText}>{message}</Text>
    </View>
  );
}

function getInitials(value: string): string {
  const parts = value.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }
  return (value.trim()[0] ?? 'P').toUpperCase();
}

export function AppScreen({
  children,
  scroll = true,
  maxWidth = 920,
}: PropsWithChildren<{ scroll?: boolean; maxWidth?: number }>) {
  const { theme } = useTheme();
  const { width } = useWindowDimensions();
  const isDesktop = width >= 768;
  const content = <View style={[styles.screenContent, { maxWidth }]}>{children}</View>;

  if (isDesktop) {
    return (
      <View style={[styles.screenRoot, { backgroundColor: theme.background }]}>
        {scroll ? (
          <ScrollView contentContainerStyle={styles.screenScrollContent}>{content}</ScrollView>
        ) : (
          content
        )}
      </View>
    );
  }

  return (
    <SafeAreaView style={[styles.screenRoot, { backgroundColor: theme.background }]}>
      {scroll ? (
        <ScrollView
          contentContainerStyle={styles.screenScrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {content}
        </ScrollView>
      ) : (
        content
      )}
    </SafeAreaView>
  );
}

const toneStyles = StyleSheet.create({
  default: {},
  blue: {
    backgroundColor: colors.softBlue,
    borderColor: '#BFDBFE',
  },
  teal: {
    backgroundColor: colors.softTeal,
    borderColor: '#CCFBF1',
  },
  orange: {
    backgroundColor: colors.softOrange,
    borderColor: '#FED7AA',
  },
});

const darkToneStyles = StyleSheet.create({
  default: {},
  blue: {
    backgroundColor: '#172554',
    borderColor: '#1D4ED8',
  },
  teal: {
    backgroundColor: '#134E4A',
    borderColor: '#0F766E',
  },
  orange: {
    backgroundColor: '#431407',
    borderColor: '#C2410C',
  },
});

const buttonStyles = StyleSheet.create({
  primary: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 14,
    elevation: 5,
  },
  secondary: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
  },
  ghost: {
    backgroundColor: colors.softBlue,
    borderColor: '#BFDBFE',
  },
});

const badgeStyles = StyleSheet.create({
  idle: {
    backgroundColor: colors.muted,
  },
  processing: {
    backgroundColor: colors.warning,
  },
  success: {
    backgroundColor: colors.success,
  },
  warning: {
    backgroundColor: colors.accent,
  },
  error: {
    backgroundColor: colors.error,
  },
  primary: {
    backgroundColor: colors.primary,
  },
});

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderRadius: 24,
    borderWidth: 1,
    padding: 20,
    gap: 14,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.05,
    shadowRadius: 18,
    elevation: 3,
  },
  button: {
    minHeight: 56,
    borderRadius: 18,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  buttonDisabled: {
    opacity: 0.58,
    shadowOpacity: 0,
    elevation: 0,
  },
  buttonText: {
    fontSize: 15,
    fontWeight: '800',
  },
  primaryButtonText: {
    color: '#FFFFFF',
  },
  secondaryButtonText: {
    color: colors.text,
  },
  disabledButtonText: {
    color: '#94A3B8',
  },
  sectionHeader: {
    gap: 6,
  },
  centered: {
    alignItems: 'center',
  },
  centeredText: {
    textAlign: 'center',
  },
  eyebrow: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
    lineHeight: 30,
  },
  sectionSubtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
  },
  stateCard: {
    alignItems: 'center',
    paddingVertical: 28,
  },
  stateMark: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.softTeal,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stateMarkDot: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: colors.secondary,
  },
  stateTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: '900',
    textAlign: 'center',
  },
  stateMessage: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
  },
  errorState: {
    backgroundColor: colors.softRed,
    borderColor: '#FECACA',
    borderRadius: 18,
    borderWidth: 1,
    padding: 16,
    gap: 6,
  },
  errorTitle: {
    color: colors.error,
    fontSize: 15,
    fontWeight: '900',
  },
  errorMessage: {
    color: '#991B1B',
    fontSize: 14,
    lineHeight: 20,
  },
  badge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  badgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
  },
  sidebar: {
    width: 256,
    height: '100%',
    backgroundColor: '#F3F3FE',
    borderRightWidth: 1,
    borderRightColor: '#C3C6D7',
    paddingTop: 24,
    paddingHorizontal: 8,
  },
  roleSidebar: {
    width: 260,
    paddingHorizontal: 16,
    paddingVertical: 24,
  },
  sidebarBrand: {
    paddingHorizontal: 8,
    marginBottom: 32,
  },
  sidebarTitle: {
    color: '#004AC6',
    fontSize: 20,
    fontWeight: '900',
  },
  sidebarSubtitle: {
    color: '#434655',
    fontSize: 12,
    marginTop: 4,
  },
  sidebarNav: {
    gap: 8,
  },
  roleSidebarSpacer: {
    flex: 1,
  },
  roleUtilityNav: {
    borderTopWidth: 1,
    paddingTop: 14,
    gap: 6,
  },
  userMiniCard: {
    marginTop: 8,
    minHeight: 58,
    borderRadius: 14,
    borderWidth: 1,
    padding: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  userMiniIdentity: {
    flex: 1,
    minWidth: 0,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
  },
  userMiniAvatarWrap: {
    position: 'relative',
  },
  userMiniText: {
    flex: 1,
    minWidth: 0,
  },
  userMiniName: {
    fontSize: 13,
    fontWeight: '800',
  },
  userMiniRole: {
    fontSize: 11,
    fontWeight: '700',
    marginTop: 2,
  },
  userMiniIconButton: {
    width: 34,
    height: 34,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileNotificationDot: {
    position: 'absolute',
    top: -1,
    right: -1,
    width: 9,
    height: 9,
    borderRadius: 999,
    backgroundColor: colors.accent,
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  avatarShell: {
    backgroundColor: colors.softTeal,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: {
    resizeMode: 'cover',
  },
  avatarInitials: {
    color: '#0F766E',
    fontWeight: '900',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.28)',
    justifyContent: 'flex-end',
    alignItems: 'flex-start',
    padding: 18,
  },
  quickView: {
    width: 360,
    maxWidth: '100%',
    borderRadius: 18,
    borderWidth: 1,
    padding: 18,
    gap: 16,
  },
  quickHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  quickIdentity: {
    flex: 1,
    minWidth: 0,
    gap: 4,
  },
  quickName: {
    fontSize: 20,
    fontWeight: '900',
  },
  quickEmail: {
    fontSize: 13,
    fontWeight: '700',
  },
  quickBadge: {
    alignSelf: 'flex-start',
    marginTop: 4,
  },
  quickActions: {
    gap: 10,
  },
  settingsOverlay: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.42)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 18,
  },
  settingsModal: {
    width: '100%',
    maxWidth: 980,
    height: '88%',
    maxHeight: 720,
    borderRadius: 20,
    borderWidth: 1,
    overflow: 'hidden',
    flexDirection: 'row',
  },
  settingsNav: {
    width: 238,
    borderRightWidth: 1,
    padding: 16,
    gap: 6,
  },
  settingsNavTitle: {
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  settingsNavItem: {
    minHeight: 42,
    borderRadius: 10,
    paddingHorizontal: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
  },
  settingsNavText: {
    fontSize: 13,
    fontWeight: '800',
  },
  settingsContent: {
    flex: 1,
    padding: 22,
    gap: 14,
  },
  settingsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  settingsTitle: {
    fontSize: 24,
    fontWeight: '900',
  },
  settingsCloseButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingsPanel: {
    gap: 14,
  },
  settingsRow: {
    minHeight: 58,
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    justifyContent: 'center',
    gap: 3,
  },
  settingsRowLabel: {
    fontSize: 12,
    fontWeight: '800',
  },
  settingsRowValue: {
    fontSize: 15,
    fontWeight: '800',
  },
  settingsAvatarRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  settingsAvatarActions: {
    flex: 1,
    gap: 8,
  },
  settingsLabel: {
    fontSize: 13,
    fontWeight: '900',
  },
  settingsInput: {
    minHeight: 48,
    borderRadius: 12,
    borderWidth: 1,
    paddingHorizontal: 12,
    fontSize: 15,
    fontWeight: '700',
  },
  settingsParagraph: {
    fontSize: 14,
    lineHeight: 21,
    fontWeight: '700',
  },
  themeSegment: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  themeSegmentButton: {
    minHeight: 44,
    borderRadius: 12,
    borderWidth: 1,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  themeSegmentText: {
    fontSize: 14,
    fontWeight: '900',
  },
  noticeInline: {
    minHeight: 42,
    borderRadius: 12,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 12,
  },
  noticeInlineText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '800',
  },
  sidebarItem: {
    minHeight: 48,
    borderRadius: 8,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  roleSidebarItem: {
    borderRadius: 12,
    paddingLeft: 14,
    paddingRight: 12,
    gap: 12,
    position: 'relative',
  },
  roleSidebarItemActive: {
    backgroundColor: '#ECFDF5',
  },
  roleUtilityItem: {
    minHeight: 44,
  },
  roleNavBar: {
    width: 4,
    alignSelf: 'stretch',
    borderRadius: 999,
    backgroundColor: 'transparent',
    marginLeft: -8,
  },
  roleNavBarActive: {
    backgroundColor: '#14B8A6',
  },
  sidebarItemActive: {
    backgroundColor: '#004AC6',
  },
  sidebarItemDisabled: {
    opacity: 0.72,
  },
  sidebarItemText: {
    color: '#434655',
    fontSize: 14,
    fontWeight: '600',
  },
  sidebarItemTextActive: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  roleSidebarItemTextActive: {
    fontWeight: '700',
  },
  menuNotificationBadge: {
    minWidth: 20,
    height: 20,
    borderRadius: 999,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
    marginLeft: 'auto',
  },
  menuNotificationText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '800',
  },
  roleUtilityText: {
    fontSize: 13,
  },
  sidebarItemTextDisabled: {
    color: '#737686',
  },
  screenRoot: {
    flex: 1,
    backgroundColor: '#FAF8FF',
  },
  screenContent: {
    width: '100%',
    maxWidth: 920,
    alignSelf: 'center',
    paddingHorizontal: 20,
    paddingVertical: 24,
    gap: 16,
  },
  screenScrollContent: {
    flexGrow: 1,
  },
});
