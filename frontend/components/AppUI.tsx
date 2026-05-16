import { PropsWithChildren } from 'react';
import { type Href, usePathname, useRouter } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleProp,
  StyleSheet,
  Text,
  TextStyle,
  useWindowDimensions,
  View,
  ViewStyle,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

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
};

type SidebarVariant = 'student' | 'teacher';

const studentSidebarItems: SidebarItemConfig[] = [
  { label: 'Trang chủ', href: '/(tabs)', icon: 'home-outline', match: ['/', '/(tabs)'] },
  {
    label: 'Luyện tập',
    href: '/(tabs)/practice',
    icon: 'microphone-outline',
    match: ['/practice', '/(tabs)/practice'],
  },
  {
    label: 'Lỗi phổ biến',
    href: '/(tabs)/mistakes',
    icon: 'alert-circle-outline',
    match: ['/mistakes', '/(tabs)/mistakes'],
  },
  {
    label: 'Lịch sử',
    href: '/(tabs)/history',
    icon: 'history',
    match: ['/history', '/(tabs)/history'],
  },
  {
    label: 'Tiến độ',
    href: '/(tabs)/progress',
    icon: 'chart-line',
    match: ['/progress', '/(tabs)/progress'],
  },
  {
    label: 'Ôn từ vựng',
    href: '/(tabs)/vocabulary',
    icon: 'cards-outline',
    match: [
      '/vocabulary',
      '/(tabs)/vocabulary',
      '/quiz',
      '/(tabs)/quiz',
      '/quiz-results',
      '/(tabs)/quiz-results',
    ],
  },
  {
    label: 'Hồ sơ',
    href: '/(tabs)/profile',
    icon: 'account-outline',
    match: ['/profile', '/(tabs)/profile'],
  },
];

const teacherSidebarItems: SidebarItemConfig[] = [
  {
    label: 'Tổng quan',
    href: '/(tabs)/teacher',
    icon: 'view-dashboard-outline',
    match: ['/teacher', '/(tabs)/teacher'],
  },
  {
    label: 'Học viên',
    href: '/(tabs)/students',
    icon: 'account-group-outline',
    match: ['/students', '/(tabs)/students', '/student-detail', '/(tabs)/student-detail'],
  },
  {
    label: 'Phân tích lớp',
    icon: 'chart-line',
    match: [],
  },
  {
    label: 'Lỗi phổ biến',
    icon: 'alert-circle-outline',
    match: [],
  },
  {
    label: 'Giao bài luyện',
    icon: 'clipboard-edit-outline',
    match: [],
  },
  {
    label: 'Báo cáo',
    icon: 'file-chart-outline',
    match: [],
  },
  {
    label: 'Hồ sơ',
    href: '/(tabs)/profile',
    icon: 'account-outline',
    match: ['/profile', '/(tabs)/profile'],
  },
];

export function AppCard({ children, style, tone = 'default' }: AppCardProps) {
  return <View style={[styles.card, toneStyles[tone], style]}>{children}</View>;
}

export function AppButton({
  title,
  onPress,
  disabled = false,
  loading = false,
  variant = 'primary',
  style,
}: AppButtonProps) {
  const isDisabled = disabled || loading;

  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={isDisabled}
      style={[
        styles.button,
        buttonStyles[variant],
        isDisabled ? styles.buttonDisabled : null,
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#FFFFFF' : colors.primary} />
      ) : (
        <Text
          style={[
            styles.buttonText,
            variant === 'primary' ? styles.primaryButtonText : styles.secondaryButtonText,
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
  const centered = align === 'center';

  return (
    <View style={[styles.sectionHeader, centered ? styles.centered : null]}>
      {eyebrow ? <Text style={styles.eyebrow}>{eyebrow}</Text> : null}
      <Text style={[styles.sectionTitle, centered ? styles.centeredText : null]}>{title}</Text>
      {subtitle ? (
        <Text style={[styles.sectionSubtitle, centered ? styles.centeredText : null]}>
          {subtitle}
        </Text>
      ) : null}
    </View>
  );
}

export function EmptyState({ title, message }: StateProps) {
  return (
    <AppCard style={styles.stateCard}>
      <View style={styles.stateMark}>
        <View style={styles.stateMarkDot} />
      </View>
      <Text style={styles.stateTitle}>{title}</Text>
      {message ? <Text style={styles.stateMessage}>{message}</Text> : null}
    </AppCard>
  );
}

export function LoadingState({ title, message }: StateProps) {
  return (
    <AppCard style={styles.stateCard}>
      <ActivityIndicator color={colors.primary} />
      <Text style={styles.stateTitle}>{title}</Text>
      {message ? <Text style={styles.stateMessage}>{message}</Text> : null}
    </AppCard>
  );
}

export function ErrorState({ title, message }: StateProps) {
  return (
    <View style={styles.errorState}>
      <Text style={styles.errorTitle}>{title}</Text>
      {message ? <Text style={styles.errorMessage}>{message}</Text> : null}
    </View>
  );
}

export function StatusBadge({
  label,
  tone = 'idle',
  style,
  textStyle,
}: StatusBadgeProps) {
  return (
    <View style={[styles.badge, badgeStyles[tone], style]}>
      <Text style={[styles.badgeText, textStyle]}>{label}</Text>
    </View>
  );
}

export function AppSidebar({ variant = 'student' }: { variant?: SidebarVariant }) {
  const pathname = usePathname();
  const router = useRouter();
  const items = variant === 'teacher' ? teacherSidebarItems : studentSidebarItems;

  return (
    <View style={styles.sidebar}>
      <View style={styles.sidebarBrand}>
        <Text style={styles.sidebarTitle}>Trợ lý Phát âm</Text>
        <Text style={styles.sidebarSubtitle}>Học cùng AI</Text>
      </View>

      <View style={styles.sidebarNav}>
        {items.map((item) => {
          const isActive = item.match.includes(pathname);
          const href = item.href;

          return (
            <Pressable
              key={item.label}
              accessibilityRole="link"
              onPress={href ? () => router.push(href) : undefined}
              disabled={!href}
              style={[
                styles.sidebarItem,
                isActive ? styles.sidebarItemActive : null,
                !href ? styles.sidebarItemDisabled : null,
              ]}
            >
              <MaterialCommunityIcons
                name={item.icon}
                size={24}
                color={isActive ? '#FFFFFF' : href ? '#434655' : '#737686'}
              />
              <Text
                style={[
                  styles.sidebarItemText,
                  isActive ? styles.sidebarItemTextActive : null,
                  !href ? styles.sidebarItemTextDisabled : null,
                ]}
              >
                {item.label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

export function AppScreen({
  children,
  scroll = true,
  maxWidth = 920,
}: PropsWithChildren<{ scroll?: boolean; maxWidth?: number }>) {
  const { width } = useWindowDimensions();
  const isDesktop = width >= 768;
  const content = <View style={[styles.screenContent, { maxWidth }]}>{children}</View>;

  if (isDesktop) {
    return (
      <View style={styles.screenRoot}>
        {scroll ? (
          <ScrollView contentContainerStyle={styles.screenScrollContent}>{content}</ScrollView>
        ) : (
          content
        )}
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.screenRoot}>
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
  sidebarItem: {
    minHeight: 48,
    borderRadius: 8,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
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
