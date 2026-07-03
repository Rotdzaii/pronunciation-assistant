import { Tabs, usePathname } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Pressable, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { AppSidebar, ErrorState, LoadingState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';
import { useTheme } from '../../lib/theme';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type TabIconProps = {
  focused: boolean;
  name: IconName;
};


export default function TabsLayout() {
  const { width } = useWindowDimensions();
  const pathname = usePathname();
  const { appRole, loading, roleError, roleLoading } = useAuth();
  const { theme } = useTheme();
  const isDesktop = width >= 768;
  const isTeacherRoute = [
    '/teacher',
    '/(tabs)/teacher',
    '/students',
    '/(tabs)/students',
    '/student-detail',
    '/(tabs)/student-detail',
  ].includes(pathname);
  const isAdminRoute = ['/admin', '/(tabs)/admin'].includes(pathname);
  console.debug('[TabsLayout] render', { role: appRole, pathname });

  if (loading || (roleLoading && !appRole)) {
    return (
      <View style={[styles.loadingShell, { backgroundColor: theme.background }]}>
        <LoadingState
          title="Đang kiểm tra phiên đăng nhập"
          message="Vui lòng chờ trong giây lát."
        />
      </View>
    );
  }


  if (roleError || !appRole) {
    return (
      <View style={[styles.loadingShell, { backgroundColor: theme.background }]}>
        <ErrorState
          title="Không thể mở khu vực học tập"
          message={roleError ?? 'Không thể xác định vai trò tài khoản từ backend.'}
        />
      </View>
    );
  }

  const canUseStudent = appRole === 'student';
  const canUseTeacher = appRole === 'teacher';
  const canUseAdmin = appRole === 'admin';
  const sidebarVariant = appRole;

  return (
    <View style={[styles.shell, { backgroundColor: theme.background }]}>
      {isDesktop ? <AppSidebar variant={sidebarVariant} /> : null}
      <View style={styles.content}>
        <Tabs
          screenOptions={{
            headerShown: !isDesktop,
            headerStyle: {
              backgroundColor: theme.background,
            },
            headerShadowVisible: false,
            headerTitleAlign: 'center',
            headerTitleStyle: {
              color: theme.text,
              fontSize: 18,
              fontWeight: '800',
            },
            tabBarActiveTintColor: theme.primary,
            tabBarInactiveTintColor: theme.textMuted,
            tabBarLabelStyle: {
              fontSize: 12,
              fontWeight: '800',
              paddingBottom: 4,
            },
            tabBarStyle: [
              styles.tabBar,
              {
                backgroundColor: theme.surface,
                borderTopColor: theme.border,
                shadowColor: theme.shadow,
              },
              isDesktop || isTeacherRoute || isAdminRoute ? styles.hiddenTabBar : null,
            ],
          }}
        >
          <Tabs.Screen
            name="index"
            options={{
              href: canUseStudent ? undefined : null,
              title: 'Trang chủ',
              tabBarIcon: ({ focused }) => <TabIcon focused={focused} name="home-outline" />,
            }}
          />
          <Tabs.Screen
            name="practice"
            options={{
              href: canUseStudent ? undefined : null,
              title: 'Luyện tập',
              tabBarIcon: ({ focused }) => (
                <TabIcon focused={focused} name="microphone-outline" />
              ),
            }}
          />
          <Tabs.Screen
            name="mistakes"
            options={{
              href: canUseStudent ? undefined : null,
              title: 'Lỗi phổ biến',
              tabBarIcon: ({ focused }) => (
                <TabIcon focused={focused} name="alert-circle-outline" />
              ),
            }}
          />
          <Tabs.Screen
            name="history"
            options={{
              href: canUseStudent ? undefined : null,
              title: 'Lịch sử',
              tabBarIcon: ({ focused }) => <TabIcon focused={focused} name="history" />,
            }}
          />
          <Tabs.Screen
            name="profile"
            options={{
              title: 'Hồ sơ',
              tabBarIcon: ({ focused }) => <TabIcon focused={focused} name="account-outline" />,
            }}
          />
          <Tabs.Screen name="practice-mode" options={{ title: 'Chọn chế độ luyện', href: null }} />
          <Tabs.Screen name="sentence" options={{ title: 'Luyện câu', href: null }} />
          <Tabs.Screen name="processing" options={{ title: 'AI đang chấm', href: null }} />
          <Tabs.Screen name="result" options={{ title: 'Kết quả phát âm', href: null }} />
          <Tabs.Screen name="progress" options={{ title: 'Tiến độ', href: null }} />
          <Tabs.Screen name="vocabulary" options={{ title: 'Ôn tập từ vựng', href: null }} />
          <Tabs.Screen name="quiz" options={{ title: 'Câu hỏi ôn tập', href: null }} />
          <Tabs.Screen name="quiz-results" options={{ title: 'Kết quả ôn tập', href: null }} />
          <Tabs.Screen
            name="teacher"
            options={{
              title: 'Bảng điều khiển giáo viên',
              href: canUseTeacher ? '/(tabs)/teacher' : null,
            }}
          />
          <Tabs.Screen
            name="admin"
            options={{
              title: 'Quản trị Phoenix',
              href: canUseAdmin ? '/(tabs)/admin' : null,
            }}
          />
          <Tabs.Screen name="students" options={{ title: 'Danh sách học viên', href: null }} />
          <Tabs.Screen name="student-detail" options={{ title: 'Chi tiết học viên', href: null }} />
          <Tabs.Screen name="assessment/index" options={{ title: 'Kiểm tra', href: null }} />
        </Tabs>
      </View>
    </View>
  );
}

function TabIcon({ focused, name }: TabIconProps) {
  const { theme } = useTheme();
  return (
    <View style={[styles.tabIcon, { backgroundColor: theme.surfaceAlt }, focused ? styles.tabIconActive : null, focused ? { backgroundColor: theme.softBlue } : null]}>
      <MaterialCommunityIcons
        name={name}
        size={22}
        color={focused ? theme.primary : theme.textMuted}
      />
    </View>
  );
}

function AdminModeSwitch({
  isTeacherRoute,
  onStudentMode,
  onTeacherMode,
}: {
  isTeacherRoute: boolean;
  onStudentMode: () => void;
  onTeacherMode: () => void;
}) {
  return (
    <View style={styles.adminSwitch}>
      <Pressable
        accessibilityRole="button"
        onPress={onStudentMode}
        style={[styles.adminSwitchButton, !isTeacherRoute ? styles.adminSwitchButtonActive : null]}
      >
        <Text style={[styles.adminSwitchText, !isTeacherRoute ? styles.adminSwitchTextActive : null]}>
          Chế độ học viên
        </Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        onPress={onTeacherMode}
        style={[styles.adminSwitchButton, isTeacherRoute ? styles.adminSwitchButtonActive : null]}
      >
        <Text style={[styles.adminSwitchText, isTeacherRoute ? styles.adminSwitchTextActive : null]}>
          Chế độ giảng viên
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  loadingShell: {
    flex: 1,
    backgroundColor: '#FAF8FF',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  shell: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#FAF8FF',
  },
  content: {
    flex: 1,
  },
  adminSwitch: {
    position: 'absolute',
    top: 10,
    right: 14,
    zIndex: 10,
    flexDirection: 'row',
    gap: 6,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 4,
  },
  adminSwitchButton: {
    minHeight: 34,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  adminSwitchButtonActive: {
    backgroundColor: colors.primary,
  },
  adminSwitchText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
  },
  adminSwitchTextActive: {
    color: '#FFFFFF',
  },
  tabBar: {
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: 1,
    height: 72,
    paddingTop: 8,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
    elevation: 8,
  },
  hiddenTabBar: {
    display: 'none',
  },
  tabIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
  },
  tabIconActive: {
    backgroundColor: colors.softBlue,
  },
});
