import { Redirect, Tabs, usePathname } from 'expo-router';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { StyleSheet, useWindowDimensions, View } from 'react-native';
import { AppSidebar, LoadingState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type TabIconProps = {
  focused: boolean;
  name: IconName;
};

export default function TabsLayout() {
  const { width } = useWindowDimensions();
  const pathname = usePathname();
  const { session, loading } = useAuth();
  const isDesktop = width >= 768;
  const isTeacherRoute = [
    '/teacher',
    '/(tabs)/teacher',
    '/students',
    '/(tabs)/students',
    '/student-detail',
    '/(tabs)/student-detail',
  ].includes(pathname);

  if (loading) {
    return (
      <View style={styles.loadingShell}>
        <LoadingState
          title="Đang kiểm tra phiên đăng nhập"
          message="Vui lòng chờ trong giây lát."
        />
      </View>
    );
  }

  if (!session) {
    return <Redirect href="/welcome" />;
  }

  return (
    <View style={styles.shell}>
      {isDesktop ? <AppSidebar variant={isTeacherRoute ? 'teacher' : 'student'} /> : null}
      <View style={styles.content}>
        <Tabs
          screenOptions={{
            headerShown: !isDesktop,
            headerStyle: {
              backgroundColor: colors.background,
            },
            headerShadowVisible: false,
            headerTitleAlign: 'center',
            headerTitleStyle: {
              color: colors.text,
              fontSize: 18,
              fontWeight: '800',
            },
            tabBarActiveTintColor: colors.primary,
            tabBarInactiveTintColor: colors.muted,
            tabBarLabelStyle: {
              fontSize: 12,
              fontWeight: '800',
              paddingBottom: 4,
            },
            tabBarStyle: [
              styles.tabBar,
              isDesktop || isTeacherRoute ? styles.hiddenTabBar : null,
            ],
          }}
        >
          <Tabs.Screen
            name="index"
            options={{
              title: 'Trang chủ',
              tabBarIcon: ({ focused }) => <TabIcon focused={focused} name="home-outline" />,
            }}
          />
          <Tabs.Screen
            name="practice"
            options={{
              title: 'Luyện tập',
              tabBarIcon: ({ focused }) => (
                <TabIcon focused={focused} name="microphone-outline" />
              ),
            }}
          />
          <Tabs.Screen
            name="mistakes"
            options={{
              title: 'Lỗi phổ biến',
              tabBarIcon: ({ focused }) => (
                <TabIcon focused={focused} name="alert-circle-outline" />
              ),
            }}
          />
          <Tabs.Screen
            name="history"
            options={{
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
          <Tabs.Screen name="teacher" options={{ title: 'Bảng điều khiển giáo viên', href: null }} />
          <Tabs.Screen name="students" options={{ title: 'Danh sách học viên', href: null }} />
          <Tabs.Screen name="student-detail" options={{ title: 'Chi tiết học viên', href: null }} />
        </Tabs>
      </View>
    </View>
  );
}

function TabIcon({ focused, name }: TabIconProps) {
  return (
    <View style={[styles.tabIcon, focused ? styles.tabIconActive : null]}>
      <MaterialCommunityIcons
        name={name}
        size={22}
        color={focused ? colors.primary : colors.muted}
      />
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
