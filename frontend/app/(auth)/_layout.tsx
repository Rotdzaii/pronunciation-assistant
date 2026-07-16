import { Stack, usePathname } from 'expo-router';
import { View } from 'react-native';
import { LoadingState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';

export default function AuthLayout() {
  const { loading } = useAuth();
  const pathname = usePathname();

  // The confirmation callback owns its own progress state and must remain
  // visible while AuthProvider restores or exchanges a new session.
  if (loading && pathname !== '/callback') {
    return (
      <View
        style={{
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: colors.background,
          padding: 20,
        }}
      >
        <LoadingState
          title="Đang kiểm tra phiên đăng nhập"
          message="Vui lòng chờ trong giây lát."
        />
      </View>
    );
  }

  return <Stack screenOptions={{ headerShown: false }} />;
}
