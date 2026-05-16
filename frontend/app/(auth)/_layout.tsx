import { Redirect, Stack } from 'expo-router';
import { View } from 'react-native';
import { LoadingState, colors } from '../../components/AppUI';
import { useAuth } from '../../lib/auth';

export default function AuthLayout() {
  const { session, loading } = useAuth();

  if (loading) {
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

  if (session) {
    return <Redirect href="/(tabs)" />;
  }

  return <Stack screenOptions={{ headerShown: false }} />;
}
