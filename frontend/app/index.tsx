import { useEffect, useRef } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { LoadingState, colors } from '../components/AppUI';
import { useAuth } from '../lib/auth';

export default function Index() {
  const router = useRouter();
  const didRouteRef = useRef(false);
  const { appRole, session, loading, roleLoading } = useAuth();

  useEffect(() => {
    if (didRouteRef.current || loading || roleLoading) {
      return;
    }

    didRouteRef.current = true;
    if (session) {
      if (appRole === 'admin') {
        router.replace('/(tabs)/admin');
      } else if (appRole === 'teacher') {
        router.replace('/(tabs)/teacher');
      } else {
        router.replace('/(tabs)');
      }
    } else {
      router.replace('/welcome');
    }
  }, [appRole, loading, roleLoading, session?.user?.id, router]);

  return (
    <View style={styles.container}>
      <View style={styles.brandMark}>
        <Text style={styles.brandMarkText}>PA</Text>
      </View>
      <LoadingState
        title="Đang mở Pronunciation Assistant"
        message="Hệ thống đang kiểm tra phiên đăng nhập của bạn."
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    gap: 16,
  },
  brandMark: {
    width: 64,
    height: 64,
    borderRadius: 22,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.22,
    shadowRadius: 18,
    elevation: 6,
  },
  brandMarkText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '900',
  },
});
