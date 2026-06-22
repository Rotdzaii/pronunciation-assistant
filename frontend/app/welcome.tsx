import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import type { ReactNode } from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, LoadingState } from '../components/AppUI';
import { useAuth } from '../lib/auth';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

const benefits = [
  {
    icon: 'microphone-outline' as IconName,
    title: 'Chấm phát âm bằng AI',
    text: 'Ghi âm, nhận phản hồi rõ ràng bằng tiếng Việt và biết âm nào cần sửa.',
  },
  {
    icon: 'chart-line' as IconName,
    title: 'Theo dõi tiến bộ',
    text: 'Lưu lịch sử luyện tập, điểm số và nhóm lỗi thường gặp theo thời gian.',
  },
  {
    icon: 'school-outline' as IconName,
    title: 'Phù hợp lớp học',
    text: 'Giáo viên có khu vực riêng để theo dõi lớp, học viên và xu hướng lỗi.',
  },
];

const modes = ['Luyện từ', 'Luyện câu', 'Chủ đề giao tiếp', 'Thư viện lỗi', 'Ôn tập từ vựng'];

function getHomePathForRole(role: string | null) {
  if (role === 'admin') return '/(tabs)/admin';
  if (role === 'teacher') return '/(tabs)/teacher';
  return '/(tabs)';
}
export default function WelcomeScreen() {
  const router = useRouter();
  const { appRole, session, loading } = useAuth();
  const { width } = useWindowDimensions();
  const isWide = width >= 900;

  if (loading) {
    return (
      <View style={styles.loadingShell}>
        <LoadingState title="Đang mở ứng dụng" message="Hệ thống đang kiểm tra phiên đăng nhập." />
      </View>
    );
  }

  if (session) {
    return (
      <View style={styles.loadingShell}>
        <LoadingState title="Bạn đã đăng nhập" message="Chọn tiếp tục để vào khu vực học tập phù hợp với tài khoản." />
        <Pressable
          accessibilityRole="button"
          onPress={() => router.replace(getHomePathForRole(appRole))}
          style={styles.primaryButton}
        >
          <Text style={styles.primaryButtonText}>Tiếp tục</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.nav}>
          <View style={styles.brandRow}>
            <View style={styles.brandMark}>
              <MaterialCommunityIcons name="waveform" size={24} color="#FFFFFF" />
            </View>
            <Text style={styles.brandText}>Pronunciation Assistant</Text>
          </View>
          <Pressable onPress={() => router.push('/(auth)/login')} style={styles.navLogin}>
            <Text style={styles.navLoginText}>Đăng nhập</Text>
          </Pressable>
        </View>

        <View style={[styles.hero, isWide ? styles.heroWide : null]}>
          <View style={styles.heroCopy}>
            <View style={styles.eyebrowPill}>
              <MaterialCommunityIcons name="star-four-points-outline" size={16} color={colors.primary} />
              <Text style={styles.eyebrowText}>AI pronunciation coach</Text>
            </View>
            <Text style={styles.heroTitle}>Luyện phát âm tiếng Anh tự tin hơn mỗi ngày</Text>
            <Text style={styles.heroText}>
              Một trải nghiệm học nhẹ nhàng cho người Việt: ghi âm, nhận góp ý cụ thể,
              luyện lại đúng lỗi và theo dõi tiến bộ theo từng buổi học.
            </Text>
            <View style={styles.heroActions}>
              <Pressable style={styles.primaryButton} onPress={() => router.push('/(auth)/register')}>
                <Text style={styles.primaryButtonText}>Bắt đầu</Text>
              </Pressable>
              <Pressable style={styles.secondaryButton} onPress={() => router.push('/(auth)/login')}>
                <Text style={styles.secondaryButtonText}>Xem demo</Text>
              </Pressable>
            </View>
          </View>

          <View style={styles.heroVisual}>
            <View style={styles.phoneCard}>
              <View style={styles.phoneHeader}>
                <View>
                  <Text style={styles.phoneLabel}>Bài luyện hôm nay</Text>
                  <Text style={styles.phoneWord}>Architecture</Text>
                </View>
                <View style={styles.scoreBubble}>
                  <Text style={styles.scoreBubbleValue}>87</Text>
                  <Text style={styles.scoreBubbleLabel}>điểm</Text>
                </View>
              </View>
              <View style={styles.waveRow}>
                {[18, 32, 46, 28, 54, 36, 22].map((height, index) => (
                  <View key={`${height}-${index}`} style={[styles.waveBar, { height }]} />
                ))}
              </View>
              <View style={styles.aiNote}>
                <MaterialCommunityIcons name="robot-outline" size={20} color={colors.primary} />
                <Text style={styles.aiNoteText}>
                  Bạn cần bật rõ âm cuối /t/ hơn và giảm nhấn ở âm tiết cuối.
                </Text>
              </View>
            </View>
          </View>
        </View>

        <Section title="Lợi ích chính">
          <View style={styles.benefitGrid}>
            {benefits.map((item) => (
              <View key={item.title} style={styles.benefitCard}>
                <View style={styles.benefitIcon}>
                  <MaterialCommunityIcons name={item.icon} size={24} color={colors.primary} />
                </View>
                <Text style={styles.benefitTitle}>{item.title}</Text>
                <Text style={styles.benefitText}>{item.text}</Text>
              </View>
            ))}
          </View>
        </Section>

        <Section title="Vì sao nên học cùng chúng tôi?">
          <View style={styles.reasonCard}>
            <Text style={styles.reasonText}>
              Ứng dụng tập trung vào phát âm, không biến bài học thành danh sách điểm số khô cứng.
              Mỗi lượt luyện đều hướng người học đến một hành động nhỏ: nghe lại, sửa một âm,
              rồi luyện tiếp với mục tiêu rõ ràng.
            </Text>
          </View>
        </Section>

        <Section title="Chế độ học nổi bật">
          <View style={styles.modeGrid}>
            {modes.map((mode) => (
              <View key={mode} style={styles.modePill}>
                <Text style={styles.modePillText}>{mode}</Text>
              </View>
            ))}
          </View>
        </Section>

        <View style={styles.highlightCard}>
          <View style={styles.highlightIcon}>
            <MaterialCommunityIcons name="headphones" size={28} color="#FFFFFF" />
          </View>
          <View style={styles.highlightCopy}>
            <Text style={styles.highlightTitle}>AI phát hiện lỗi phát âm theo ngữ cảnh</Text>
            <Text style={styles.highlightText}>
              Phần phản hồi ưu tiên giải thích dễ hiểu: âm nào sai, vì sao sai và nên luyện lại thế nào.
            </Text>
          </View>
        </View>

        <View style={styles.footerCta}>
          <Text style={styles.footerTitle}>Sẵn sàng luyện thử?</Text>
          <Text style={styles.footerText}>
            Tạo tài khoản học viên hoặc đăng nhập để bắt đầu ghi âm bài đầu tiên.
          </Text>
          <View style={styles.heroActions}>
            <Pressable style={styles.primaryButton} onPress={() => router.push('/(auth)/register')}>
              <Text style={styles.primaryButtonText}>Bắt đầu</Text>
            </Pressable>
            <Pressable style={styles.secondaryButton} onPress={() => router.push('/(auth)/login')}>
              <Text style={styles.secondaryButtonText}>Đăng nhập</Text>
            </Pressable>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#F8FAFC',
  },
  loadingShell: {
    flex: 1,
    backgroundColor: '#F8FAFC',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  scrollContent: {
    width: '100%',
    maxWidth: 1180,
    alignSelf: 'center',
    padding: 20,
    gap: 28,
  },
  nav: {
    minHeight: 56,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  brandMark: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandText: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  navLogin: {
    minHeight: 42,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  navLoginText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
  hero: {
    gap: 24,
    borderRadius: 28,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    padding: 22,
    overflow: 'hidden',
  },
  heroWide: {
    minHeight: 520,
    flexDirection: 'row',
    alignItems: 'center',
    padding: 34,
  },
  heroCopy: {
    flex: 1,
    gap: 18,
  },
  eyebrowPill: {
    alignSelf: 'flex-start',
    minHeight: 34,
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 12,
  },
  eyebrowText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  heroTitle: {
    color: colors.text,
    fontSize: 44,
    lineHeight: 52,
    fontWeight: '900',
    maxWidth: 620,
  },
  heroText: {
    color: colors.muted,
    fontSize: 17,
    lineHeight: 27,
    maxWidth: 620,
  },
  heroActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  primaryButton: {
    minHeight: 52,
    borderRadius: 999,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  secondaryButton: {
    minHeight: 52,
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  secondaryButtonText: {
    color: colors.primary,
    fontSize: 15,
    fontWeight: '900',
  },
  heroVisual: {
    flex: 1,
    minHeight: 330,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#EEF6FF',
    borderRadius: 26,
    padding: 20,
  },
  phoneCard: {
    width: '100%',
    maxWidth: 390,
    borderRadius: 28,
    backgroundColor: colors.surface,
    borderColor: '#BFDBFE',
    borderWidth: 1,
    padding: 22,
    gap: 22,
  },
  phoneHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  phoneLabel: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
  phoneWord: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '900',
  },
  scoreBubble: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: colors.softTeal,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 6,
    borderColor: colors.secondary,
  },
  scoreBubbleValue: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  scoreBubbleLabel: {
    color: colors.muted,
    fontSize: 10,
    fontWeight: '800',
  },
  waveRow: {
    height: 74,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  waveBar: {
    width: 10,
    borderRadius: 999,
    backgroundColor: colors.primary,
  },
  aiNote: {
    borderRadius: 18,
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderWidth: 1,
    padding: 14,
    flexDirection: 'row',
    gap: 10,
  },
  aiNoteText: {
    flex: 1,
    color: colors.text,
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '700',
  },
  section: {
    gap: 14,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
  },
  benefitGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
  },
  benefitCard: {
    flexGrow: 1,
    flexBasis: 260,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 18,
    gap: 10,
  },
  benefitIcon: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  benefitTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  benefitText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  reasonCard: {
    borderRadius: 22,
    backgroundColor: colors.softTeal,
    borderColor: '#CCFBF1',
    borderWidth: 1,
    padding: 20,
  },
  reasonText: {
    color: colors.text,
    fontSize: 16,
    lineHeight: 26,
    fontWeight: '700',
  },
  modeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  modePill: {
    borderRadius: 999,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  modePillText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  highlightCard: {
    borderRadius: 26,
    backgroundColor: colors.primary,
    padding: 22,
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 16,
  },
  highlightIcon: {
    width: 58,
    height: 58,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  highlightCopy: {
    flex: 1,
    minWidth: 240,
    gap: 6,
  },
  highlightTitle: {
    color: '#FFFFFF',
    fontSize: 21,
    fontWeight: '900',
  },
  highlightText: {
    color: '#DBEAFE',
    fontSize: 15,
    lineHeight: 23,
  },
  footerCta: {
    borderRadius: 26,
    backgroundColor: colors.surface,
    borderColor: colors.border,
    borderWidth: 1,
    padding: 24,
    gap: 12,
    alignItems: 'center',
  },
  footerTitle: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '900',
    textAlign: 'center',
  },
  footerText: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
});
