import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, colors } from '../../components/AppUI';

export default function QuizResultsScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ correct?: string; total?: string }>();
  const correct = parseNumber(params.correct);
  const total = parseNumber(params.total) || 1;
  const incorrect = Math.max(0, total - correct);
  const percent = Math.round((correct / total) * 100);
  const badge = percent >= 80 ? 'Nhớ rất tốt' : percent >= 50 ? 'Đang tiến bộ' : 'Cần ôn lại';

  return (
    <AppScreen maxWidth={760}>
      <AppCard style={styles.resultCard}>
        <View style={styles.badgeIcon}>
          <MaterialCommunityIcons name="medal-outline" size={42} color="#FFFFFF" />
        </View>
        <Text style={styles.title}>{badge}</Text>
        <Text style={styles.score}>{percent}%</Text>
        <Text style={styles.subtitle}>
          Bạn trả lời đúng {correct}/{total} câu trong vòng ôn tập từ vựng.
        </Text>

        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{correct}</Text>
            <Text style={styles.statLabel}>Đúng</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>{incorrect}</Text>
            <Text style={styles.statLabel}>Chưa đúng</Text>
          </View>
        </View>

        <View style={styles.suggestion}>
          <Text style={styles.suggestionTitle}>Gợi ý ôn tiếp</Text>
          <Text style={styles.suggestionText}>
            Quay lại luyện phát âm một từ bạn vừa ôn, sau đó ghi âm để kiểm tra cách đọc.
          </Text>
        </View>

        <View style={styles.actions}>
          <Pressable style={styles.primaryButton} onPress={() => router.replace('/(tabs)/quiz')}>
            <Text style={styles.primaryButtonText}>Làm lại quiz</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => router.replace('/(tabs)/practice')}>
            <Text style={styles.secondaryButtonText}>Luyện phát âm</Text>
          </Pressable>
        </View>
      </AppCard>
    </AppScreen>
  );
}

function parseNumber(value: string | string[] | undefined) {
  const raw = Array.isArray(value) ? value[0] : value;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

const styles = StyleSheet.create({
  resultCard: {
    borderRadius: 26,
    alignItems: 'center',
    gap: 14,
  },
  badgeIcon: {
    width: 82,
    height: 82,
    borderRadius: 30,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    color: colors.text,
    fontSize: 26,
    fontWeight: '900',
    textAlign: 'center',
  },
  score: {
    color: colors.primary,
    fontSize: 52,
    lineHeight: 58,
    fontWeight: '900',
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    textAlign: 'center',
  },
  statsRow: {
    width: '100%',
    flexDirection: 'row',
    gap: 12,
  },
  statBox: {
    flex: 1,
    minHeight: 92,
    borderRadius: 18,
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  statValue: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '900',
  },
  statLabel: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
  suggestion: {
    width: '100%',
    borderRadius: 18,
    backgroundColor: colors.softTeal,
    borderColor: '#CCFBF1',
    borderWidth: 1,
    padding: 16,
    gap: 6,
  },
  suggestionTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  suggestionText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  actions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  primaryButton: {
    minHeight: 50,
    borderRadius: 999,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '900',
  },
  secondaryButton: {
    minHeight: 50,
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  secondaryButtonText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
});
