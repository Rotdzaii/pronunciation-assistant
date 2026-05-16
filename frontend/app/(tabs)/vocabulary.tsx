import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, SectionHeader, colors } from '../../components/AppUI';
import { vocabularyQuestions } from '../../lib/vocabularyQuiz';

export default function VocabularyScreen() {
  const router = useRouter();

  return (
    <AppScreen maxWidth={980}>
      <SectionHeader
        eyebrow="Module phụ"
        title="Ôn tập từ vựng"
        subtitle="Một vòng kiểm tra ngắn để nhớ lại từ đã gặp trong bài luyện phát âm."
      />

      <AppCard tone="teal" style={styles.heroCard}>
        <View style={styles.heroIcon}>
          <MaterialCommunityIcons name="cards-playing-outline" size={30} color={colors.primary} />
        </View>
        <View style={styles.heroCopy}>
          <Text style={styles.heroTitle}>Quiz nhanh 3 câu</Text>
          <Text style={styles.heroText}>
            Trả lời từng câu, nhận phản hồi ngay và xem gợi ý ôn tiếp sau khi hoàn thành.
          </Text>
        </View>
        <Pressable style={styles.startButton} onPress={() => router.push('/(tabs)/quiz')}>
          <Text style={styles.startButtonText}>Bắt đầu ôn tập</Text>
        </Pressable>
      </AppCard>

      <View style={styles.wordGrid}>
        {vocabularyQuestions.map((item) => (
          <AppCard key={item.id} style={styles.wordCard}>
            <Text style={styles.word}>{item.word}</Text>
            <Text style={styles.meaning}>{item.answer}</Text>
            <Text style={styles.explanation}>{item.explanation}</Text>
          </AppCard>
        ))}
      </View>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  heroCard: {
    borderRadius: 22,
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 16,
  },
  heroIcon: {
    width: 64,
    height: 64,
    borderRadius: 22,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  heroCopy: {
    flex: 1,
    minWidth: 240,
    gap: 6,
  },
  heroTitle: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '900',
  },
  heroText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  startButton: {
    minHeight: 48,
    borderRadius: 999,
    backgroundColor: colors.primary,
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  startButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '900',
  },
  wordGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  wordCard: {
    flexGrow: 1,
    flexBasis: 240,
    borderRadius: 18,
  },
  word: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  meaning: {
    color: colors.primary,
    fontSize: 15,
    fontWeight: '900',
  },
  explanation: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 20,
  },
});
