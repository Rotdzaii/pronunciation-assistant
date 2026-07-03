import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, ErrorState, SectionHeader, colors } from '../../components/AppUI';
import { fetchVocabularyItems } from '../../lib/api';
import type { VocabularyItem } from '../../types';

export default function VocabularyScreen() {
  const router = useRouter();
  const [items, setItems] = useState<VocabularyItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setFetchError(null);
    fetchVocabularyItems(20, 0)
      .then(data => { if (!cancelled) setItems(data.items); })
      .catch(err => { if (!cancelled) setFetchError(err instanceof Error ? err.message : 'Không tải được danh sách từ.'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

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

      {loading ? (
        <ActivityIndicator color={colors.primary} style={styles.loader} />
      ) : fetchError ? (
        <ErrorState title="Không tải được từ vựng" message={fetchError} />
      ) : (
        <View style={styles.wordGrid}>
          {items.map((item) => (
            <AppCard key={item.id} style={styles.wordCard}>
              <Text style={styles.word}>{item.word}</Text>
              <Text style={styles.meaning}>{item.meaning_vi ?? ''}</Text>
              <Text style={styles.explanation}>{item.phonetic ?? ''}</Text>
            </AppCard>
          ))}
        </View>
      )}
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
  loader: {
    marginTop: 40,
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
