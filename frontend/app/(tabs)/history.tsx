import { useEffect, useState } from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';
import {
  AppCard,
  EmptyState,
  ErrorState,
  LoadingState,
  SectionHeader,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import { ScreenContainer } from '../../components/ScreenContainer';
import { fetchPracticeHistory } from '../../lib/api';
import { useAuth } from '../../lib/auth';
import type { PracticeHistoryItem } from '../../types';

export default function HistoryScreen() {
  const { accessToken } = useAuth();
  const [history, setHistory] = useState<PracticeHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadHistory = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchPracticeHistory(accessToken);
        setHistory(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không tải được lịch sử.');
      } finally {
        setLoading(false);
      }
    };

    void loadHistory();
  }, [accessToken]);

  return (
    <ScreenContainer>
      <SectionHeader
        eyebrow="Tiến độ cá nhân"
        title="Lịch sử luyện tập"
        subtitle="Theo dõi các lần luyện phát âm của bạn."
      />

      {loading ? (
        <LoadingState
          title="Đang tải lịch sử"
          message="Hệ thống đang lấy các lượt luyện tập gần nhất."
        />
      ) : null}

      {error ? <ErrorState title="Không thể tải lịch sử" message={error} /> : null}

      {!loading && !error ? (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
          renderItem={({ item }) => <HistoryCard item={item} />}
          ListEmptyComponent={
            <EmptyState
              title="Chưa có dữ liệu lịch sử"
              message="Sau khi gửi bài ghi âm cho AI chấm điểm, kết quả sẽ xuất hiện tại đây."
            />
          }
        />
      ) : null}
    </ScreenContainer>
  );
}

function HistoryCard({ item }: { item: PracticeHistoryItem }) {
  const problemText = item.problem_phonemes.join(', ') || 'Chưa phát hiện lỗi nổi bật';

  return (
    <AppCard style={styles.card}>
      <View style={styles.cardTop}>
        <View style={styles.scoreMark}>
          <Text style={styles.scoreValue}>{Math.round(item.score)}</Text>
          <Text style={styles.scoreLabel}>điểm</Text>
        </View>
        <View style={styles.cardInfo}>
          <Text style={styles.cardTitle}>Lượt luyện phát âm</Text>
          <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
          <StatusBadge label="Đã chấm" tone="success" style={styles.badge} />
        </View>
      </View>

      <View style={styles.problemBox}>
        <Text style={styles.problemLabel}>Âm cần chú ý</Text>
        <Text style={styles.problemText}>{problemText}</Text>
      </View>
    </AppCard>
  );
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

const styles = StyleSheet.create({
  list: {
    gap: 14,
    paddingBottom: 16,
  },
  card: {
    gap: 16,
  },
  cardTop: {
    flexDirection: 'row',
    gap: 14,
    alignItems: 'center',
  },
  scoreMark: {
    width: 78,
    height: 78,
    borderRadius: 39,
    backgroundColor: colors.softBlue,
    borderWidth: 6,
    borderColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreValue: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  scoreLabel: {
    color: colors.muted,
    fontSize: 10,
    fontWeight: '800',
  },
  cardInfo: {
    flex: 1,
    gap: 6,
  },
  cardTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: '900',
  },
  cardDate: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 18,
  },
  badge: {
    marginTop: 2,
  },
  problemBox: {
    backgroundColor: colors.background,
    borderColor: colors.border,
    borderRadius: 18,
    borderWidth: 1,
    padding: 14,
    gap: 5,
  },
  problemLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  problemText: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '700',
    lineHeight: 21,
  },
});
