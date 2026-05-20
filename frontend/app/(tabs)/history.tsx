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
import {
  clampScore,
  formatFeedbackLines,
  formatProblemPhonemes,
  getFeedbackScorerBadge,
  getScoreTone,
} from '../../lib/practiceFormatters';
import type { PracticeHistoryItem, PracticeJobStatus } from '../../types';

const SCORE_RING_SEGMENTS = 40;

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
        const data = await fetchPracticeHistory(accessToken, { limit: 20, offset: 0 });
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
  const scoreTone = getScoreTone(item.score);
  const problemLines = formatProblemPhonemes(item.problem_phonemes).slice(0, 2);
  const feedbackLines = formatFeedbackLines(item.feedback).slice(0, 3);
  const scorerBadge = getFeedbackScorerBadge(item.feedback);

  return (
    <AppCard style={styles.card}>
      <View style={styles.cardTop}>
        <ScoreRing score={item.score} />
        <View style={styles.cardInfo}>
          <Text style={styles.cardTitle}>{item.target_word || 'Lượt luyện phát âm'}</Text>
          <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
          <View style={styles.badgeRow}>
            <StatusBadge
              label={getStatusLabel(item.status)}
              tone={getStatusTone(item.status)}
              style={styles.badge}
            />
            {item.score != null ? (
              <View style={[styles.scoreBadge, { backgroundColor: scoreTone.color }]}>
                <Text style={styles.scoreBadgeText}>{scoreTone.label}</Text>
              </View>
            ) : null}
            {scorerBadge ? (
              <View style={styles.scorerBadge}>
                <Text style={styles.scorerBadgeText}>{scorerBadge}</Text>
              </View>
            ) : null}
          </View>
        </View>
      </View>

      <View style={styles.problemBox}>
        <Text style={styles.problemLabel}>Âm cần chú ý</Text>
        {problemLines.map((line, index) => (
          <Text key={`${line}-${index}`} style={styles.problemText}>
            {line}
          </Text>
        ))}
        <Text style={styles.problemLabel}>Nhận xét</Text>
        {feedbackLines.map((line, index) => (
          <Text key={`${line}-${index}`} style={styles.feedbackText}>
            {line}
          </Text>
        ))}
      </View>
    </AppCard>
  );
}

function ScoreRing({ score }: { score: number | null }) {
  const clampedScore = clampScore(score);
  const roundedScore = score == null ? '--' : Math.round(clampedScore);
  const filledSegments = Math.round((clampedScore / 100) * SCORE_RING_SEGMENTS);
  const scoreTone = getScoreTone(score);

  return (
    <View style={[styles.scoreRing, { backgroundColor: scoreTone.softColor }]}>
      {Array.from({ length: SCORE_RING_SEGMENTS }).map((_, index) => {
        const isFilled = index < filledSegments;
        return (
          <View
            key={index}
            style={[
              styles.scoreRingSegment,
              {
                backgroundColor: isFilled ? scoreTone.color : colors.border,
                transform: [
                  { rotate: `${(360 / SCORE_RING_SEGMENTS) * index}deg` },
                  { translateY: -34 },
                ],
              },
            ]}
          />
        );
      })}
      <View style={styles.scoreRingCenter}>
        <Text style={styles.scoreValue}>{roundedScore}</Text>
        <Text style={styles.scoreLabel}>điểm</Text>
      </View>
    </View>
  );
}

function getStatusLabel(status: PracticeJobStatus) {
  if (status === 'completed') {
    return 'Đã chấm';
  }
  if (status === 'failed') {
    return 'Lỗi';
  }
  return 'Đang xử lý';
}

function getStatusTone(status: PracticeJobStatus) {
  if (status === 'completed') {
    return 'success';
  }
  if (status === 'failed') {
    return 'error';
  }
  return 'processing';
}

function formatDate(value: string | null) {
  if (!value) {
    return '';
  }

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
  scoreRing: {
    width: 78,
    height: 78,
    borderRadius: 39,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  scoreRingSegment: {
    position: 'absolute',
    width: 4,
    height: 10,
    borderRadius: 999,
    top: 34,
    left: 37,
  },
  scoreRingCenter: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: colors.surface,
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
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  scoreBadge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  scoreBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
  },
  scorerBadge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  scorerBadgeText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '800',
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
  feedbackText: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '700',
    lineHeight: 20,
  },
});
