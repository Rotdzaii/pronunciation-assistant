import { useEffect, useState } from 'react';
import { FlatList, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
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
import { fetchPracticeHistory, reportStudentPracticeResult } from '../../lib/api';
import { formatReviewReason, formatReviewStatus } from '../../lib/format';
import { useAuth } from '../../lib/auth';
import {
  clampScore,
  formatFeedbackLines,
  formatProblemPhonemes,
  getScoreTone,
} from '../../lib/practiceFormatters';
import type { PracticeHistoryItem, PracticeJobStatus, ReportPracticeResultPayload, StudentReviewRequest } from '../../types';

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
  const [reportOpen, setReportOpen] = useState(false);
  const [reportReason, setReportReason] = useState<ReportPracticeResultPayload['reason']>('ai_scored_wrong');
  const [reportNote, setReportNote] = useState('');
  const [reviewRequest, setReviewRequest] = useState<StudentReviewRequest | null>(null);
  const [reporting, setReporting] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const scoreTone = getScoreTone(item.score);
  const problemLines = formatProblemPhonemes(item.problem_phonemes);
  const feedbackLines = formatFeedbackLines(item.feedback);

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

      <View style={styles.reportBox}>
        {reviewRequest ? (
          <View style={styles.reportSubmitted}>
            <Text style={styles.reportTitle}>Đã gửi yêu cầu xem lại</Text>
            <Text style={styles.feedbackText}>{formatReviewStatus(reviewRequest.status)} - {formatReviewReason(reviewRequest.reason)}</Text>
          </View>
        ) : (
          <Pressable accessibilityRole="button" onPress={() => setReportOpen((value) => !value)} style={styles.reportButton}>
            <Text style={styles.reportButtonText}>Báo cáo kết quả</Text>
          </Pressable>
        )}

        {reportOpen && !reviewRequest ? (
          <View style={styles.reportForm}>
            <Text style={styles.reportTitle}>Yêu cầu giảng viên xem lại</Text>
            <View style={styles.reasonGrid}>
              {reportReasons.map((reason) => (
                <Pressable
                  key={reason}
                  accessibilityRole="button"
                  onPress={() => setReportReason(reason)}
                  style={[styles.reasonButton, reportReason === reason ? styles.reasonButtonActive : null]}
                >
                  <Text style={[styles.reasonButtonText, reportReason === reason ? styles.reasonButtonTextActive : null]}>
                    {formatReviewReason(reason)}
                  </Text>
                </Pressable>
              ))}
            </View>
            <TextInput
              value={reportNote}
              onChangeText={setReportNote}
              placeholder="Ghi chú ngắn cho giảng viên"
              placeholderTextColor="#94A3B8"
              style={styles.reportInput}
            />
            {reportError ? <Text style={styles.reportError}>{reportError}</Text> : null}
            <Pressable
              accessibilityRole="button"
              disabled={reporting}
              onPress={async () => {
                setReporting(true);
                setReportError(null);
                try {
                  const response = await reportStudentPracticeResult(item.id, {
                    reason: reportReason,
                    student_note: reportNote.trim() || null,
                  });
                  setReviewRequest(response);
                  setReportOpen(false);
                } catch (err) {
                  setReportError(err instanceof Error ? err.message : 'Chưa thể gửi yêu cầu xem lại.');
                } finally {
                  setReporting(false);
                }
              }}
              style={[styles.submitReportButton, reporting ? styles.disabledButton : null]}
            >
              <Text style={styles.submitReportText}>{reporting ? 'Đang gửi...' : 'Gửi yêu cầu xem lại'}</Text>
            </Pressable>
          </View>
        ) : null}
      </View>
    </AppCard>
  );
}

const reportReasons: ReportPracticeResultPayload['reason'][] = [
  'ai_scored_wrong',
  'audio_issue',
  'result_mismatch',
  'other',
];

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
  reportBox: {
    gap: 10,
  },
  reportButton: {
    alignSelf: 'flex-start',
    minHeight: 40,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  reportButtonText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  reportSubmitted: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: colors.softBlue,
    padding: 12,
    gap: 4,
  },
  reportForm: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 12,
    gap: 10,
  },
  reportTitle: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  reasonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  reasonButton: {
    minHeight: 34,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  reasonButtonActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  reasonButtonText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
  },
  reasonButtonTextActive: {
    color: '#FFFFFF',
  },
  reportInput: {
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 14,
    paddingHorizontal: 12,
  },
  reportError: {
    color: colors.error,
    fontSize: 13,
    fontWeight: '800',
  },
  submitReportButton: {
    minHeight: 42,
    borderRadius: 10,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  submitReportText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '900',
  },
  disabledButton: {
    opacity: 0.58,
  },
});
