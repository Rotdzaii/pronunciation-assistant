import { useEffect, useMemo, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import {
  AppCard,
  AppScreen,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import { fetchTeacherAnalytics } from '../../lib/api';
import { useAuth } from '../../lib/auth';
import type { TeacherAnalyticsResponse } from '../../types';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

export default function TeacherScreen() {
  const { accessToken } = useAuth();
  const [data, setData] = useState<TeacherAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetchTeacherAnalytics(accessToken);
        setData(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không tải được dữ liệu.');
      } finally {
        setLoading(false);
      }
    };

    void loadData();
  }, [accessToken]);

  const hasData = Boolean(data);
  const averageScore = data ? Math.round(data.avg_score) : null;
  const classStatus = useMemo(() => {
    if (!data) {
      return null;
    }

    return data.active_jobs > 0 ? 'Đang xử lý bài nộp' : 'Dữ liệu đã cập nhật';
  }, [data]);

  return (
    <AppScreen>
      <View style={styles.headerBar}>
        <View style={styles.headerTitleWrap}>
          <Text style={styles.appTitle}>Bảng điều khiển giáo viên</Text>
        </View>
        <View style={styles.profileButton}>
          <MaterialCommunityIcons name="account-circle-outline" size={26} color={colors.primary} />
        </View>
      </View>

      <View style={styles.dashboardHeader}>
        <View style={styles.dashboardCopy}>
          <Text style={styles.sectionTitle}>Tổng quan</Text>
          <Text style={styles.sectionSubtitle}>
            Theo dõi tình hình luyện phát âm của các lớp hôm nay.
          </Text>
        </View>

        <View style={styles.headerActions}>
          <ControlButton icon="calendar-month-outline" label="Tuần này" />
          <ControlButton icon="download-outline" label="Xuất báo cáo" primary disabled />
        </View>
      </View>

      {loading ? (
        <LoadingState
          title="Đang tải dữ liệu lớp học"
          message="Hệ thống đang tổng hợp các chỉ số luyện tập mới nhất."
        />
      ) : null}

      {error ? <ErrorState title="Không thể tải bảng điều khiển" message={error} /> : null}

      {!loading && !error && hasData && data ? (
        <>
          <View style={styles.metricsGrid}>
            <MetricCard
              icon="account-group-outline"
              label="Tổng học viên"
              value={String(data.total_students)}
              tone="blue"
            />
            <MetricCard
              icon="microphone-outline"
              label="Tác vụ đang xử lý"
              value={String(data.active_jobs)}
              tone="teal"
            />
            <MetricCard
              icon="chart-areaspline"
              label="Điểm trung bình"
              value={averageScore === null ? 'Chưa có dữ liệu' : `${averageScore}%`}
              tone="orange"
              badge={classStatus}
            />
          </View>

          <View style={styles.contentGrid}>
            <View style={styles.mainColumn}>
              <AIInsightCard />
              <StudentDirectoryCard />
            </View>

            <View style={styles.sideColumn}>
              <CommonErrorsCard />
              <ReportCard />
            </View>
          </View>
        </>
      ) : null}

      {!loading && !error && !hasData ? (
        <EmptyState
          title="Chưa có dữ liệu lớp học"
          message="Khi học viên gửi bài luyện tập, thống kê lớp học sẽ hiển thị tại đây."
        />
      ) : null}
    </AppScreen>
  );
}

function MetricCard({
  icon,
  label,
  value,
  tone,
  badge,
}: {
  icon: IconName;
  label: string;
  value: string;
  tone: 'blue' | 'teal' | 'orange';
  badge?: string | null;
}) {
  return (
    <AppCard style={styles.metricCard} tone={tone}>
      <View style={styles.metricTop}>
        <View style={styles.metricIcon}>
          <MaterialCommunityIcons name={icon} size={24} color={colors.primary} />
        </View>
        {badge ? <StatusBadge label={badge} tone="primary" style={styles.metricBadge} /> : null}
      </View>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </AppCard>
  );
}

function AIInsightCard() {
  return (
    <AppCard style={styles.aiCard}>
      <View style={styles.aiMark}>
        <MaterialCommunityIcons name="brain" size={22} color="#B4C5FF" />
        <Text style={styles.aiEyebrow}>Gợi ý từ AI</Text>
      </View>
      <Text style={styles.aiTitle}>Chưa có gợi ý can thiệp mới</Text>
      <Text style={styles.aiText}>
        Khi backend cung cấp phân tích theo lớp hoặc theo nhóm lỗi, gợi ý giao bài luyện sẽ xuất
        hiện tại đây.
      </Text>
      <Pressable accessibilityRole="button" disabled style={styles.aiButton}>
        <Text style={styles.aiButtonText}>Giao bài luyện</Text>
        <MaterialCommunityIcons name="arrow-right" size={18} color={colors.primary} />
      </Pressable>
    </AppCard>
  );
}

function CommonErrorsCard() {
  return (
    <AppCard style={styles.sideCard}>
      <View style={styles.cardHeaderRow}>
        <Text style={styles.cardTitle}>Lỗi phát âm phổ biến</Text>
        <MaterialCommunityIcons name="information-outline" size={20} color={colors.muted} />
      </View>
      <Text style={styles.cardDescription}>
        Chưa có dữ liệu lỗi phát âm theo lớp trong phản hồi hiện tại.
      </Text>
      <View style={styles.emptyPanel}>
        <MaterialCommunityIcons name="chart-bar" size={28} color={colors.primary} />
        <Text style={styles.emptyTitle}>Đang chờ dữ liệu lỗi</Text>
        <Text style={styles.emptyText}>
          Không hiển thị mẫu lỗi hoặc tỷ lệ khi API chưa trả về dữ liệu thật.
        </Text>
      </View>
    </AppCard>
  );
}

function StudentDirectoryCard() {
  return (
    <AppCard style={styles.directoryCard}>
      <View style={styles.directoryHeader}>
        <Text style={styles.cardTitle}>Danh sách học viên</Text>
        <View style={styles.directoryControls}>
          <View style={styles.searchShell}>
            <MaterialCommunityIcons name="magnify" size={18} color={colors.muted} />
            <Text style={styles.searchPlaceholder}>Tìm học viên...</Text>
          </View>
          <View style={styles.filterRow}>
            <FilterChip label="Cần hỗ trợ" />
            <FilterChip label="Tiến bộ tốt" />
          </View>
        </View>
      </View>

      <View style={styles.emptyList}>
        <MaterialCommunityIcons name="account-search-outline" size={34} color={colors.primary} />
        <Text style={styles.emptyTitle}>Chưa có danh sách học viên</Text>
        <Text style={styles.emptyText}>
          Khu vực này sẽ hiển thị học viên thật khi backend cung cấp dữ liệu danh sách lớp.
        </Text>
      </View>

      <View style={styles.directoryFooter}>
        <Text style={styles.footerAction}>Xem tất cả học viên</Text>
      </View>
    </AppCard>
  );
}

function ReportCard() {
  return (
    <AppCard style={styles.reportCard}>
      <Text style={styles.cardTitle}>Báo cáo</Text>
      <Text style={styles.cardDescription}>
        Báo cáo chi tiết sẽ dùng dữ liệu thật từ các lớp và bài luyện khi API sẵn sàng.
      </Text>
      <Pressable accessibilityRole="button" disabled style={styles.reportButton}>
        <Text style={styles.reportButtonText}>Xem báo cáo chi tiết</Text>
        <MaterialCommunityIcons name="chevron-right" size={18} color={colors.primary} />
      </Pressable>
    </AppCard>
  );
}

function ControlButton({
  icon,
  label,
  primary = false,
  disabled = false,
}: {
  icon: IconName;
  label: string;
  primary?: boolean;
  disabled?: boolean;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      style={[
        styles.controlButton,
        primary ? styles.primaryControlButton : styles.secondaryControlButton,
        disabled ? styles.disabledButton : null,
      ]}
    >
      <MaterialCommunityIcons name={icon} size={18} color={primary ? '#FFFFFF' : colors.text} />
      <Text style={[styles.controlButtonText, primary ? styles.primaryControlButtonText : null]}>
        {label}
      </Text>
    </Pressable>
  );
}

function FilterChip({ label }: { label: string }) {
  return (
    <View style={styles.filterChip}>
      <Text style={styles.filterChipText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  headerBar: {
    minHeight: 64,
    borderBottomWidth: 1,
    borderBottomColor: '#E1E2ED',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: -20,
    marginTop: -24,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  headerTitleWrap: {
    flex: 1,
  },
  appTitle: {
    color: colors.primary,
    fontSize: 20,
    fontWeight: '900',
  },
  profileButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F3F3FE',
  },
  dashboardHeader: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 16,
  },
  dashboardCopy: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 320,
    gap: 4,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 40,
  },
  sectionSubtitle: {
    color: colors.muted,
    fontSize: 17,
    lineHeight: 26,
  },
  headerActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  controlButton: {
    minHeight: 42,
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  secondaryControlButton: {
    backgroundColor: '#EDEDF9',
    borderColor: '#C3C6D7',
  },
  primaryControlButton: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  disabledButton: {
    opacity: 0.62,
  },
  controlButtonText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '800',
  },
  primaryControlButtonText: {
    color: '#FFFFFF',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flexGrow: 1,
    flexBasis: 220,
    minHeight: 156,
    borderRadius: 12,
    justifyContent: 'space-between',
  },
  metricTop: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 8,
  },
  metricIcon: {
    width: 42,
    height: 42,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#DBE1FF',
  },
  metricBadge: {
    maxWidth: 150,
  },
  metricLabel: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '800',
  },
  metricValue: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 39,
  },
  contentGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  mainColumn: {
    flexGrow: 2,
    flexBasis: 520,
    gap: 16,
  },
  sideColumn: {
    flexGrow: 1,
    flexBasis: 280,
    gap: 16,
  },
  aiCard: {
    borderRadius: 12,
    borderColor: colors.primary,
    backgroundColor: colors.primary,
    padding: 24,
  },
  aiMark: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  aiEyebrow: {
    color: '#B4C5FF',
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  aiTitle: {
    color: '#FFFFFF',
    fontSize: 21,
    fontWeight: '900',
    lineHeight: 28,
  },
  aiText: {
    color: '#EEF2FF',
    fontSize: 14,
    lineHeight: 21,
    maxWidth: 560,
  },
  aiButton: {
    alignSelf: 'flex-start',
    minHeight: 42,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    opacity: 0.78,
  },
  aiButtonText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
  sideCard: {
    borderRadius: 12,
    minHeight: 300,
  },
  reportCard: {
    borderRadius: 12,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  cardTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  cardDescription: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  emptyPanel: {
    flex: 1,
    minHeight: 170,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#E1E2ED',
    backgroundColor: '#F3F3FE',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 18,
    gap: 8,
  },
  emptyTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
    textAlign: 'center',
  },
  emptyText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
  },
  directoryCard: {
    borderRadius: 12,
    padding: 0,
    overflow: 'hidden',
  },
  directoryHeader: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E1E2ED',
    gap: 14,
  },
  directoryControls: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  searchShell: {
    flexGrow: 1,
    flexBasis: 220,
    minHeight: 42,
    borderRadius: 8,
    backgroundColor: '#F3F3FE',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    gap: 8,
  },
  searchPlaceholder: {
    color: colors.muted,
    fontSize: 14,
  },
  filterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  filterChip: {
    minHeight: 36,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#C3C6D7',
    backgroundColor: '#EDEDF9',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  filterChipText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  emptyList: {
    minHeight: 190,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    gap: 8,
  },
  directoryFooter: {
    backgroundColor: '#F3F3FE',
    borderTopWidth: 1,
    borderTopColor: '#E1E2ED',
    alignItems: 'center',
    paddingVertical: 14,
  },
  footerAction: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
  reportButton: {
    minHeight: 42,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    backgroundColor: '#F3F3FE',
    opacity: 0.72,
  },
  reportButtonText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
});
