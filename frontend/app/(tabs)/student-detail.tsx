import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, colors } from '../../components/AppUI';
import { BackButton } from '../../components/BackButton';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type StudentRouteParams = {
  name?: string;
  className?: string;
  level?: string;
  student_id?: string;
};

export default function StudentDetailScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<StudentRouteParams>();
  const studentName = normalizeParam(params.name);
  const className = normalizeParam(params.className);
  const level = normalizeParam(params.level);
  const studentId = normalizeParam(params.student_id);
  const hasStudent = Boolean(studentName);

  return (
    <AppScreen maxWidth={1200}>
      <View style={styles.topHeader}>
        <Text style={styles.appName}>Trợ lý Phát âm</Text>
        <View style={styles.accountButton}>
          <MaterialCommunityIcons name="account-circle-outline" size={27} color="#2E3039" />
        </View>
      </View>

      <BackButton
        label="Quay lại danh sách"
        fallbackHref="/(tabs)/students"
        style={styles.backLink}
      />

      {!hasStudent ? (
        <AppCard style={styles.emptyStateCard}>
          <View style={styles.emptyIcon}>
            <MaterialCommunityIcons name="account-search-outline" size={34} color={colors.primary} />
          </View>
          <Text style={styles.emptyTitle}>Chưa chọn học viên.</Text>
          <Text style={styles.emptyMessage}>
            Vui lòng chọn một học viên từ danh sách để xem chi tiết.
          </Text>
        </AppCard>
      ) : null}

      <View style={styles.dashboardGrid}>
        <View style={styles.leftColumn}>
          <ProfileCard
            name={studentName}
            className={className}
            level={level}
            hasStudent={hasStudent}
          />
          <CommonMistakesCard hasStudent={hasStudent} />
        </View>

        <View style={styles.rightColumn}>
          <AIInterventionCard hasStudent={hasStudent} studentId={studentId} onAssign={() => router.push('/(tabs)/teacher')} />
          <ScoreTrendCard hasStudent={hasStudent} />
          <RecentHistoryCard hasStudent={hasStudent} />
        </View>
      </View>
    </AppScreen>
  );
}

function ProfileCard({
  name,
  className,
  level,
  hasStudent,
}: {
  name: string | null;
  className: string | null;
  level: string | null;
  hasStudent: boolean;
}) {
  return (
    <AppCard style={styles.profileCard}>
      <View style={styles.avatar}>
        <MaterialCommunityIcons
          name={hasStudent ? 'account-outline' : 'account-question-outline'}
          size={46}
          color={colors.primary}
        />
        <View style={styles.avatarStatus}>
          <MaterialCommunityIcons name="microphone-outline" size={13} color="#FFFFFF" />
        </View>
      </View>

      <Text style={styles.profileName}>{name ?? 'Chưa chọn học viên'}</Text>
      <Text style={styles.profileMeta}>
        {hasStudent
          ? [level, className].filter(Boolean).join(' • ') || 'Thông tin lớp chưa có'
          : 'Chưa có thông tin hồ sơ'}
      </Text>

      <View style={styles.profileStats}>
        <ProfileStat value="--" label="Độ chính xác tổng thể" color={colors.primary} />
        <View style={styles.statDivider} />
        <ProfileStat value="--" label="Ngày liên tiếp" color="#943700" icon="fire" />
      </View>
    </AppCard>
  );
}

function ProfileStat({
  value,
  label,
  color,
  icon,
}: {
  value: string;
  label: string;
  color: string;
  icon?: IconName;
}) {
  return (
    <View style={styles.profileStat}>
      <View style={styles.statValueRow}>
        <Text style={[styles.statValue, { color }]}>{value}</Text>
        {icon ? <MaterialCommunityIcons name={icon} size={20} color={color} /> : null}
      </View>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function CommonMistakesCard({ hasStudent }: { hasStudent: boolean }) {
  return (
    <AppCard style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>Lỗi thường gặp</Text>
        <MaterialCommunityIcons name="alert-circle-outline" size={24} color="#737686" />
      </View>
      <Text style={styles.cardText}>
        {hasStudent
          ? 'Chưa có dữ liệu lỗi phát âm gần đây cho học viên này.'
          : 'Chưa có lỗi nổi bật cho học viên được chọn.'}
      </Text>
      <View style={styles.chipRow}>
        <EmptyChip tone="red" />
        <EmptyChip tone="orange" />
        <EmptyChip tone="lavender" />
      </View>
    </AppCard>
  );
}

function AIInterventionCard({
  hasStudent,
  studentId,
  onAssign,
}: {
  hasStudent: boolean;
  studentId: string | null;
  onAssign: () => void;
}) {
  const canAssign = hasStudent && Boolean(studentId);
  return (
    <AppCard style={styles.aiCard}>
      <View style={styles.aiContent}>
        <View style={styles.aiIcon}>
          <MaterialCommunityIcons name="head-cog-outline" size={24} color="#006F64" />
        </View>
        <View style={styles.aiTextColumn}>
          <View style={styles.aiTitleRow}>
            <Text style={styles.aiTitle}>Chiến lược can thiệp AI</Text>
            <View style={styles.recommendBadge}>
              <Text style={styles.recommendText}>Đề xuất</Text>
            </View>
          </View>
          <Text style={styles.aiDescription}>
            {hasStudent
              ? 'Chưa có đề xuất can thiệp từ dữ liệu luyện tập thật của học viên này.'
              : 'AI sẽ đề xuất bài tập mục tiêu khi có hồ sơ học viên và dữ liệu luyện tập.'}
          </Text>
          <Pressable
            accessibilityRole="button"
            disabled={!canAssign}
            onPress={canAssign ? onAssign : undefined}
            style={[styles.aiButton, !canAssign ? styles.aiButtonDisabled : null]}
          >
            <MaterialCommunityIcons name="clipboard-plus-outline" size={18} color="#FFFFFF" />
            <Text style={styles.aiButtonText}>Giao bài tập mục tiêu</Text>
          </Pressable>
        </View>
      </View>
    </AppCard>
  );
}

function ScoreTrendCard({ hasStudent }: { hasStudent: boolean }) {
  return (
    <AppCard style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>Xu hướng điểm số (7 phiên gần nhất)</Text>
        <View style={styles.legend}>
          <View style={styles.legendDot} />
          <Text style={styles.legendText}>Độ chính xác</Text>
        </View>
      </View>
      <View style={styles.chart}>
        <View style={styles.gridLineTop} />
        <View style={styles.gridLineMiddle} />
        <View style={styles.gridLineBottom} />
        {['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'].map((label) => (
          <View key={label} style={styles.barColumn}>
            <View style={[styles.emptyBar, hasStudent ? styles.awaitingBar : null]} />
            <Text style={styles.dayLabel}>{label}</Text>
          </View>
        ))}
      </View>
      <Text style={styles.chartEmptyText}>
        {hasStudent
          ? 'Chưa có đủ dữ liệu điểm số để vẽ xu hướng.'
          : 'Chọn học viên để xem xu hướng điểm số.'}
      </Text>
    </AppCard>
  );
}

function RecentHistoryCard({ hasStudent }: { hasStudent: boolean }) {
  return (
    <AppCard style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.cardTitle}>Lịch sử luyện tập gần đây</Text>
        <Text style={styles.seeAll}>Xem tất cả</Text>
      </View>
      <View style={styles.historyList}>
        {[0, 1, 2].map((item) => (
          <View key={item} style={styles.historySkeleton}>
            <View style={styles.historyAvatar} />
            <View style={styles.historyLines}>
              <View style={styles.historyLineWide} />
              <View style={styles.historyLineShort} />
            </View>
            <View style={styles.historyPill} />
          </View>
        ))}
      </View>
      <Text style={styles.syncText}>
        {hasStudent
          ? 'Đang đồng bộ dữ liệu mới nhất...'
          : 'Chưa có lịch sử luyện tập để hiển thị.'}
      </Text>
    </AppCard>
  );
}

function EmptyChip({ tone }: { tone: 'red' | 'orange' | 'lavender' }) {
  return <View style={[styles.emptyChip, chipStyles[tone]]} />;
}

function normalizeParam(value: string | string[] | undefined) {
  if (Array.isArray(value)) {
    return value[0] || null;
  }

  return value || null;
}

const chipStyles = StyleSheet.create({
  red: {
    backgroundColor: '#FFDAD6',
  },
  orange: {
    backgroundColor: '#FFDBCD',
  },
  lavender: {
    backgroundColor: '#E1E2ED',
  },
});

const styles = StyleSheet.create({
  topHeader: {
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
  appName: {
    color: '#004AC6',
    fontSize: 24,
    fontWeight: '900',
  },
  accountButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  backLink: {
    alignSelf: 'flex-start',
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 18,
    marginTop: 14,
    marginBottom: 10,
  },
  emptyStateCard: {
    borderRadius: 12,
    alignItems: 'center',
    paddingVertical: 26,
    backgroundColor: '#FFFFFF',
  },
  emptyIcon: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: '#DBE1FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
    textAlign: 'center',
  },
  emptyMessage: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
  },
  dashboardGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 24,
  },
  leftColumn: {
    flexGrow: 1,
    flexBasis: 330,
    maxWidth: 420,
    gap: 24,
  },
  rightColumn: {
    flexGrow: 2,
    flexBasis: 540,
    gap: 24,
  },
  profileCard: {
    borderRadius: 12,
    alignItems: 'center',
    padding: 24,
  },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    borderWidth: 4,
    borderColor: '#FFFFFF',
    backgroundColor: '#DBE1FF',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 3,
  },
  avatarStatus: {
    position: 'absolute',
    right: 2,
    bottom: 6,
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FFFFFF',
    backgroundColor: colors.secondary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  profileName: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '900',
    marginTop: 18,
    textAlign: 'center',
  },
  profileMeta: {
    color: '#434655',
    fontSize: 14,
    lineHeight: 20,
    marginTop: 6,
    textAlign: 'center',
  },
  profileStats: {
    width: '100%',
    borderTopWidth: 1,
    borderTopColor: '#E1E2ED',
    flexDirection: 'row',
    alignItems: 'stretch',
    justifyContent: 'center',
    marginTop: 24,
    paddingTop: 18,
  },
  profileStat: {
    flex: 1,
    alignItems: 'center',
    gap: 4,
  },
  statDivider: {
    width: 1,
    backgroundColor: '#E1E2ED',
    marginHorizontal: 12,
  },
  statValueRow: {
    minHeight: 32,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  statValue: {
    fontSize: 25,
    fontWeight: '900',
  },
  statLabel: {
    color: '#434655',
    fontSize: 12,
    lineHeight: 17,
    textAlign: 'center',
  },
  card: {
    borderRadius: 12,
    padding: 24,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  cardTitle: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '900',
    flexShrink: 1,
  },
  cardText: {
    color: '#434655',
    fontSize: 14,
    lineHeight: 21,
  },
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  emptyChip: {
    width: 96,
    height: 34,
    borderRadius: 17,
  },
  aiCard: {
    borderRadius: 12,
    borderColor: 'rgba(0, 107, 95, 0.22)',
    backgroundColor: '#E0F7F6',
    padding: 24,
  },
  aiContent: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 18,
  },
  aiIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#71F8E4',
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiTextColumn: {
    flex: 1,
    gap: 10,
  },
  aiTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
  },
  aiTitle: {
    color: '#005048',
    fontSize: 22,
    fontWeight: '900',
  },
  recommendBadge: {
    borderRadius: 999,
    backgroundColor: '#006B5F',
    paddingHorizontal: 9,
    paddingVertical: 4,
  },
  recommendText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  aiDescription: {
    color: '#434655',
    fontSize: 16,
    lineHeight: 25,
  },
  aiButton: {
    alignSelf: 'flex-start',
    minHeight: 48,
    borderRadius: 8,
    backgroundColor: '#007A6C',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 20,
    opacity: 0.62,
  },
  aiButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  legend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#004AC6',
  },
  legendText: {
    color: '#434655',
    fontSize: 13,
  },
  chart: {
    height: 230,
    borderBottomWidth: 1,
    borderBottomColor: '#E1E2ED',
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 8,
    paddingTop: 18,
    paddingBottom: 34,
  },
  gridLineTop: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 42,
    borderTopWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#E1E2ED',
  },
  gridLineMiddle: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 100,
    borderTopWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#E1E2ED',
  },
  gridLineBottom: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 158,
    borderTopWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#E1E2ED',
  },
  barColumn: {
    flex: 1,
    height: '100%',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 12,
  },
  emptyBar: {
    width: '100%',
    maxWidth: 95,
    height: 118,
    borderTopLeftRadius: 6,
    borderTopRightRadius: 6,
    backgroundColor: '#DBE1FF',
    opacity: 0.72,
  },
  awaitingBar: {
    backgroundColor: '#B4C5FF',
  },
  dayLabel: {
    color: '#434655',
    fontSize: 12,
  },
  chartEmptyText: {
    color: colors.muted,
    fontSize: 13,
    textAlign: 'center',
  },
  seeAll: {
    color: '#004AC6',
    fontSize: 14,
    fontWeight: '800',
  },
  historyList: {
    gap: 14,
  },
  historySkeleton: {
    minHeight: 78,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#F1F5F9',
    backgroundColor: '#FAFAFD',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 18,
    paddingHorizontal: 18,
  },
  historyAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#F0F0FB',
  },
  historyLines: {
    flex: 1,
    gap: 10,
  },
  historyLineWide: {
    width: '42%',
    height: 16,
    borderRadius: 4,
    backgroundColor: '#F0F0FB',
  },
  historyLineShort: {
    width: '28%',
    height: 12,
    borderRadius: 4,
    backgroundColor: '#F3F3FE',
  },
  historyPill: {
    width: 56,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#F0F0FB',
  },
  syncText: {
    color: colors.muted,
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
  },
  aiButtonDisabled: {
    opacity: 0.5,
  },
});
