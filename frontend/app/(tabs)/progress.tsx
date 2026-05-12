import { StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, EmptyState, SectionHeader, colors } from '../../components/AppUI';

export default function ProgressScreen() {
  return (
    <AppScreen>
      <SectionHeader
        eyebrow="Thống kê"
        title="Tiến độ"
        subtitle="Theo dõi điểm số, chuỗi ngày luyện và nhóm âm cần cải thiện."
      />
      <EmptyState
        title="Chưa có dữ liệu tiến độ."
        message="Hoàn thành vài bài luyện để xem thống kê."
      />
      <View style={styles.grid}>
        <MetricPlaceholder label="Điểm trung bình" />
        <MetricPlaceholder label="Chuỗi ngày luyện" />
        <MetricPlaceholder label="Âm cải thiện" />
      </View>
    </AppScreen>
  );
}

function MetricPlaceholder({ label }: { label: string }) {
  return (
    <AppCard style={styles.metricCard} tone="blue">
      <Text style={styles.metricValue}>--</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </AppCard>
  );
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flexGrow: 1,
    flexBasis: 180,
    minHeight: 112,
    justifyContent: 'center',
  },
  metricValue: {
    color: colors.text,
    fontSize: 30,
    fontWeight: '900',
  },
  metricLabel: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
});
