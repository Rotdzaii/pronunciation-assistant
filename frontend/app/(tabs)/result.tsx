import { StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, EmptyState, SectionHeader, colors } from '../../components/AppUI';
import { BackButton } from '../../components/BackButton';

export default function ResultScreen() {
  return (
    <AppScreen>
      <BackButton label="Quay lại luyện tập" fallbackHref="/(tabs)/practice" />
      <SectionHeader
        eyebrow="Kết quả"
        title="Kết quả phát âm"
        subtitle="Kết quả thật sẽ hiển thị sau khi một lượt chấm hoàn tất."
      />
      <EmptyState title="Chưa có kết quả để hiển thị." />
      <View style={styles.grid}>
        <AppCard style={styles.placeholder}>
          <View style={styles.scoreRing}>
            <Text style={styles.scoreText}>--</Text>
          </View>
          <Text style={styles.placeholderTitle}>Vòng điểm</Text>
        </AppCard>
        <AppCard style={styles.placeholder}>
          <Text style={styles.placeholderTitle}>Âm cần luyện</Text>
          <Text style={styles.placeholderText}>Các âm nổi bật sẽ xuất hiện tại đây.</Text>
        </AppCard>
        <AppCard style={styles.placeholder}>
          <Text style={styles.placeholderTitle}>Góp ý tiếng Việt</Text>
          <Text style={styles.placeholderText}>Phần giải thích và gợi ý sửa lỗi sẽ được nối với kết quả AI.</Text>
        </AppCard>
      </View>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  placeholder: {
    flexGrow: 1,
    flexBasis: 220,
  },
  scoreRing: {
    width: 104,
    height: 104,
    borderRadius: 52,
    borderWidth: 10,
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreText: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
  },
  placeholderTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  placeholderText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20,
  },
});
