import { StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { AppCard, AppScreen, SectionHeader, colors } from '../../components/AppUI';

const modes = [
  'Luyện từ đơn',
  'Luyện câu ngắn',
  'Lỗi người Việt hay gặp',
  'Luyện nhanh 3 phút',
  'Chủ đề phỏng vấn',
  'Chủ đề giao tiếp lớp học',
];

export default function PracticeModeScreen() {
  const router = useRouter();

  return (
    <AppScreen>
      <SectionHeader
        eyebrow="Luyện tập"
        title="Chọn chế độ luyện"
        subtitle="Các chế độ này là khung điều hướng sản phẩm, nội dung thật sẽ được nối sau."
      />

      <View style={styles.grid}>
        {modes.map((mode, index) => (
          <AppCard
            key={mode}
            style={styles.modeCard}
            tone={index % 3 === 0 ? 'blue' : index % 3 === 1 ? 'teal' : 'orange'}
          >
            <Text style={styles.modeTitle}>{mode}</Text>
            <Text style={styles.modeText}>Mở luồng luyện phù hợp khi dữ liệu bài học sẵn sàng.</Text>
            <Text
              accessibilityRole="link"
              onPress={() => router.push(index === 1 ? '/(tabs)/sentence' : '/(tabs)/practice')}
              style={styles.modeLink}
            >
              Chọn chế độ
            </Text>
          </AppCard>
        ))}
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
  modeCard: {
    flexGrow: 1,
    flexBasis: 260,
    minHeight: 148,
  },
  modeTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: '900',
  },
  modeText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 20,
  },
  modeLink: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
});
