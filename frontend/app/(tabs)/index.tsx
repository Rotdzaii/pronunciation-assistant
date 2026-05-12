import { StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { AppButton, AppCard, AppScreen, SectionHeader, colors } from '../../components/AppUI';

const paths = ['Âm cuối', 'Nguyên âm ngắn/dài', 'Trọng âm từ', 'Cụm phụ âm', 'Câu giao tiếp'];

export default function HomeScreen() {
  const router = useRouter();

  return (
    <AppScreen>
      <SectionHeader
        eyebrow="Trang chủ"
        title="Trang chủ"
        subtitle="Hôm nay luyện 5 phút nhé?"
      />

      <AppCard tone="blue">
        <Text style={styles.cardTitle}>Bài luyện tiếp theo</Text>
        <Text style={styles.cardText}>
          Chọn một mục nhỏ, ghi âm và để AI góp ý phát âm bằng tiếng Việt.
        </Text>
        <AppButton
          title="Bắt đầu luyện tập"
          onPress={() => router.push('/(tabs)/practice-mode')}
          style={styles.cta}
        />
      </AppCard>

      <View style={styles.grid}>
        {paths.map((item) => (
          <AppCard key={item} style={styles.pathCard}>
            <View style={styles.mark} />
            <Text style={styles.pathTitle}>{item}</Text>
            <Text style={styles.pathText}>Sẵn sàng để kết nối nội dung luyện tập.</Text>
          </AppCard>
        ))}
      </View>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  cardTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  cardText: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
  },
  cta: {
    alignSelf: 'flex-start',
    minWidth: 180,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  pathCard: {
    flexGrow: 1,
    flexBasis: 220,
    minHeight: 132,
  },
  mark: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.softBlue,
    borderWidth: 6,
    borderColor: colors.primary,
  },
  pathTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  pathText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
});
