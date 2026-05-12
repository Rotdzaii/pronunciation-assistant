import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, colors } from '../../components/AppUI';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type ExampleRow = {
  left?: string;
  right?: string;
  center?: string;
  wrong?: boolean;
};

type MistakeCategory = {
  title: string;
  description: string;
  icon: IconName;
  iconTone: 'red' | 'orange' | 'blue' | 'teal' | 'lavender';
  examples: ExampleRow[];
};

const categories: MistakeCategory[] = [
  {
    title: 'Âm cuối',
    description:
      'Người Việt thường quên phát âm các phụ âm cuối từ, làm sai lệch nghĩa.',
    icon: 'volume-off',
    iconTone: 'red',
    examples: [
      { left: 'Wi', right: 'With', wrong: true },
      { left: 'Fi', right: 'Five', wrong: true },
    ],
  },
  {
    title: 'Độ dài nguyên âm',
    description: 'Không phân biệt rõ nguyên âm ngắn và dài.',
    icon: 'swap-horizontal',
    iconTone: 'orange',
    examples: [
      { left: 'Ngắn:', right: 'Ship' },
      { left: 'Dài:', right: 'Sheep' },
    ],
  },
  {
    title: 'Trọng âm từ',
    description:
      'Đọc đều các âm tiết như tiếng Việt thay vì nhấn trọng âm.',
    icon: 'format-letter-case',
    iconTone: 'blue',
    examples: [{ center: 'pho-to-graph' }],
  },
  {
    title: 'Cụm phụ âm',
    description:
      'Khó phát âm liên tiếp nhiều phụ âm mà không chèn nguyên âm vào giữa.',
    icon: 'apps',
    iconTone: 'teal',
    examples: [{ left: 'Sờ-tờ-rít', right: 'Street', wrong: true }],
  },
  {
    title: 'Nối âm',
    description:
      'Thiếu sự liên kết giữa các từ khi nói nhanh, làm câu nghe mất tự nhiên.',
    icon: 'link-variant',
    iconTone: 'lavender',
    examples: [{ center: 'and apple -> an_apple' }],
  },
];

export default function MistakesScreen() {
  const router = useRouter();

  return (
    <AppScreen maxWidth={600}>
      <View style={styles.header}>
        <Text style={styles.title}>Thư viện lỗi phổ biến</Text>
        <Text style={styles.subtitle}>
          Khắc phục các vấn đề phát âm tiếng Anh thường gặp của người Việt.
        </Text>
      </View>

      <View style={styles.list}>
        {categories.map((category) => (
          <MistakeCard
            key={category.title}
            category={category}
            onPractice={() => router.push('/(tabs)/practice-mode')}
          />
        ))}
      </View>
    </AppScreen>
  );
}

function MistakeCard({
  category,
  onPractice,
}: {
  category: MistakeCategory;
  onPractice: () => void;
}) {
  return (
    <AppCard style={styles.card}>
      <View style={styles.cardIntro}>
        <View style={[styles.iconBlock, iconToneStyles[category.iconTone]]}>
          <MaterialCommunityIcons
            name={category.icon}
            size={24}
            color={iconColor[category.iconTone]}
          />
        </View>
        <View style={styles.cardCopy}>
          <Text style={styles.cardTitle}>{category.title}</Text>
          <Text style={styles.cardDescription}>{category.description}</Text>
        </View>
      </View>

      <View style={styles.exampleBox}>
        {category.examples.map((example, index) => (
          <ExampleLine key={`${category.title}-${index}`} example={example} />
        ))}
      </View>

      <Pressable accessibilityRole="button" onPress={onPractice} style={styles.practiceButton}>
        <MaterialCommunityIcons name="microphone-outline" size={22} color="#FFFFFF" />
        <Text style={styles.practiceButtonText}>Luyện tập ngay</Text>
      </Pressable>
    </AppCard>
  );
}

function ExampleLine({ example }: { example: ExampleRow }) {
  if (example.center) {
    return (
      <View style={styles.centerExample}>
        <Text style={styles.exampleWord}>{example.center}</Text>
      </View>
    );
  }

  return (
    <View style={styles.exampleRow}>
      <Text style={[styles.exampleLeft, example.wrong ? styles.wrongExample : null]}>
        {example.left}
      </Text>
      <Text style={styles.exampleRight}>{example.right}</Text>
    </View>
  );
}

const iconColor = {
  red: '#BA1A1A',
  orange: '#FFFFFF',
  blue: '#FFFFFF',
  teal: '#006F64',
  lavender: '#003EA8',
};

const iconToneStyles = StyleSheet.create({
  red: {
    backgroundColor: '#FFDAD6',
  },
  orange: {
    backgroundColor: '#BC4800',
  },
  blue: {
    backgroundColor: colors.primary,
  },
  teal: {
    backgroundColor: '#71F8E4',
  },
  lavender: {
    backgroundColor: '#B4C5FF',
  },
});

const styles = StyleSheet.create({
  header: {
    gap: 6,
    marginBottom: 18,
  },
  title: {
    color: '#191B23',
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 40,
  },
  subtitle: {
    color: '#434655',
    fontSize: 16,
    lineHeight: 24,
  },
  list: {
    gap: 16,
  },
  card: {
    borderRadius: 12,
    padding: 16,
    gap: 16,
    backgroundColor: '#FFFFFF',
    borderColor: 'rgba(195, 198, 215, 0.45)',
  },
  cardIntro: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 16,
  },
  iconBlock: {
    width: 56,
    height: 56,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardCopy: {
    flex: 1,
    gap: 4,
  },
  cardTitle: {
    color: '#191B23',
    fontSize: 22,
    fontWeight: '900',
    lineHeight: 29,
  },
  cardDescription: {
    color: '#434655',
    fontSize: 14,
    lineHeight: 21,
  },
  exampleBox: {
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(195, 198, 215, 0.42)',
    backgroundColor: '#F3F3FE',
    paddingHorizontal: 10,
    paddingVertical: 8,
    gap: 8,
  },
  exampleRow: {
    minHeight: 24,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  centerExample: {
    minHeight: 42,
    alignItems: 'center',
    justifyContent: 'center',
  },
  exampleLeft: {
    color: '#434655',
    fontSize: 15,
    lineHeight: 22,
  },
  wrongExample: {
    color: '#BA1A1A',
    textDecorationLine: 'line-through',
  },
  exampleRight: {
    color: '#004AC6',
    fontSize: 15,
    fontWeight: '700',
    lineHeight: 22,
  },
  exampleWord: {
    color: '#004AC6',
    fontSize: 17,
    fontWeight: '800',
    lineHeight: 24,
    textAlign: 'center',
  },
  practiceButton: {
    minHeight: 56,
    borderRadius: 8,
    backgroundColor: '#004AC6',
    shadowColor: '#004AC6',
    shadowOffset: { width: 0, height: 7 },
    shadowOpacity: 0.22,
    shadowRadius: 12,
    elevation: 4,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  practiceButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
});
