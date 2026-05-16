import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppScreen, colors } from '../../components/AppUI';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type MistakeCategory = {
  slug: string;
  title: string;
  englishTitle: string;
  description: string;
  icon: IconName;
  tone: 'blue' | 'teal' | 'orange' | 'lavender' | 'green';
  difficulty: string;
  warning?: string;
  examples: Array<{
    label: string;
    wrong?: string;
    correct?: string;
    note?: string;
  }>;
  wide?: boolean;
};

const categories: MistakeCategory[] = [
  {
    title: 'Âm cuối',
    slug: 'final-sounds',
    englishTitle: 'Ending Sounds',
    description:
      'Người Việt thường bỏ qua phụ âm cuối như /s/, /z/, /t/, /d/, /k/, /g/, khiến nghĩa của từ thay đổi hoặc câu nghe thiếu rõ ràng.',
    icon: 'account-voice',
    tone: 'blue',
    difficulty: 'Mức độ khó: Cao',
    warning: 'Lỗi thường gặp',
    examples: [
      { label: 'with', wrong: 'wi', correct: 'with', note: 'Giữ âm /ð/ và bật nhẹ âm cuối.' },
      { label: 'five', wrong: 'fi', correct: 'five', note: 'Không bỏ âm /v/ cuối từ.' },
    ],
  },
  {
    title: 'Trọng âm từ',
    slug: 'word-stress',
    englishTitle: 'Word Stress',
    description:
      'Nhấn sai trọng âm hoặc đọc đều từng âm tiết làm người nghe khó nhận ra từ, đặc biệt với từ dài hoặc từ học thuật.',
    icon: 'chart-timeline-variant',
    tone: 'teal',
    difficulty: 'Mức độ khó: Trung bình',
    examples: [
      { label: 'photograph', wrong: 'pho-to-GRAPH', correct: 'PHO-to-graph' },
      { label: 'important', wrong: 'IM-por-tant', correct: 'im-POR-tant' },
    ],
  },
  {
    title: 'Độ dài nguyên âm',
    slug: 'vowel-length',
    englishTitle: 'Vowel Length',
    description:
      'Cần phân biệt nguyên âm ngắn và dài, ví dụ /ɪ/ trong “ship” khác với /iː/ trong “sheep”.',
    icon: 'waves',
    tone: 'orange',
    difficulty: 'Mức độ khó: Trung bình',
    examples: [
      { label: 'minimal pair', wrong: 'ship /ʃɪp/', correct: 'sheep /ʃiːp/' },
      { label: 'minimal pair', wrong: 'bit /bɪt/', correct: 'beat /biːt/' },
    ],
  },
  {
    title: 'Cụm phụ âm',
    slug: 'consonant-clusters',
    englishTitle: 'Consonant Clusters',
    description:
      'Khi nhiều phụ âm đứng cạnh nhau, người học dễ chèn thêm nguyên âm hoặc lược bớt âm trong cụm.',
    icon: 'set-merge',
    tone: 'lavender',
    difficulty: 'Mức độ khó: Cao',
    warning: 'Rất phổ biến',
    examples: [
      { label: 'street', wrong: 'sờ-trít', correct: 'street /striːt/' },
      { label: 'crisps', wrong: 'crisp', correct: 'crisps /krɪsps/' },
    ],
  },
  {
    title: 'Nối âm',
    slug: 'linking-sounds',
    englishTitle: 'Linking Sounds',
    description:
      'Nối âm giúp câu nói trôi chảy hơn, nhất là khi một từ kết thúc bằng phụ âm và từ sau bắt đầu bằng nguyên âm.',
    icon: 'link-variant',
    tone: 'green',
    difficulty: 'Mức độ khó: Nâng cao',
    examples: [
      { label: 'linking', wrong: 'an apple', correct: 'an_apple' },
      { label: 'linking', wrong: 'turn off', correct: 'turn_off' },
    ],
    wide: true,
  },
];

export default function MistakesScreen() {
  const router = useRouter();

  return (
    <AppScreen maxWidth={1180}>
      <View style={styles.header}>
        <View style={styles.headerPill}>
          <MaterialCommunityIcons name="book-open-page-variant-outline" size={16} color={colors.primary} />
          <Text style={styles.headerPillText}>Tài liệu học tập</Text>
        </View>
        <Text style={styles.title}>Thư viện lỗi phổ biến</Text>
        <Text style={styles.subtitle}>
          Tập trung cải thiện những âm vị và quy tắc phát âm thường gây khó khăn nhất cho người nói tiếng Việt.
          Chinh phục từng lỗi để giọng nói tự nhiên và rõ ràng hơn.
        </Text>
      </View>

      <View style={styles.grid}>
        {categories.map((category) => (
          <MistakeCard
            key={category.title}
            category={category}
            onPractice={() => router.push(`/mistakes/${category.slug}/practice`)}
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
    <View style={[styles.card, category.wide ? styles.wideCard : null]}>
      <View style={[styles.iconBlock, toneStyles[category.tone]]}>
        <MaterialCommunityIcons name={category.icon} size={25} color={toneIconColor[category.tone]} />
      </View>

      <Text style={styles.cardTitle}>
        {category.title} ({category.englishTitle})
      </Text>
      <Text style={styles.cardDescription}>{category.description}</Text>

      <View style={styles.exampleBox}>
        {category.examples.map((example) => (
          <View key={`${category.title}-${example.label}-${example.correct}`} style={styles.exampleRow}>
            <Text style={styles.exampleLabel}>{example.label}</Text>
            <View style={styles.examplePair}>
              {example.wrong ? <Text style={styles.wrongText}>{example.wrong}</Text> : null}
              {example.correct ? <Text style={styles.correctText}>{example.correct}</Text> : null}
            </View>
            {example.note ? <Text style={styles.exampleNote}>{example.note}</Text> : null}
          </View>
        ))}
      </View>

      <View style={styles.badgeRow}>
        <View style={styles.difficultyBadge}>
          <Text style={styles.difficultyText}>{category.difficulty}</Text>
        </View>
        {category.warning ? (
          <View style={styles.warningBadge}>
            <Text style={styles.warningText}>{category.warning}</Text>
          </View>
        ) : null}
      </View>

      <Pressable accessibilityRole="button" onPress={onPractice} style={category.wide ? styles.primaryButton : styles.outlineButton}>
        <Text style={category.wide ? styles.primaryButtonText : styles.outlineButtonText}>
          Luyện tập ngay
        </Text>
        <MaterialCommunityIcons
          name="arrow-right"
          size={18}
          color={category.wide ? '#FFFFFF' : colors.primary}
        />
      </Pressable>
    </View>
  );
}

const toneIconColor = {
  blue: colors.primary,
  teal: '#006B5F',
  orange: '#FFFFFF',
  lavender: colors.primary,
  green: '#007A6C',
};

const toneStyles = StyleSheet.create({
  blue: {
    backgroundColor: colors.softBlue,
  },
  teal: {
    backgroundColor: colors.softTeal,
  },
  orange: {
    backgroundColor: colors.accent,
  },
  lavender: {
    backgroundColor: '#DBE1FF',
  },
  green: {
    backgroundColor: '#CCFBF1',
  },
});

const styles = StyleSheet.create({
  header: {
    gap: 12,
    marginBottom: 16,
  },
  headerPill: {
    alignSelf: 'flex-start',
    minHeight: 36,
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 13,
  },
  headerPillText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  title: {
    color: colors.text,
    fontSize: 36,
    lineHeight: 44,
    fontWeight: '900',
  },
  subtitle: {
    color: colors.muted,
    fontSize: 17,
    lineHeight: 27,
    maxWidth: 820,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 22,
  },
  card: {
    flexGrow: 1,
    flexBasis: 300,
    minHeight: 440,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 24,
    gap: 16,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.04,
    shadowRadius: 18,
    elevation: 3,
  },
  wideCard: {
    flexBasis: 620,
    minHeight: 300,
  },
  iconBlock: {
    width: 58,
    height: 58,
    borderRadius: 29,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTitle: {
    color: colors.text,
    fontSize: 24,
    lineHeight: 31,
    fontWeight: '900',
  },
  cardDescription: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 23,
  },
  exampleBox: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
    padding: 12,
    gap: 10,
  },
  exampleRow: {
    gap: 5,
  },
  exampleLabel: {
    color: colors.muted,
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  examplePair: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  wrongText: {
    color: colors.error,
    backgroundColor: colors.softRed,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    fontSize: 13,
    fontWeight: '900',
    textDecorationLine: 'line-through',
  },
  correctText: {
    color: colors.primary,
    backgroundColor: colors.softBlue,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    fontSize: 13,
    fontWeight: '900',
  },
  exampleNote: {
    color: colors.muted,
    fontSize: 12,
    lineHeight: 18,
  },
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 'auto',
  },
  difficultyBadge: {
    borderRadius: 999,
    backgroundColor: '#DBE1FF',
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  difficultyText: {
    color: colors.text,
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  warningBadge: {
    borderRadius: 999,
    backgroundColor: colors.softRed,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  warningText: {
    color: colors.error,
    fontSize: 11,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  outlineButton: {
    minHeight: 54,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.primary,
    backgroundColor: colors.surface,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  outlineButtonText: {
    color: colors.primary,
    fontSize: 15,
    fontWeight: '900',
  },
  primaryButton: {
    alignSelf: 'flex-start',
    minHeight: 54,
    borderRadius: 10,
    backgroundColor: colors.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 22,
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 14,
    elevation: 4,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
});
