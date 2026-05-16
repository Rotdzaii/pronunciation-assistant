import { useMemo, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, colors } from '../../components/AppUI';
import { vocabularyQuestions } from '../../lib/vocabularyQuiz';

export default function QuizScreen() {
  const router = useRouter();
  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [correctCount, setCorrectCount] = useState(0);
  const question = vocabularyQuestions[index];
  const isAnswered = selected !== null;
  const isCorrect = selected === question.answer;
  const progress = useMemo(
    () => `${index + 1}/${vocabularyQuestions.length}`,
    [index],
  );

  const handleAnswer = (choice: string) => {
    if (isAnswered) {
      return;
    }
    setSelected(choice);
    if (choice === question.answer) {
      setCorrectCount((value) => value + 1);
    }
  };

  const handleNext = () => {
    if (index + 1 >= vocabularyQuestions.length) {
      const finalCorrect = correctCount + (isCorrect ? 0 : 0);
      router.replace({
        pathname: '/(tabs)/quiz-results',
        params: {
          correct: String(finalCorrect),
          total: String(vocabularyQuestions.length),
        },
      });
      return;
    }

    setIndex((value) => value + 1);
    setSelected(null);
  };

  return (
    <AppScreen maxWidth={760}>
      <AppCard style={styles.quizCard}>
        <View style={styles.topRow}>
          <Text style={styles.progressText}>Câu {progress}</Text>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${((index + 1) / vocabularyQuestions.length) * 100}%` }]} />
          </View>
        </View>

        <View style={styles.wordPill}>
          <MaterialCommunityIcons name="book-open-page-variant-outline" size={18} color={colors.primary} />
          <Text style={styles.wordPillText}>{question.word}</Text>
        </View>

        <Text style={styles.prompt}>{question.prompt}</Text>

        <View style={styles.choiceList}>
          {question.choices.map((choice) => {
            const active = selected === choice;
            const correct = isAnswered && choice === question.answer;
            const wrong = active && !correct;
            return (
              <Pressable
                key={choice}
                disabled={isAnswered}
                onPress={() => handleAnswer(choice)}
                style={[
                  styles.choice,
                  correct ? styles.choiceCorrect : null,
                  wrong ? styles.choiceWrong : null,
                ]}
              >
                <Text
                  style={[
                    styles.choiceText,
                    correct || wrong ? styles.choiceTextActive : null,
                  ]}
                >
                  {choice}
                </Text>
              </Pressable>
            );
          })}
        </View>

        {isAnswered ? (
          <View style={[styles.feedbackBox, isCorrect ? styles.feedbackCorrect : styles.feedbackWrong]}>
            <Text style={styles.feedbackTitle}>{isCorrect ? 'Chính xác!' : 'Chưa đúng'}</Text>
            <Text style={styles.feedbackText}>{question.explanation}</Text>
          </View>
        ) : null}

        <Pressable
          disabled={!isAnswered}
          onPress={handleNext}
          style={[styles.nextButton, !isAnswered ? styles.nextButtonDisabled : null]}
        >
          <Text style={styles.nextButtonText}>
            {index + 1 >= vocabularyQuestions.length ? 'Xem kết quả' : 'Câu tiếp theo'}
          </Text>
        </Pressable>
      </AppCard>
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  quizCard: {
    borderRadius: 24,
    gap: 18,
  },
  topRow: {
    gap: 8,
  },
  progressText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '900',
  },
  progressTrack: {
    height: 10,
    borderRadius: 999,
    backgroundColor: colors.border,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 999,
    backgroundColor: colors.primary,
  },
  wordPill: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  wordPillText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  prompt: {
    color: colors.text,
    fontSize: 26,
    lineHeight: 34,
    fontWeight: '900',
  },
  choiceList: {
    gap: 10,
  },
  choice: {
    minHeight: 58,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  choiceCorrect: {
    borderColor: colors.success,
    backgroundColor: '#DCFCE7',
  },
  choiceWrong: {
    borderColor: colors.error,
    backgroundColor: colors.softRed,
  },
  choiceText: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '800',
  },
  choiceTextActive: {
    color: colors.text,
  },
  feedbackBox: {
    borderRadius: 16,
    padding: 14,
    gap: 4,
  },
  feedbackCorrect: {
    backgroundColor: '#DCFCE7',
  },
  feedbackWrong: {
    backgroundColor: colors.softRed,
  },
  feedbackTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  feedbackText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  nextButton: {
    minHeight: 54,
    borderRadius: 999,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  nextButtonDisabled: {
    opacity: 0.45,
  },
  nextButtonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
});
