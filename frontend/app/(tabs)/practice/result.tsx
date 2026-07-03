import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Audio } from 'expo-av';
import { MaterialIcons } from '@expo/vector-icons';
import { ScoreRing, TipCard, WaveformBars, WordToken } from '../../../components/practice';
import type { WordStatus } from '../../../components/practice';
import { colors, radius, spacing, typography } from '../../../constants/theme';
import {
  clampScore,
  formatFeedbackLines,
  formatProblemPhonemes,
} from '../../../lib/practiceFormatters';
import type { PracticeFeedback, ProblemPhoneme } from '../../../types';

const BREAKPOINT_TABLET = 768;

// Static waveform bar heights (decorative, matching result.html proportions)
const STUDENT_WAVEFORM   = [6, 10, 16, 12, 8, 4, 10, 14, 6, 10, 4, 8, 14, 10, 4];
const REFERENCE_WAVEFORM = [8, 12, 16, 16, 12, 8, 12, 16, 16, 12, 8, 6];

function getScoreTitle(score: number): string {
  if (score >= 90) return 'Xuất sắc! Phát âm hoàn hảo!';
  if (score >= 75) return 'Tốt lắm! Bạn đang tiến bộ!';
  if (score >= 60) return 'Khá ổn, tiếp tục cố gắng!';
  if (score >= 40) return 'Cần cải thiện thêm';
  return 'Hãy luyện tập nhiều hơn!';
}

export default function ResultScreen() {
  const { width } = useWindowDimensions();
  const isTablet = width >= BREAKPOINT_TABLET;
  const router = useRouter();
  const { score, word, problem_phonemes, feedback, audio_url } =
    useLocalSearchParams<{
      score?: string;
      word?: string;
      problem_phonemes?: string;
      feedback?: string;
      audio_url?: string;
    }>();

  // ── parse route params ───────────────────────────────────────────────────
  const scoreNum = clampScore(parseInt(score ?? '0', 10));

  const parsedProblems: ProblemPhoneme[] = (() => {
    try {
      const p = JSON.parse(problem_phonemes ?? '[]');
      return Array.isArray(p) ? p : [];
    } catch { return []; }
  })();

  const parsedFeedback: PracticeFeedback | null = (() => {
    try {
      const f = JSON.parse(feedback ?? 'null');
      if (f && !Array.isArray(f) && typeof f === 'object') return f as PracticeFeedback;
      return null;
    } catch { return null; }
  })();

  const feedbackLines = formatFeedbackLines(parsedFeedback);
  const problemLines  = formatProblemPhonemes(parsedProblems);

  const wordStatus: WordStatus =
    scoreNum >= 70 ? 'correct' : scoreNum >= 40 ? 'warning' : 'incorrect';

  const hasMeaningfulProblems =
    parsedProblems.length > 0 &&
    problemLines[0] !== 'Chưa phát hiện lỗi nổi bật.';

  // Tip: prefer feedback.tips[0], else first feedback line, else generic
  const tipText = (() => {
    if (parsedFeedback && Array.isArray(parsedFeedback.tips) && parsedFeedback.tips.length > 0) {
      const t = parsedFeedback.tips[0];
      if (typeof t === 'string' && t.trim()) return t.trim();
    }
    const first = feedbackLines[0];
    if (first && first !== 'Chưa có nhận xét chi tiết.') return first;
    return 'Luyện tập đều đặn mỗi ngày để cải thiện phát âm nhanh nhất.';
  })();

  // ── student recording playback (expo-av Audio.Sound) ─────────────────────
  const [isPlayingStudent, setIsPlayingStudent] = useState(false);
  const [studentLoading, setStudentLoading]     = useState(false);
  const studentSoundRef = useRef<Audio.Sound | null>(null);

  useEffect(() => {
    return () => {
      void studentSoundRef.current?.unloadAsync();
    };
  }, []);

  const toggleStudentPlayback = useCallback(async () => {
    const uri = Array.isArray(audio_url) ? audio_url[0] : audio_url;
    if (!uri) return;

    if (isPlayingStudent) {
      await studentSoundRef.current?.pauseAsync();
      setIsPlayingStudent(false);
      return;
    }

    setStudentLoading(true);
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
      });

      if (!studentSoundRef.current) {
        const { sound } = await Audio.Sound.createAsync(
          { uri },
          {},
          (status) => {
            if (status.isLoaded && status.didJustFinish) {
              setIsPlayingStudent(false);
            }
          },
        );
        studentSoundRef.current = sound;
      }
      await studentSoundRef.current.playAsync();
      setIsPlayingStudent(true);
    } catch {
      setIsPlayingStudent(false);
    } finally {
      setStudentLoading(false);
    }
  }, [audio_url, isPlayingStudent]);

  const hasStudentAudio = Boolean(
    Array.isArray(audio_url) ? audio_url[0] : audio_url,
  );

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {/* Back button — spans full width above the two-column area */}
        <Pressable style={styles.backBtn} onPress={() => router.back()}>
          <MaterialIcons name="arrow-back" size={24} color={colors.onSurface} />
        </Pressable>

        {/* ── Two-column container ── */}
        <View style={[styles.twoCol, isTablet && styles.twoColRow]}>

          {/* LEFT COLUMN: Score Card + Analysis Card */}
          <View style={[styles.leftCol, isTablet && styles.leftColTablet]}>

            {/* Score & Welcome Card */}
            <View style={styles.scoreCard}>
              <ScoreRing score={scoreNum} size={160} />
              <Text style={styles.scoreTitle}>{getScoreTitle(scoreNum)}</Text>
              <Text style={styles.scoreDesc}>{feedbackLines[0]}</Text>
            </View>

            {/* Interactive Sentence / Analysis Result Card */}
            <View style={styles.analysisCard}>
              <View style={styles.analysisHeader}>
                <MaterialIcons name="text-format" size={20} color={colors.primary} />
                <Text style={styles.analysisLabel}>ANALYSIS RESULT</Text>
              </View>

              {/* Single word token for target word */}
              <View style={styles.wordRow}>
                {word ? (
                  <WordToken word={word} status={wordStatus} />
                ) : (
                  <Text style={styles.noWord}>—</Text>
                )}
              </View>

              {/* Problem phonemes list (when meaningful) */}
              {hasMeaningfulProblems && (
                <View style={styles.phonemeList}>
                  {problemLines.map((line, i) => (
                    <Text key={i} style={styles.phonemeLine}>• {line}</Text>
                  ))}
                </View>
              )}

              {/* Detailed Feedback box (rose, left border) */}
              <View style={styles.feedbackBox}>
                <MaterialIcons
                  name="error"
                  size={20}
                  color={colors.danger}
                  style={styles.feedbackIcon}
                />
                <View style={styles.feedbackBody}>
                  <Text style={styles.feedbackTitle}>Detailed Feedback</Text>
                  {feedbackLines.map((line, i) => (
                    <Text key={i} style={styles.feedbackText}>{line}</Text>
                  ))}
                </View>
              </View>
            </View>

          </View>

          {/* RIGHT COLUMN: Listen & Compare + CTA + Tip + Bento */}
          <View style={[styles.rightCol, isTablet && styles.rightColTablet]}>

            {/* Listen & Compare Card */}
            <View style={styles.listenCard}>
              <Text style={styles.listenTitle}>Listen & Compare</Text>

              {/* Student recording player */}
              <View style={styles.playerCard}>
                <View style={styles.playerHeader}>
                  <Text style={styles.playerLabel}>YOUR RECORDING</Text>
                  <View style={styles.attemptBadge}>
                    <Text style={styles.attemptText}>Attempt #1</Text>
                  </View>
                </View>
                <View style={styles.playerRow}>
                  <Pressable
                    style={[
                      styles.playBtn,
                      { backgroundColor: colors.primary },
                      (!hasStudentAudio || studentLoading) && styles.playBtnDisabled,
                    ]}
                    onPress={toggleStudentPlayback}
                    disabled={!hasStudentAudio || studentLoading}
                  >
                    {studentLoading ? (
                      <ActivityIndicator size="small" color={colors.white} />
                    ) : (
                      <MaterialIcons
                        name={isPlayingStudent ? 'pause' : 'play-arrow'}
                        size={22}
                        color={colors.white}
                      />
                    )}
                  </Pressable>
                  <View style={styles.waveformBox}>
                    <WaveformBars
                      heights={STUDENT_WAVEFORM}
                      color={colors.primary}
                      barWidth={4}
                      gap={2}
                      opacity={hasStudentAudio ? (isPlayingStudent ? 1 : 0.6) : 0.25}
                      align="center"
                    />
                  </View>
                </View>
              </View>

              {/* Reference audio player (no API endpoint — show disabled) */}
              <View style={[styles.playerCard, styles.referenceCard]}>
                <View style={styles.playerHeader}>
                  <Text style={styles.playerLabel}>CORRECT PRONUNCIATION</Text>
                  <MaterialIcons name="check-circle" size={16} color={colors.primary} />
                </View>
                <View style={styles.playerRow}>
                  <Pressable
                    style={[styles.playBtn, { backgroundColor: colors.secondaryContainer }, styles.playBtnDisabled]}
                    disabled
                  >
                    <MaterialIcons name="play-arrow" size={22} color={colors.white} />
                  </Pressable>
                  <View style={styles.waveformBox}>
                    <WaveformBars
                      heights={REFERENCE_WAVEFORM}
                      color={colors.secondaryContainer}
                      barWidth={4}
                      gap={2}
                      opacity={0.25}
                      align="center"
                    />
                  </View>
                </View>
                <Text style={styles.referenceNote}>Âm mẫu chưa khả dụng</Text>
              </View>
            </View>

            {/* Practice Again CTA */}
            <Pressable style={styles.ctaButton} onPress={() => router.back()}>
              <MaterialIcons name="replay" size={22} color={colors.white} />
              <Text style={styles.ctaText}>Practice Again</Text>
            </Pressable>

            {/* Learning Tip (solid variant) */}
            <TipCard
              variant="solid"
              icon={<MaterialIcons name="lightbulb" size={24} color={colors.tertiary} />}
              title="Learning Tip"
              description={tipText}
            />

            {/* Mini stats bento — real data: score + error count */}
            <View style={styles.bento}>
              <View style={styles.bentoCell}>
                <Text style={[styles.bentoValue, { color: colors.primary }]}>
                  {scoreNum}
                </Text>
                <Text style={styles.bentoLabel}>ĐIỂM SỐ</Text>
              </View>
              <View style={styles.bentoCell}>
                <Text style={[styles.bentoValue, { color: colors.secondaryContainer }]}>
                  {parsedProblems.length}
                </Text>
                <Text style={styles.bentoLabel}>LỖI PHÁT ÂM</Text>
              </View>
            </View>

          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scroll: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xxl,
    gap: spacing.lg,
  },

  // ── two-column layout ────────────────────────────────────────────────────
  // Default (phone): single column, children stack vertically with gap.
  twoCol: {
    flexDirection: 'column',
    gap: spacing.lg,
  },
  // Tablet override: side-by-side with items anchored to their own top.
  twoColRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.lg,
  },
  // Left column (score + analysis) — full width on phone, flex:2 on tablet.
  leftCol: {
    gap: spacing.lg,
  },
  leftColTablet: {
    flex: 2,
  },
  // Right column (audio + cta + tip + bento) — full width on phone, flex:1 on tablet.
  rightCol: {
    gap: spacing.lg,
  },
  rightColTablet: {
    flex: 1,
  },

  // ── back button ──────────────────────────────────────────────────────────
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.surfaceContainerHigh,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'flex-start',
  },

  // ── score card ───────────────────────────────────────────────────────────
  scoreCard: {
    backgroundColor: colors.surfaceLowest,
    borderRadius: radius.xxl,
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.slate100,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 2,
    gap: spacing.sm,
  },
  scoreTitle: {
    ...typography.h2,
    color: colors.onSurface,
    textAlign: 'center',
    marginTop: spacing.md,
  },
  scoreDesc: {
    ...typography.bodyMd,
    color: colors.onSurfaceVariant,
    textAlign: 'center',
  },

  // ── analysis card ────────────────────────────────────────────────────────
  analysisCard: {
    backgroundColor: colors.white,
    borderRadius: radius.xxl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: colors.slate100,
    borderBottomWidth: 4,
    borderBottomColor: colors.slate100,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 3,
    gap: spacing.md,
  },
  analysisHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  analysisLabel: {
    ...typography.labelBold,
    color: colors.slate500,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  wordRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    minHeight: 56,
  },
  noWord: {
    ...typography.h1,
    color: colors.onSurfaceVariant,
  },
  phonemeList: {
    gap: 4,
    paddingVertical: spacing.xs,
  },
  phonemeLine: {
    ...typography.caption,
    color: colors.onSurfaceVariant,
    lineHeight: 18,
  },
  feedbackBox: {
    flexDirection: 'row',
    gap: spacing.md,
    backgroundColor: colors.dangerLight,
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: colors.dangerBorder,
    padding: spacing.md,
    marginTop: spacing.xs,
  },
  feedbackIcon: {
    marginTop: 2,
  },
  feedbackBody: {
    flex: 1,
    gap: 4,
  },
  feedbackTitle: {
    ...typography.labelBold,
    color: colors.dangerDark,
  },
  feedbackText: {
    fontSize: 14,
    color: colors.dangerDark,
    lineHeight: 22,
  },

  // ── listen & compare ─────────────────────────────────────────────────────
  listenCard: {
    backgroundColor: colors.surfaceContainerHigh,
    borderRadius: radius.xxl,
    padding: spacing.lg,
    gap: spacing.md,
  },
  listenTitle: {
    ...typography.labelBold,
    color: colors.onSurfaceVariant,
    paddingHorizontal: spacing.xs,
  },
  playerCard: {
    backgroundColor: colors.white,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.sm,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.04,
    shadowRadius: 4,
    elevation: 1,
  },
  referenceCard: {
    borderWidth: 2,
    borderColor: `${colors.primaryContainer}33`,
  },
  playerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  playerLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.slate400,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  attemptBadge: {
    backgroundColor: colors.successBadge,
    borderRadius: radius.full,
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
  },
  attemptText: {
    fontSize: 10,
    fontWeight: '700',
    color: colors.successMid,
  },
  playerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  playBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playBtnDisabled: {
    opacity: 0.5,
  },
  waveformBox: {
    flex: 1,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  referenceNote: {
    ...typography.caption,
    color: colors.outlineVariant,
    textAlign: 'center',
  },

  // ── CTA button ───────────────────────────────────────────────────────────
  ctaButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.primary,
    borderRadius: radius.xl,
    paddingVertical: spacing.lg,
    shadowColor: colors.onPrimaryFixedVariant,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.45,
    shadowRadius: 0,
    elevation: 6,
  },
  ctaText: {
    ...typography.bodyLg,
    color: colors.white,
    fontWeight: '700',
  },

  // ── mini stats bento ─────────────────────────────────────────────────────
  bento: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  bentoCell: {
    flex: 1,
    backgroundColor: colors.white,
    borderRadius: radius.xl,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.slate100,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.04,
    shadowRadius: 4,
    elevation: 1,
    gap: 4,
  },
  bentoValue: {
    fontSize: 28,
    fontWeight: '900',
    lineHeight: 34,
  },
  bentoLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: colors.slate400,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
});
