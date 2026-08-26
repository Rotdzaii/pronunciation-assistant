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
import { TipCard, WaveformBars, WordToken } from '../../../components/practice';
import type { WordStatus } from '../../../components/practice';
import { colors, radius, spacing, typography } from '../../../constants/theme';
import { fetchPracticeHistoryAudio, fetchPracticeStatus } from '../../../lib/api';
import { useAuth } from '../../../lib/auth';
import type { PracticeFeedback, PracticeJob, ReliabilityStatus } from '../../../types';

const COMPACT_BREAKPOINT = 768;
const WIDE_BREAKPOINT = 1200;

// Static waveform bar heights (decorative, matching result.html proportions)
const STUDENT_WAVEFORM   = [6, 10, 16, 12, 8, 4, 10, 14, 6, 10, 4, 8, 14, 10, 4];

function formatPronunciation(value: string | null): string | null {
  if (!value) return null;

  const phonetic = value.trim().replace(/^[\s/\[\]]+|[\s/\[\]]+$/g, '').trim();
  return phonetic ? `/${phonetic}/` : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asRecordList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.map(asRecord).filter((item): item is Record<string, unknown> => item !== null)
    : [];
}

type DiagnosisPhoneCandidate = {
  phone: string;
  confidence: number | null;
};

function diagnosisPhoneCandidate(value: unknown): DiagnosisPhoneCandidate | null {
  const segment = asRecord(value);
  const phone = typeof segment?.phone === 'string' ? segment.phone.trim() : '';
  if (!phone) return null;
  return {
    phone,
    confidence: typeof segment?.diagnosis_confidence === 'number' ? segment.diagnosis_confidence : null,
  };
}

function selectSuspectedPhone(
  diagnosis: Record<string, unknown> | null,
  feedback: PracticeFeedback | null,
  aiResult: Record<string, unknown> | null,
  problemPhonemes: unknown,
): DiagnosisPhoneCandidate | null {
  // Priority 1: explicit suspected_problem_phone on diagnosis
  const diagnosisPhone = typeof diagnosis?.suspected_problem_phone === 'string'
    ? diagnosis.suspected_problem_phone.trim()
    : '';
  if (diagnosisPhone) {
    return {
      phone: diagnosisPhone,
      confidence: typeof diagnosis?.diagnosis_confidence === 'number'
        ? diagnosis.diagnosis_confidence
        : null,
    };
  }

  // Priority 2: top_segment (singular) inside diagnosis paired with diagnosis-level confidence
  const topSegment = asRecord(diagnosis?.top_segment);
  const topSegmentPhone = typeof topSegment?.phone === 'string' ? topSegment.phone.trim() : '';
  if (topSegmentPhone) {
    return {
      phone: topSegmentPhone,
      confidence: typeof diagnosis?.diagnosis_confidence === 'number'
        ? diagnosis.diagnosis_confidence
        : typeof topSegment?.diagnosis_confidence === 'number'
          ? topSegment.diagnosis_confidence
          : null,
    };
  }

  // Priority 3: segment lists sorted by confidence
  const feedbackMetadata = asRecord(feedback?.metadata);
  const aiMetadata = asRecord(aiResult?.metadata);
  const topSegmentCandidates = [
    ...asRecordList(feedbackMetadata?.top_issue_segments),
    ...asRecordList(aiMetadata?.top_issue_segments),
    ...asRecordList(diagnosis?.suspected_segments),
    ...asRecordList(feedback?.segments),
    ...asRecordList(aiResult?.segments),
  ]
    .map(diagnosisPhoneCandidate)
    .filter((candidate): candidate is DiagnosisPhoneCandidate => candidate !== null)
    .sort((left, right) => (right.confidence ?? -1) - (left.confidence ?? -1));
  if (topSegmentCandidates.length > 0) return topSegmentCandidates[0];

  // Priority 4 (last resort): problem_phonemes[0]
  for (const phoneme of Array.isArray(problemPhonemes) ? problemPhonemes : []) {
    if (typeof phoneme === 'string' && phoneme.trim()) {
      return { phone: phoneme.trim(), confidence: null };
    }
    const record = asRecord(phoneme);
    const phone = typeof record?.phoneme === 'string'
      ? record.phoneme.trim()
      : typeof record?.phone === 'string'
        ? record.phone.trim()
        : '';
    if (phone) return { phone, confidence: null };
  }
  return null;
}

function errorTypeLabel(errorType: unknown): string | null {
  if (errorType === 'addition') return 'Thêm âm';
  if (errorType === 'deletion') return 'Bỏ âm';
  if (errorType === 'substitution') return 'Thay thế âm';
  return null;
}

export default function ResultScreen() {
  const { width } = useWindowDimensions();
  const isWide = width >= WIDE_BREAKPOINT;
  const stackStatCards = width < 360;
  const router = useRouter();
  const { id } = useLocalSearchParams<{ id?: string }>();
  const { accessToken } = useAuth();

  // ── fetch practice job by id ─────────────────────────────────────────────
  const [job, setJob] = useState<PracticeJob | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) { setFetchError('Không tìm thấy kết quả.'); return; }
    let cancelled = false;
    fetchPracticeStatus(id, accessToken)
      .then(data => { if (!cancelled) setJob(data); })
      .catch(() => { if (!cancelled) setFetchError('Không tải được kết quả. Vui lòng thử lại.'); });
    return () => { cancelled = true; };
  }, [id, accessToken]);

  // ── derive display values from fetched job ───────────────────────────────
  const word = job?.target_word ?? '';

  const parsedFeedback: PracticeFeedback | null =
    job?.feedback && !Array.isArray(job.feedback) && typeof job.feedback === 'object'
      ? (job.feedback as PracticeFeedback)
      : null;

  // ── reliability state (new canonical contract; defaults to 'valid' for old records) ──
  const reliabilityStatus: ReliabilityStatus =
    (parsedFeedback?.reliability as { status?: ReliabilityStatus } | undefined)?.status ?? 'valid';
  const isInvalid  = reliabilityStatus === 'invalid';
  const hasWarning = reliabilityStatus === 'valid_with_warning';
  const aiResult = asRecord(parsedFeedback?.ai_result);
  const diagnosis = asRecord(parsedFeedback?.diagnosis) ?? asRecord(aiResult?.diagnosis);
  const modelCapability =
    (typeof parsedFeedback?.model_capability === 'string' ? parsedFeedback.model_capability : null)
    ?? (typeof aiResult?.model_capability === 'string' ? aiResult.model_capability : null)
    ?? 'error_type_classifier_only';
  const isClassifierOnly = modelCapability === 'error_type_classifier_only';
  const predictedErrorType = diagnosis?.predicted_error_type ?? parsedFeedback?.predicted_error_type;
  const predictedErrorLabel = errorTypeLabel(predictedErrorType);
  const suspectedDiagnosis = isInvalid
    ? null
    : selectSuspectedPhone(diagnosis, parsedFeedback, aiResult, job?.problem_phonemes);
  const suspectedProblemPhone = suspectedDiagnosis ? formatPronunciation(suspectedDiagnosis.phone) : null;
  const diagnosisConfidence = typeof suspectedDiagnosis?.confidence === 'number'
    ? Math.round(Math.max(0, Math.min(1, suspectedDiagnosis.confidence)) * 100)
    : null;
  const hasSuspectedDiagnosis = !isInvalid && Boolean(predictedErrorLabel);
  const displayPronunciation = formatPronunciation(
    typeof parsedFeedback?.display_pronunciation === 'string'
      ? parsedFeedback.display_pronunciation
      : null,
  );
  const expectedPhones: string[] =
    Array.isArray(parsedFeedback?.expected_phones)
      ? parsedFeedback.expected_phones.filter((phone): phone is string => typeof phone === 'string')
      : [];
  const wordStatus: WordStatus = 'warning';

  const tipText = (() => {
    if (isInvalid) return 'Hãy thử lại với bản ghi âm rõ ràng hơn.';
    if (!isClassifierOnly && parsedFeedback && Array.isArray(parsedFeedback.tips) && parsedFeedback.tips.length > 0) {
      const t = parsedFeedback.tips[0];
      if (typeof t === 'string' && t.trim()) return t.trim();
    }
    return 'Luyện tập đều đặn mỗi ngày để cải thiện phát âm nhanh nhất.';
  })();

  // ── student recording playback (expo-av Audio.Sound) ─────────────────────
  const [isPlayingStudent, setIsPlayingStudent] = useState(false);
  const [studentLoading, setStudentLoading]     = useState(false);
  const [signedAudioUrl, setSignedAudioUrl]     = useState<string | null>(null);
  const [audioError, setAudioError]             = useState<string | null>(null);
  const studentSoundRef = useRef<Audio.Sound | null>(null);

  // Unload audio and reset playback state whenever the viewed result changes.
  useEffect(() => {
    return () => {
      void studentSoundRef.current?.unloadAsync();
      studentSoundRef.current = null;
      setSignedAudioUrl(null);
      setIsPlayingStudent(false);
      setAudioError(null);
    };
  }, [id]);

  const toggleStudentPlayback = useCallback(async () => {
    if (!job?.id) return;

    if (isPlayingStudent) {
      await studentSoundRef.current?.pauseAsync();
      setIsPlayingStudent(false);
      return;
    }

    setStudentLoading(true);
    setAudioError(null);
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
      });

      // Fetch a short-lived signed URL on first play; reuse on subsequent plays.
      let uri = signedAudioUrl;
      if (!uri) {
        const response = await fetchPracticeHistoryAudio(job.id, accessToken);
        uri = response.audio_url;
        setSignedAudioUrl(uri);
      }

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
        await studentSoundRef.current.playAsync();
      } else {
        await studentSoundRef.current.replayAsync();
      }
      setIsPlayingStudent(true);
    } catch {
      setIsPlayingStudent(false);
      setAudioError('Không thể phát bản ghi âm. Vui lòng thử lại.');
    } finally {
      setStudentLoading(false);
    }
  }, [job?.id, accessToken, isPlayingStudent, signedAudioUrl]);

  // A storage object path means there is audio to play (may not be an HTTP URL).
  const hasStudentAudio = Boolean(job?.audio_url);

  // ── loading / error gates ────────────────────────────────────────────────
  if (!job && !fetchError) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.centerState}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loadingText}>Đang tải kết quả...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (fetchError) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.centerState}>
          <MaterialIcons name="error-outline" size={48} color={colors.error} />
          <Text style={styles.errorText}>{fetchError}</Text>
          <Pressable style={styles.retryBtn} onPress={() => router.back()}>
            <Text style={styles.retryBtnText}>Quay lại</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

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
        <View style={[styles.twoCol, isWide && styles.twoColRow]}>

          {/* LEFT COLUMN: Score Card + Analysis Card */}
          <View style={[styles.leftCol, isWide && styles.leftColWide]}>

            {/* Score & Welcome Card */}
            <View style={styles.scoreCard}>
              {isInvalid ? (
                <>
                  <View style={styles.reliabilityBannerInvalid}>
                    <MaterialIcons name="error-outline" size={20} color="#991b1b" />
                    <Text style={styles.reliabilityBannerTextInvalid}>
                      Không thể xác định chính xác vị trí từng âm. Vui lòng thử lại.
                    </Text>
                  </View>
                </>
              ) : (
                <>
                  {isClassifierOnly ? (
                    <Text style={styles.capabilityNotice}>
                      Mô hình hiện chỉ phân loại loại lỗi khi có lỗi và chưa xác định được bạn phát âm đúng hay sai.
                    </Text>
                  ) : null}
                  {hasWarning && (
                    <View style={styles.reliabilityBannerWarning}>
                      <MaterialIcons name="warning-amber" size={16} color="#92400e" />
                      <Text style={styles.reliabilityBannerTextWarning}>
                        Chất lượng ghi âm thấp — kết quả có thể không chính xác.
                      </Text>
                    </View>
                  )}
                  {displayPronunciation && <Text style={styles.displayPronunciation}>{displayPronunciation}</Text>}
                  {!displayPronunciation && expectedPhones.length > 0 && (
                    <Text style={styles.displayPronunciation}>{formatPronunciation(expectedPhones.join(''))}</Text>
                  )}
                </>
              )}
            </View>

            {/* Interactive Sentence / Analysis Result Card */}
            <View style={styles.analysisCard}>
              <View style={styles.analysisHeader}>
                <MaterialIcons name="text-format" size={20} color={colors.primary} />
                <Text style={styles.analysisLabel}>PHÂN TÍCH LOẠI LỖI KHẢ DĨ</Text>
              </View>

              {/* Single word token for target word */}
              <View style={styles.wordRow}>
                {word ? (
                  <WordToken word={word} status={wordStatus} />
                ) : (
                  <Text style={styles.noWord}>—</Text>
                )}
              </View>

              {/* Feedback box — invalid / green / red */}
              {isInvalid ? (
                <View style={styles.feedbackBoxInvalid}>
                  <MaterialIcons name="mic-off" size={20} color="#991b1b" style={styles.feedbackIcon} />
                  <View style={styles.feedbackBody}>
                    <Text style={styles.feedbackTitleInvalid}>Không thể đánh giá</Text>
                    <Text style={styles.feedbackTextInvalid}>
                      Không thể xác định chính xác vị trí từng âm. Vui lòng thử lại.
                    </Text>
                  </View>
                </View>
              ) : (
              <View style={styles.feedbackBox}>
                <MaterialIcons
                  name="warning-amber"
                  size={20}
                  color={colors.warning}
                  style={styles.feedbackIcon}
                />
                <View style={styles.feedbackBody}>
                  <Text style={styles.feedbackTitle}>
                    {hasSuspectedDiagnosis ? 'Loại lỗi mô hình nghi ngờ' : 'Chưa có loại lỗi dự đoán'}
                  </Text>
                  {hasSuspectedDiagnosis ? (
                    <>
                      <Text style={styles.feedbackText}>Loại lỗi dự đoán: {predictedErrorLabel}</Text>
                      {suspectedProblemPhone ? <Text style={styles.feedbackText}>Mô hình nghi ngờ lỗi tại âm {suspectedProblemPhone}.</Text> : null}
                      {diagnosisConfidence !== null ? <Text style={styles.feedbackText}>Độ tin cậy phân loại lỗi: {diagnosisConfidence}%.</Text> : null}
                    </>
                  ) : (
                    <Text style={styles.feedbackText}>
                      Mô hình chưa có đủ tín hiệu để đưa ra loại lỗi dự đoán.
                    </Text>
                  )}
                </View>
              </View>
              )}
            </View>

          </View>

          {/* RIGHT COLUMN: Listen & Compare + CTA + Tip + Bento */}
          <View style={[styles.rightCol, isWide && styles.rightColWide]}>

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

              {audioError ? (
                <Text style={styles.audioErrorText}>{audioError}</Text>
              ) : null}

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
            <View style={[styles.bento, stackStatCards && styles.bentoStack]}>
              <View style={styles.bentoCell}>
                <Text style={[styles.bentoValue, { color: isInvalid ? colors.slate400 : colors.primary }]}>
                  {isInvalid || !hasSuspectedDiagnosis ? '—' : predictedErrorLabel}
                </Text>
                <Text style={styles.bentoLabel}>LOẠI LỖI MÔ HÌNH NGHI NGỜ</Text>
              </View>
              <View style={styles.bentoCell}>
                <Text style={[styles.bentoValue, { color: isInvalid ? colors.slate400 : colors.secondaryContainer }]}>
                  {isInvalid || diagnosisConfidence === null ? '—' : `${diagnosisConfidence}%`}
                </Text>
                <Text style={styles.bentoLabel}>ĐỘ TIN CẬY PHÂN LOẠI</Text>
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
    width: '100%',
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
    width: '100%',
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
    minWidth: 0,
  },
  leftColWide: {
    flex: 2,
  },
  // Right column (audio + cta + tip + bento) — full width on phone, flex:1 on tablet.
  rightCol: {
    gap: spacing.lg,
    minWidth: 0,
  },
  rightColWide: {
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
  capabilityNotice: {
    ...typography.bodyMd,
    color: colors.onSurfaceVariant,
    textAlign: 'center',
    lineHeight: 22,
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
  phonemeLineRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  phonemeLine: {
    ...typography.caption,
    color: colors.danger,
    fontWeight: '600',
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
  feedbackBoxSuccess: {
    flexDirection: 'row',
    gap: spacing.md,
    backgroundColor: colors.successLight,
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: colors.success,
    padding: spacing.md,
    marginTop: spacing.xs,
  },
  feedbackTitleSuccess: {
    ...typography.labelBold,
    color: colors.successDark,
  },
  feedbackTextSuccess: {
    fontSize: 14,
    color: colors.successDark,
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
  audioErrorText: {
    ...typography.caption,
    color: colors.error,
    textAlign: 'center',
    paddingHorizontal: spacing.xs,
  },
  referenceNote: {
    ...typography.caption,
    color: colors.outlineVariant,
    textAlign: 'center',
  },

  // ── CTA button ───────────────────────────────────────────────────────────
  ctaButton: {
    width: '100%',
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
  bentoStack: {
    flexDirection: 'column',
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

  // ── reliability banners ─────────────────────────────────────────────────
  reliabilityBannerInvalid: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: '#fee2e2',
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
    padding: spacing.md,
    marginVertical: spacing.sm,
    width: '100%',
  },
  reliabilityBannerTextInvalid: {
    flex: 1,
    fontSize: 13,
    color: '#991b1b',
    lineHeight: 20,
  },
  reliabilityBannerWarning: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: colors.warningLight,
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: colors.warning,
    padding: spacing.sm,
    width: '100%',
  },
  reliabilityBannerTextWarning: {
    flex: 1,
    fontSize: 12,
    color: '#92400e',
    lineHeight: 18,
  },

  // ── score type & display pronunciation ──────────────────────────────────
  scoreTypeLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.slate400,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  displayPronunciation: {
    ...typography.bodyMd,
    color: colors.slate500,
    fontStyle: 'italic',
    letterSpacing: 0.5,
  },

  // ── phone results (canonical IPA chips) ──────────────────────────────────
  phoneResultsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    paddingVertical: spacing.xs,
  },
  phoneChip: {
    fontSize: 15,
    fontWeight: '600',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.sm,
    backgroundColor: colors.slate100,
    color: colors.onSurface,
    overflow: 'hidden',
  },
  phoneChipOk: {
    backgroundColor: '#d1fae5',
    color: '#065f46',
  },
  phoneChipBad: {
    backgroundColor: colors.dangerLight,
    color: colors.dangerDark,
  },

  // ── feedback box — invalid state ─────────────────────────────────────────
  feedbackBoxInvalid: {
    flexDirection: 'row',
    gap: spacing.md,
    backgroundColor: '#fee2e2',
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
    padding: spacing.md,
    marginTop: spacing.xs,
  },
  feedbackTitleInvalid: {
    ...typography.labelBold,
    color: '#991b1b',
  },
  feedbackTextInvalid: {
    fontSize: 14,
    color: '#991b1b',
    lineHeight: 22,
  },

  // ── loading / error state ────────────────────────────────────────────────
  centerState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  loadingText: {
    fontSize: 15,
    color: colors.onSurfaceVariant,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 15,
    color: colors.error,
    textAlign: 'center',
    lineHeight: 22,
  },
  retryBtn: {
    marginTop: spacing.sm,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary,
    borderRadius: radius.xl,
  },
  retryBtnText: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.white,
  },
});
