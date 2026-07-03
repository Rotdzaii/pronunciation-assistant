import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Modal,
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
import { ErrorState, LoadingState } from '../../../components/AppUI';
import { MicButton, WaveformBars } from '../../../components/practice';
import { colors, radius, spacing, typography } from '../../../constants/theme';
import {
  fetchAssignmentWords,
  fetchAssignmentStatus,
  startAssessment,
  submitAssessment,
  uploadPracticeAudio,
  createPracticeJob,
} from '../../../lib/api';
import { useAuth } from '../../../lib/auth';
import type { AssignmentWords } from '../../../types';

// ── types ─────────────────────────────────────────────────────────────────────
type WordState = {
  item_id: string;
  word: string;
  phonetic: string | null;
  audio_url: string | null;
  practice_job_id: string | null;
  recorded: boolean;
};

const WAVEFORM_BARS = [18, 36, 24, 48, 30, 42, 18];

export default function AssessmentScreen() {
  const { assignment_id } = useLocalSearchParams<{ assignment_id?: string }>();
  const { accessToken } = useAuth();
  const router = useRouter();
  const { width } = useWindowDimensions();
  const isDesktop = width >= 768;

  // ── load state ────────────────────────────────────────────────────────────
  const [loadingPhase, setLoadingPhase] = useState<'checking' | 'ready' | 'locked' | 'error'>('checking');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [assignmentWords, setAssignmentWords] = useState<AssignmentWords | null>(null);

  // ── assessment run state ──────────────────────────────────────────────────
  const [wordIndex, setWordIndex] = useState(0);
  const [wordStates, setWordStates] = useState<WordState[]>([]);
  const [timeLeft, setTimeLeft] = useState(60);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // ── submission state ──────────────────────────────────────────────────────
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);

  useEffect(() => {
    if (!assignment_id) { setLoadError('Thiếu assignment_id.'); setLoadingPhase('error'); return; }
    void init(assignment_id);
  }, [assignment_id]);

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current);
    void soundRef.current?.unloadAsync();
  }, []);

  const init = async (id: string) => {
    try {
      const statusRes = await fetchAssignmentStatus(id);
      if (statusRes.is_locked || !statusRes.can_start) {
        setLoadingPhase('locked');
        return;
      }
      await startAssessment(id);
      const words = await fetchAssignmentWords(id);
      setAssignmentWords(words);
      setWordStates(words.items.map(item => ({
        item_id: item.id,
        word: item.word,
        phonetic: item.phonetic,
        audio_url: null,
        practice_job_id: null,
        recorded: false,
      })));
      setTimeLeft(words.timer_per_word_seconds);
      setLoadingPhase('ready');
      startWordTimer(words.timer_per_word_seconds);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Không thể tải bài kiểm tra.');
      setLoadingPhase('error');
    }
  };

  const startWordTimer = (seconds: number) => {
    if (timerRef.current) clearInterval(timerRef.current);
    setTimeLeft(seconds);
    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current!);
          timerRef.current = null;
          void advanceWord();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const advanceWord = async () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (recording) await stopRecording();
    setAudioUri(null);
    setRecording(null);
    setActionError(null);
    setWordIndex(prev => {
      const next = prev + 1;
      if (assignmentWords && next < assignmentWords.items.length) {
        startWordTimer(assignmentWords.timer_per_word_seconds);
      }
      return next;
    });
  };

  const startRecording = async () => {
    setActionError(null);
    const perm = await Audio.requestPermissionsAsync();
    if (!perm.granted) { setActionError('Cần quyền micro để ghi âm.'); return; }
    if (soundRef.current) { await soundRef.current.unloadAsync(); soundRef.current = null; }
    await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
    const { recording: rec } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
    setRecording(rec);
  };

  const stopRecording = async (): Promise<string | null> => {
    if (!recording) return null;
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI() ?? null;
    setAudioUri(uri);
    setRecording(null);
    return uri;
  };

  const handleMicPress = async () => {
    if (recording) { await stopRecording(); } else { await startRecording(); }
  };

  const handleSaveRecording = async () => {
    const uri = audioUri;
    if (!uri || !assignment_id || !assignmentWords) return;
    const currentWord = assignmentWords.items[wordIndex];
    if (!currentWord || !accessToken) return;
    setUploading(true);
    setActionError(null);
    try {
      const upload = await uploadPracticeAudio(uri, accessToken);
      const jobRes = await createPracticeJob(
        {
          target_word: currentWord.word,
          audio_url: upload.audio_url || upload.storage_path,
          assignment_id,
          item_id: currentWord.id,
        },
        accessToken,
      );
      setWordStates(prev => prev.map((ws, i) =>
        i === wordIndex
          ? { ...ws, audio_url: upload.audio_url || upload.storage_path, practice_job_id: jobRes.job_id, recorded: true }
          : ws
      ));
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Không lưu được bản ghi.');
    } finally {
      setUploading(false);
    }
  };

  const handleSubmitConfirm = () => {
    Alert.alert(
      'Nộp bài kiểm tra',
      'Bạn có chắc chắn muốn nộp bài? Hành động này không thể hoàn tác.',
      [
        { text: 'Huỷ', style: 'cancel' },
        { text: 'Nộp bài', style: 'destructive', onPress: () => void handleSubmit() },
      ],
    );
  };

  const handleSubmit = async () => {
    if (!assignment_id) return;
    setSubmitting(true);
    try {
      await submitAssessment(assignment_id);
      setSubmitted(true);
    } catch (err) {
      Alert.alert('Lỗi', err instanceof Error ? err.message : 'Không thể nộp bài.');
    } finally {
      setSubmitting(false);
    }
  };

  const recordedCount = wordStates.filter(w => w.recorded).length;
  const totalWords = assignmentWords?.items.length ?? 0;
  const isLastWord = wordIndex >= totalWords - 1;
  const currentItem = assignmentWords?.items[wordIndex] ?? null;
  const timerPct = assignmentWords ? (timeLeft / assignmentWords.timer_per_word_seconds) * 100 : 100;
  const timerColor = timerPct > 40 ? colors.success : timerPct > 15 ? colors.warning : colors.error;

  // ── render states ─────────────────────────────────────────────────────────
  if (loadingPhase === 'checking') return <LoadingState title="Đang tải bài kiểm tra" message="Vui lòng chờ..." />;

  if (loadingPhase === 'error') return (
    <SafeAreaView style={styles.root}>
      <ErrorState title="Không thể tải bài kiểm tra" message={loadError ?? ''} />
      <Pressable style={[styles.btn, styles.btnPrimary, { marginTop: spacing.lg, marginHorizontal: spacing.lg }]} onPress={() => router.back()}>
        <Text style={styles.btnPrimaryText}>Quay lại</Text>
      </Pressable>
    </SafeAreaView>
  );

  if (loadingPhase === 'locked') return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.lockedCard}>
          <MaterialIcons name="lock" size={48} color={colors.outline} />
          <Text style={styles.lockedTitle}>Bài kiểm tra đã đóng</Text>
          <Text style={styles.lockedDesc}>Thời hạn nộp bài đã qua hoặc bài đã được nộp.</Text>
          <Pressable style={[styles.btn, styles.btnPrimary]} onPress={() => router.back()}>
            <Text style={styles.btnPrimaryText}>Quay lại</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );

  if (submitted) return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.lockedCard}>
          <MaterialIcons name="check-circle" size={56} color={colors.success} />
          <Text style={styles.lockedTitle}>Đã nộp bài thành công!</Text>
          <Text style={styles.lockedDesc}>Giáo viên sẽ xem kết quả và phản hồi sớm nhất có thể.</Text>
          <Text style={styles.submittedStats}>{recordedCount}/{totalWords} từ đã ghi âm</Text>
          <Pressable style={[styles.btn, styles.btnPrimary]} onPress={() => router.back()}>
            <Text style={styles.btnPrimaryText}>Về trang chính</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );

  // All words done — show submit screen
  if (wordIndex >= totalWords && totalWords > 0) return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.submitCard}>
          <Text style={styles.submitTitle}>Hoàn thành bài kiểm tra</Text>
          <Text style={styles.submitDesc}>Bạn đã ghi âm xong tất cả các từ. Xem lại và nộp bài khi sẵn sàng.</Text>
          <View style={styles.statsRow}>
            <View style={styles.statCell}>
              <Text style={[styles.statValue, { color: colors.primary }]}>{recordedCount}</Text>
              <Text style={styles.statLabel}>Đã ghi âm</Text>
            </View>
            <View style={styles.statCell}>
              <Text style={[styles.statValue, { color: colors.outline }]}>{totalWords - recordedCount}</Text>
              <Text style={styles.statLabel}>Bỏ qua</Text>
            </View>
          </View>
          <View style={styles.wordSummaryList}>
            {wordStates.map((ws, i) => (
              <View key={ws.item_id} style={styles.wordSummaryRow}>
                <MaterialIcons
                  name={ws.recorded ? 'check-circle' : 'radio-button-unchecked'}
                  size={20}
                  color={ws.recorded ? colors.success : colors.outlineVariant}
                />
                <Text style={styles.wordSummaryText}>{ws.word}</Text>
                {ws.phonetic ? <Text style={styles.wordSummaryIpa}>{ws.phonetic}</Text> : null}
              </View>
            ))}
          </View>
          <Pressable
            style={[styles.btn, styles.btnPrimary, submitting && styles.btnDisabled]}
            disabled={submitting}
            onPress={handleSubmitConfirm}
          >
            {submitting ? <ActivityIndicator color="#fff" size="small" style={{ marginRight: 8 }} /> : null}
            <Text style={styles.btnPrimaryText}>Nộp bài</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );

  // Active word screen
  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        {/* Progress header */}
        <View style={styles.progressHeader}>
          <Text style={styles.progressLabel}>Từ {wordIndex + 1} / {totalWords}</Text>
          <View style={styles.streakBadge}>
            <MaterialIcons name="mic" size={14} color={colors.primary} />
            <Text style={styles.streakText}>{recordedCount} đã ghi</Text>
          </View>
        </View>

        {/* Timer bar */}
        <View style={styles.timerTrack}>
          <View style={[styles.timerFill, { width: `${timerPct}%` as any, backgroundColor: timerColor }]} />
        </View>
        <View style={styles.timerRow}>
          <MaterialIcons name="timer" size={16} color={timerColor} />
          <Text style={[styles.timerText, { color: timerColor }]}>{timeLeft}s</Text>
        </View>

        {/* Word display */}
        <View style={styles.wordCard}>
          <Text style={styles.wordText}>{currentItem?.word ?? ''}</Text>
          {currentItem?.phonetic ? (
            <View style={styles.ipaPill}>
              <Text style={styles.ipaText}>{currentItem.phonetic}</Text>
            </View>
          ) : null}
          {currentItem?.meaning_vi ? (
            <Text style={styles.meaningText}>{currentItem.meaning_vi}</Text>
          ) : null}
        </View>

        {/* Mic module */}
        <View style={styles.micModule}>
          <View style={styles.waveformAbsolute}>
            <WaveformBars heights={WAVEFORM_BARS} color={colors.micActive} barWidth={8} gap={4} opacity={recording ? 0.5 : 0.18} align="center" />
          </View>
          <MicButton isRecording={!!recording} onPress={handleMicPress} />
        </View>

        {/* Status hints */}
        {!recording && !audioUri && <Text style={styles.hintText}>Chạm micro để ghi âm</Text>}
        {!recording && audioUri && !wordStates[wordIndex]?.recorded && (
          <Text style={styles.readyText}>Bản ghi sẵn sàng — bấm Lưu hoặc ghi lại</Text>
        )}
        {wordStates[wordIndex]?.recorded && (
          <Text style={styles.savedText}>✓ Đã lưu — ghi lại nếu muốn thay thế</Text>
        )}

        {actionError ? <Text style={styles.errorText}>{actionError}</Text> : null}

        {/* Actions */}
        <View style={styles.actionRow}>
          <Pressable
            style={[styles.btn, styles.btnSecondary, (!audioUri || uploading || !!recording) && styles.btnDisabled]}
            disabled={!audioUri || uploading || !!recording}
            onPress={handleSaveRecording}
          >
            {uploading ? <ActivityIndicator color={colors.primary} size="small" style={{ marginRight: 6 }} /> : null}
            <Text style={styles.btnSecondaryText}>{uploading ? 'Đang lưu...' : 'Lưu bản ghi'}</Text>
          </Pressable>
          <Pressable
            style={[styles.btn, styles.btnPrimary, uploading && styles.btnDisabled]}
            disabled={uploading}
            onPress={isLastWord ? handleSubmitConfirm : advanceWord}
          >
            <Text style={styles.btnPrimaryText}>{isLastWord ? 'Nộp bài' : 'Từ tiếp theo →'}</Text>
          </Pressable>
        </View>

        {/* Word list progress dots */}
        <View style={styles.dotRow}>
          {wordStates.map((ws, i) => (
            <View
              key={ws.item_id}
              style={[
                styles.dot,
                i === wordIndex && styles.dotActive,
                ws.recorded && styles.dotDone,
                i < wordIndex && !ws.recorded && styles.dotSkipped,
              ]}
            />
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  scroll: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.xxl, gap: spacing.md },
  progressHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  progressLabel: { fontSize: 13, fontWeight: '700', color: colors.onSurfaceVariant },
  streakBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: colors.primaryFixed, borderRadius: radius.full, paddingHorizontal: spacing.sm, paddingVertical: 4 },
  streakText: { fontSize: 12, fontWeight: '700', color: colors.primary },
  timerTrack: { height: 6, borderRadius: 3, backgroundColor: colors.outlineVariant, overflow: 'hidden' },
  timerFill: { height: '100%', borderRadius: 3 },
  timerRow: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-end' },
  timerText: { fontSize: 13, fontWeight: '700' },
  wordCard: { alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.xl, backgroundColor: colors.surfaceLowest, borderRadius: radius.xxl, borderWidth: 1, borderColor: colors.outlineVariant },
  wordText: { ...typography.h1, color: colors.onSurface, textAlign: 'center' },
  ipaPill: { backgroundColor: colors.surfaceContainerHigh, borderRadius: radius.md, paddingHorizontal: spacing.sm, paddingVertical: 4 },
  ipaText: { fontSize: 15, fontWeight: '500', color: colors.outline },
  meaningText: { fontSize: 14, color: colors.onSurfaceVariant, textAlign: 'center', paddingHorizontal: spacing.lg },
  micModule: { width: '100%', height: 220, alignItems: 'center', justifyContent: 'center' },
  waveformAbsolute: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  hintText: { fontSize: 13, color: colors.onSurfaceVariant, textAlign: 'center' },
  readyText: { fontSize: 13, fontWeight: '600', color: colors.success, textAlign: 'center' },
  savedText: { fontSize: 13, fontWeight: '600', color: colors.primary, textAlign: 'center' },
  errorText: { fontSize: 13, color: colors.error, textAlign: 'center' },
  actionRow: { flexDirection: 'row', gap: spacing.sm },
  btn: { flex: 1, minHeight: 48, borderRadius: radius.md, alignItems: 'center', justifyContent: 'center', flexDirection: 'row', gap: 6 },
  btnPrimary: { backgroundColor: colors.primary },
  btnPrimaryText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  btnSecondary: { borderWidth: 1.5, borderColor: colors.primary, backgroundColor: 'transparent' },
  btnSecondaryText: { color: colors.primary, fontSize: 14, fontWeight: '700' },
  btnDisabled: { opacity: 0.45 },
  dotRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, justifyContent: 'center', paddingTop: spacing.sm },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.outlineVariant },
  dotActive: { backgroundColor: colors.primary, width: 20 },
  dotDone: { backgroundColor: colors.success },
  dotSkipped: { backgroundColor: colors.outlineVariant, opacity: 0.4 },
  lockedCard: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, paddingTop: spacing.xxl },
  lockedTitle: { ...typography.h2, color: colors.onSurface, textAlign: 'center' },
  lockedDesc: { fontSize: 14, color: colors.onSurfaceVariant, textAlign: 'center', lineHeight: 22 },
  submitCard: { gap: spacing.md, paddingTop: spacing.md },
  submitTitle: { ...typography.h2, color: colors.onSurface },
  submitDesc: { fontSize: 14, color: colors.onSurfaceVariant, lineHeight: 22 },
  statsRow: { flexDirection: 'row', gap: spacing.md },
  statCell: { flex: 1, backgroundColor: colors.surfaceLowest, borderRadius: radius.lg, padding: spacing.md, alignItems: 'center', borderWidth: 1, borderColor: colors.outlineVariant },
  statValue: { fontSize: 32, fontWeight: '900' },
  statLabel: { fontSize: 12, fontWeight: '700', color: colors.slate400, textTransform: 'uppercase' },
  wordSummaryList: { gap: 8 },
  wordSummaryRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingVertical: 6 },
  wordSummaryText: { flex: 1, fontSize: 15, fontWeight: '600', color: colors.onSurface },
  wordSummaryIpa: { fontSize: 13, color: colors.outline },
  submittedStats: { fontSize: 16, fontWeight: '700', color: colors.primary },
});
