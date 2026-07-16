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
import { ErrorState } from '../../../components/AppUI';
import {
  CircleIconButton,
  MicButton,
  StreakToast,
  TipCard,
  WaveformBars,
} from '../../../components/practice';
import { colors, radius, spacing, typography } from '../../../constants/theme';
import {
  createPracticeJob,
  fetchPracticeStatus,
  fetchAssignmentWords,
  fetchVocabularyItems,
  fetchVocabularySets,
  fetchVocabularySetDetail,
  fetchWordPronunciation,
  uploadPracticeAudio,
} from '../../../lib/api';
import { useAuth } from '../../../lib/auth';
import type { PracticeJob, PracticeJobStatus, VocabularyItem, VocabularySet } from '../../../types';

const WAVEFORM_BARS = [24, 48, 32, 64, 40, 56, 24];

const TOPICS = [
  { value: 'daily life',               label: 'Cuộc sống hàng ngày' },
  { value: 'school',                   label: 'Trường học' },
  { value: 'travel',                   label: 'Du lịch' },
  { value: 'work',                     label: 'Công việc' },
  { value: 'technology',               label: 'Công nghệ' },
  { value: 'pronunciation challenges', label: 'Thử thách phát âm' },
];

// Duration the streak toast stays visible before navigating to result screen.
// Adjust here if you want longer/shorter display time.
const TOAST_DURATION_MS = 3000;

export default function PracticeScreen() {
  const { width, height } = useWindowDimensions();
  const isDesktop = width >= 768;
  const { accessToken } = useAuth();
  const router = useRouter();
  const { assignment_id: assignmentId } = useLocalSearchParams<{ assignment_id?: string }>();

  // ── select-screen state ──────────────────────────────────────────────────
  const [mode, setMode] = useState<'select' | 'practice'>('select');
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedSetId, setSelectedSetId] = useState<string | null>(null);
  const [sets, setSets] = useState<VocabularySet[]>([]);
  const [setsLoading, setSetsLoading] = useState(true);
  const [startLoading, setStartLoading] = useState(false);
  const [assignmentWords, setAssignmentWords] = useState<VocabularyItem[]>([]);
  const [assignmentWordIndex, setAssignmentWordIndex] = useState(0);

  // ── practice state ───────────────────────────────────────────────────────
  const [practiceTarget, setPracticeTarget] = useState<VocabularyItem | null>(null);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<PracticeJob | null>(null);
  const [jobStatus, setJobStatus] = useState<PracticeJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // ── word sample audio state (replaces phoneme modal) ────────────────────
  const [wordSampleUrl, setWordSampleUrl] = useState<string | null>(null);
  const [wordSampleLoading, setWordSampleLoading] = useState(false);
  const [isPlayingWordSample, setIsPlayingWordSample] = useState(false);
  const wordSampleSoundRef = useRef<Audio.Sound | null>(null);

  // ── streak state ─────────────────────────────────────────────────────────
  // streak = consecutive words with score > 80 in this session.
  // Reset to 0 when going back to select screen.
  const streakRef = useRef(0);  // ref avoids stale-closure in effects
  const [toasts, setToasts] = useState<Array<{ id: string; message: string }>>([]);

  // ── refs ─────────────────────────────────────────────────────────────────
  const pollingRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const navTimerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (pollingRef.current)  clearInterval(pollingRef.current);
      if (timerRef.current)    clearInterval(timerRef.current);
      if (navTimerRef.current) clearTimeout(navTimerRef.current);
    };
  }, []);

  // Assignment practice deliberately bypasses the free-practice picker: only
  // the word IDs returned by the server for this assignment can be submitted.
  useEffect(() => {
    if (!assignmentId) return;
    let cancelled = false;
    fetchAssignmentWords(assignmentId)
      .then((data) => {
        if (cancelled) return;
        const items = data.items.map((item) => ({
          ...item, difficulty: null, sample_sentence: null, target_phonemes: [], common_mistake_tags: [], stress_pattern: null,
        }));
        if (items.length === 0) throw new Error('Bài tập này chưa có từ để luyện.');
        setAssignmentWords(items);
        setPracticeTarget(items[0]);
        setMode('practice');
      })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : 'Không thể tải bài tập.'); });
    return () => { cancelled = true; };
  }, [assignmentId]);

  useEffect(() => {
    return () => {
      if (sound) void sound.unloadAsync();
    };
  }, [sound]);

  useEffect(() => {
    return () => { void wordSampleSoundRef.current?.unloadAsync(); };
  }, []);

  // ── fetch word sample audio when target changes ───────────────────────────
  useEffect(() => {
    setWordSampleUrl(null);
    void wordSampleSoundRef.current?.unloadAsync();
    wordSampleSoundRef.current = null;
    setIsPlayingWordSample(false);
    if (!practiceTarget?.word || !accessToken) return;
    let cancelled = false;
    fetchWordPronunciation(practiceTarget.word, accessToken)
      .then(data => { if (!cancelled) setWordSampleUrl(data.audio_url ?? null); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [practiceTarget?.word, accessToken]);

  const toggleWordSamplePlayback = useCallback(async () => {
    if (!wordSampleUrl) return;
    if (isPlayingWordSample) {
      await wordSampleSoundRef.current?.pauseAsync();
      setIsPlayingWordSample(false);
      return;
    }
    setWordSampleLoading(true);
    try {
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
      if (!wordSampleSoundRef.current) {
        const { sound: ws } = await Audio.Sound.createAsync(
          { uri: wordSampleUrl },
          {},
          (s) => { if (s.isLoaded && s.didJustFinish) setIsPlayingWordSample(false); },
        );
        wordSampleSoundRef.current = ws;
      }
      await wordSampleSoundRef.current.playAsync();
      setIsPlayingWordSample(true);
    } catch {
      setIsPlayingWordSample(false);
    } finally {
      setWordSampleLoading(false);
    }
  }, [wordSampleUrl, isPlayingWordSample]);

  // ── fetch sets for select screen ─────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetchVocabularySets(50, 0)
      .then(data => { if (!cancelled) setSets(data.items); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setSetsLoading(false); });
    return () => { cancelled = true; };
  }, []);

  // ── navigate + streak logic when scoring completes ────────────────────────
  useEffect(() => {
    if (jobStatus !== 'completed' || !job || !practiceTarget) return;

    if (assignmentId) {
      setAssignmentWordIndex((current) => {
        const next = assignmentWords.length ? (current + 1) % assignmentWords.length : 0;
        setPracticeTarget(assignmentWords[next] ?? practiceTarget);
        return next;
      });
      setAudioUri(null);
      setJob(null);
      setJobId(null);
      setJobStatus(null);
      setElapsedSeconds(0);
      return;
    }

    // Update streak using a ref to always have the latest value.
    const score = job.score ?? 0;
    const newStreak = score > 80 ? streakRef.current + 1 : 0;
    streakRef.current = newStreak;

    const doNavigate = () => {
      router.push({
        pathname: '/practice/result' as any,
        params: { id: jobId },
      });
    };

    // Show toast on multiples of 5 (5, 10, 15 …), then navigate after delay.
    // For a one-shot milestone per session change the condition to `=== 5`.
    if (newStreak > 0 && newStreak % 5 === 0) {
      const toastId = String(Date.now());
      setToasts(prev => [
        ...prev,
        { id: toastId, message: `${newStreak} từ liên tiếp đạt điểm cao!` },
      ]);
      navTimerRef.current = setTimeout(doNavigate, TOAST_DURATION_MS);
    } else {
      doNavigate();
    }

    return () => {
      if (navTimerRef.current) clearTimeout(navTimerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobStatus]);

  // ── select screen handlers ───────────────────────────────────────────────
  const handleStartPractice = async () => {
    setStartLoading(true);
    setError(null);
    try {
      let item: VocabularyItem | null = null;

      if (assignmentId) {
        item = assignmentWords[assignmentWordIndex] ?? null;
        if (!item) throw new Error('Bài tập này chưa có từ để luyện.');
      } else if (selectedSetId) {
        const detail = await fetchVocabularySetDetail(selectedSetId);
        if (detail.items.length === 0) throw new Error('Bộ từ này chưa có từ nào.');
        item = detail.items[Math.floor(Math.random() * detail.items.length)];
      } else if (selectedTopic) {
        const data = await fetchVocabularyItems(50, 0, selectedTopic);
        if (data.items.length === 0) throw new Error('Chủ đề này chưa có từ nào.');
        item = data.items[Math.floor(Math.random() * data.items.length)];
      } else {
        throw new Error('Vui lòng chọn chủ đề hoặc bộ từ.');
      }

      setPracticeTarget(item);
      setAudioUri(null);
      setJob(null);
      setJobId(null);
      setJobStatus(null);
      setElapsedSeconds(0);
      setError(null);
      setMode('practice');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không tải được từ luyện tập.');
    } finally {
      setStartLoading(false);
    }
  };

  const handleBackToSelect = () => {
    if (assignmentId) { router.back(); return; }
    if (timerRef.current)    { clearInterval(timerRef.current);  timerRef.current = null; }
    if (pollingRef.current)  { clearInterval(pollingRef.current); pollingRef.current = null; }
    if (navTimerRef.current) { clearTimeout(navTimerRef.current); navTimerRef.current = null; }
    void Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
    void wordSampleSoundRef.current?.unloadAsync();
    wordSampleSoundRef.current = null;
    setWordSampleUrl(null);
    setIsPlayingWordSample(false);
    setMode('select');
    setPracticeTarget(null);
    setAudioUri(null);
    setJob(null);
    setJobId(null);
    setJobStatus(null);
    setRecording(null);
    setError(null);
    setElapsedSeconds(0);
    // Reset streak when leaving the practice session
    streakRef.current = 0;
    setToasts([]);
  };

  // ── recording ────────────────────────────────────────────────────────────
  const startRecording = async () => {
    setError(null);
    const permission = await Audio.requestPermissionsAsync();
    if (!permission.granted) {
      setError('Cần quyền micro để ghi âm.');
      return;
    }

    if (sound) {
      await sound.unloadAsync();
      setSound(null);
    }

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });

    const { recording } = await Audio.Recording.createAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY,
    );
    setElapsedSeconds(0);
    timerRef.current = setInterval(() => setElapsedSeconds(s => s + 1), 1000);
    setRecording(recording);
  };

  // Returns the recorded URI so callers can immediately use it without
  // waiting for the setState to propagate.
  const stopRecording = async (): Promise<string | null> => {
    if (!recording) return null;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    await recording.stopAndUnloadAsync();
    await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
    const uri = recording.getURI();
    setAudioUri(uri ?? null);
    setRecording(null);
    return uri ?? null;
  };

  // Accepts an optional URI so it can be called immediately after
  // stopRecording() without relying on the audioUri state update.
  const handleCreateJob = async (audioUriOverride?: string) => {
    const uri = audioUriOverride ?? audioUri;
    if (!uri) { setError('Hãy ghi âm trước khi gửi.'); return; }
    if (!practiceTarget) { setError('Chưa tải được từ mục tiêu. Vui lòng thử lại.'); return; }
    if (!accessToken) { setError('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.'); return; }

    setLoading(true);
    setError(null);
    try {
      const uploadResponse = await uploadPracticeAudio(uri, accessToken);
      const response = await createPracticeJob(
        {
          target_word: practiceTarget.word,
          audio_url: uploadResponse.audio_url || uploadResponse.storage_path,
          assignment_id: assignmentId ?? null,
          item_id: assignmentId ? practiceTarget.id : null,
          audio_storage_path: uploadResponse.storage_path,
        },
        accessToken,
      );
      setJob(null);
      setJobId(response.job_id);
      setJobStatus(response.status);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không tạo được lượt chấm.');
    } finally {
      setLoading(false);
    }
  };

  // MicButton toggles recording only — submit is a separate user action.
  const handleMicPress = async () => {
    if (recording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  };

  const handleSkipWord = () => {
    if (timerRef.current)    { clearInterval(timerRef.current);  timerRef.current = null; }
    if (pollingRef.current)  { clearInterval(pollingRef.current); pollingRef.current = null; }
    if (navTimerRef.current) { clearTimeout(navTimerRef.current); navTimerRef.current = null; }
    if (recording) { void recording.stopAndUnloadAsync(); }
    void handleStartPractice();
  };

  const playRecording = async () => {
    if (!audioUri) return;
    setError(null);
    if (sound) { await sound.replayAsync(); return; }
    const { sound: nextSound } = await Audio.Sound.createAsync({ uri: audioUri });
    setSound(nextSound);
    await nextSound.playAsync();
  };

  useEffect(() => {
    if (!jobId) return;
    if (pollingRef.current) clearInterval(pollingRef.current);

    pollingRef.current = setInterval(async () => {
      try {
        const response = await fetchPracticeStatus(jobId, accessToken);
        setJob(response);
        setJobStatus(response.status);
        if (response.status === 'completed' || response.status === 'failed') {
          if (pollingRef.current) clearInterval(pollingRef.current);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không lấy được kết quả.');
      }
    }, 3000);

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [jobId, accessToken]);

  // ── derived ──────────────────────────────────────────────────────────────
  const mm = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0');
  const ss = String(elapsedSeconds % 60).padStart(2, '0');
  const canStart = (!!selectedTopic || !!selectedSetId) && !startLoading;
  const isPolling = !!jobStatus && jobStatus !== 'completed' && jobStatus !== 'failed';
  const guideText = practiceTarget?.phonetic
    ? (PHONEME_GUIDES[practiceTarget.phonetic] ?? null)
    : null;

  // ── select screen ─────────────────────────────────────────────────────────
  if (mode === 'select') {
    const selectInner = (
      <View style={styles.selectContainer}>
        <Text style={styles.selectTitle}>Luyện phát âm</Text>
        <Text style={styles.selectSubtitle}>Chọn chủ đề hoặc bộ từ để bắt đầu</Text>

        <Text style={styles.sectionLabel}>Chủ đề</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.topicScroll}
          contentContainerStyle={styles.topicChips}
        >
          {TOPICS.map(t => (
            <Pressable
              key={t.value}
              style={[styles.topicChip, selectedTopic === t.value && styles.topicChipSelected]}
              onPress={() => {
                setSelectedTopic(prev => (prev === t.value ? null : t.value));
                setSelectedSetId(null);
              }}
            >
              <Text style={[styles.topicChipText, selectedTopic === t.value && styles.topicChipTextSelected]}>
                {t.label}
              </Text>
            </Pressable>
          ))}
        </ScrollView>

        <Text style={styles.sectionLabel}>Bộ luyện âm</Text>
        {setsLoading ? (
          <ActivityIndicator color={colors.primary} style={styles.setsLoader} />
        ) : sets.length === 0 ? (
          <Text style={styles.emptySets}>Chưa có bộ từ nào.</Text>
        ) : (
          sets.map(set => (
            <Pressable
              key={set.id}
              style={[styles.setCard, selectedSetId === set.id && styles.setCardSelected]}
              onPress={() => {
                setSelectedSetId(prev => (prev === set.id ? null : set.id));
                setSelectedTopic(null);
              }}
            >
              <Text style={[styles.setTitle, selectedSetId === set.id && styles.setTitleSelected]}>
                {set.title}
              </Text>
              {set.description ? <Text style={styles.setDesc}>{set.description}</Text> : null}
              {set.item_count != null ? (
                <Text style={styles.setCount}>{set.item_count} từ</Text>
              ) : null}
            </Pressable>
          ))
        )}

        {error ? (
          <View style={styles.errorWrap}>
            <ErrorState title="Lỗi" message={error} />
          </View>
        ) : null}

        <Pressable
          style={[styles.startPracticeButton, !canStart && styles.startBtnDisabled]}
          disabled={!canStart}
          onPress={handleStartPractice}
        >
          {startLoading ? <ActivityIndicator color="#fff" style={styles.startSpinner} /> : null}
          <Text style={styles.startBtnText}>Bắt đầu luyện tập</Text>
        </Pressable>
      </View>
    );

    if (isDesktop) {
      return (
        <View style={[styles.desktopRoot, { height }]}>
          <ScrollView contentContainerStyle={styles.selectScroll}>{selectInner}</ScrollView>
        </View>
      );
    }
    return (
      <SafeAreaView style={styles.mobileRoot}>
        <ScrollView contentContainerStyle={styles.selectScroll} showsVerticalScrollIndicator={false}>
          {selectInner}
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ── practice screen ───────────────────────────────────────────────────────
  return (
    <SafeAreaView style={styles.practiceRoot}>
      {/* Streak toast overlay — floats above scroll content, pointerEvents="box-none"
          lets taps pass through the container to the ScrollView underneath */}
      {toasts.length > 0 && (
        <View style={styles.toastContainer} pointerEvents="box-none">
          {toasts.map(t => (
            <StreakToast
              key={t.id}
              message={t.message}
              durationMs={TOAST_DURATION_MS}
              onExpire={() => setToasts(prev => prev.filter(x => x.id !== t.id))}
            />
          ))}
        </View>
      )}

      <ScrollView
        contentContainerStyle={styles.practiceScroll}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        {/* Back nav row */}
        <View style={styles.navRow}>
          <Pressable style={styles.backBtn} onPress={handleBackToSelect}>
            <Text style={styles.backBtnText}>← Quay lại</Text>
          </Pressable>
        </View>

        {/* Phrase header */}
        <View style={styles.phraseHeader}>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>MASTER THIS PHRASE</Text>
          </View>
          <Text style={styles.phraseText}>"{practiceTarget?.word ?? ''}"</Text>
        </View>

        {/* Practice area: desktop=row (mic absolute center, guide right), mobile=column */}
        <View style={[styles.practiceArea, !isDesktop && styles.practiceAreaMobile]}>

          {/* Waveform + mic */}
          <View style={[styles.micCol, isDesktop ? styles.micColDesktop : styles.micColMobile]}>
            <View style={styles.waveformAbsolute}>
              <WaveformBars
                heights={WAVEFORM_BARS}
                color={colors.micActive}
                barWidth={8}
                gap={4}
                opacity={recording ? 0.5 : 0.18}
                align="center"
              />
            </View>
            <MicButton
              isRecording={!!recording}
              onPress={handleMicPress}
            />
          </View>

          {/* Word pronunciation guide */}
          <View style={[styles.guideCol, isDesktop ? styles.guideColDesktop : styles.guideColMobile]}>
            {practiceTarget?.phonetic ? (
              <Text style={styles.guideIpa}>{practiceTarget.phonetic}</Text>
            ) : null}
            {practiceTarget?.meaning_vi ? (
              <Text style={styles.guideMeaning}>{practiceTarget.meaning_vi}</Text>
            ) : null}
            {practiceTarget?.sample_sentence ? (
              <Text style={styles.guideSample} numberOfLines={2}>
                {practiceTarget.sample_sentence}
              </Text>
            ) : null}
            {guideText ? (
              <Text style={styles.guideBody} numberOfLines={3}>{guideText}</Text>
            ) : null}
            <Pressable
              style={[
                styles.wordSampleBtn,
                (!wordSampleUrl || wordSampleLoading) && styles.wordSampleBtnDisabled,
              ]}
              onPress={() => void toggleWordSamplePlayback()}
              disabled={!wordSampleUrl || wordSampleLoading}
            >
              {wordSampleLoading ? (
                <ActivityIndicator size="small" color={colors.white} />
              ) : (
                <MaterialIcons
                  name={isPlayingWordSample ? 'pause' : 'volume-up'}
                  size={16}
                  color={colors.white}
                />
              )}
              <Text style={styles.wordSampleBtnText} numberOfLines={1}>
                {wordSampleUrl ? 'Phát âm chuẩn' : 'Chưa có mẫu'}
              </Text>
            </Pressable>
            {wordSampleUrl ? (
              <Text style={styles.guideAttr}>Free Dictionary API, CC BY-SA 3.0</Text>
            ) : null}
          </View>
        </View>

        {/* Timer — visible only while recording */}
        {recording && (
          <View style={[styles.recordingPill, { alignSelf: 'center', marginBottom: spacing.md }]}>
            <View style={styles.recDot} />
            <Text style={styles.recTimerText}>{mm}:{ss}</Text>
          </View>
        )}

        {/* Status hints */}
        {!recording && !audioUri && !isPolling && (
          <Text style={styles.hintText}>Chạm micro để bắt đầu ghi âm</Text>
        )}
        {!recording && audioUri && !isPolling && (
          <Text style={styles.readyText}>Bản ghi đã sẵn sàng ✓</Text>
        )}

        {/* AI scoring indicator */}
        {isPolling && (
          <View style={styles.processingRow}>
            <ActivityIndicator color={colors.primary} size="small" />
            <Text style={styles.processingText}>AI đang chấm điểm...</Text>
          </View>
        )}

        {/* Failed status */}
        {jobStatus === 'failed' && (
          job?.feedback?.error_type === 'audio_quality_rejected' ? (
            <View style={styles.audioWarningBox}>
              <MaterialIcons name="warning" size={22} color={colors.warning} />
              <View style={styles.audioWarningContent}>
                <Text style={styles.audioWarningTitle}>Âm thanh không rõ</Text>
                <Text style={styles.audioWarningText}>
                  Âm thanh không rõ, vui lòng thu âm lại ở nơi yên tĩnh hơn.
                </Text>
              </View>
            </View>
          ) : (
            <View style={styles.processingRow}>
              <Text style={styles.failedText}>Chấm điểm thất bại. Vui lòng thử lại.</Text>
            </View>
          )
        )}

        {/* Error */}
        {error ? (
          <View style={styles.errorWrap}>
            <ErrorState title="Cần kiểm tra lại" message={error} />
          </View>
        ) : null}

        {/* Replay + Gửi | divider | Bỏ qua — centered as one group */}
        <View style={styles.circleRow}>
          <View style={styles.circleMainBtns}>
            <CircleIconButton
              icon={<MaterialIcons name="play-arrow" size={30} color={colors.white} />}
              label="Replay"
              bgColor={colors.secondaryContainer}
              onPress={playRecording}
              disabled={!audioUri || !!recording}
            />
            <CircleIconButton
              icon={<MaterialIcons name="send" size={28} color={colors.white} />}
              label="Gửi"
              bgColor={colors.secondaryContainer}
              onPress={() => handleCreateJob()}
              disabled={!audioUri || loading || !!recording}
            />
          </View>
          <View style={styles.circleDividerArea}>
            <View style={styles.circleDivider} />
          </View>
          <CircleIconButton
            icon={<MaterialIcons name="skip-next" size={28} color={colors.onSurfaceVariant} />}
            label="Bỏ qua"
            bgColor={colors.surfaceContainerHigh}
            onPress={handleSkipWord}
            disabled={loading}
          />
        </View>

        {/* Teacher Tip */}
        <TipCard
          variant="dashed"
          icon={<MaterialIcons name="auto-awesome" size={24} color={colors.primary} />}
          title="Teacher Tip"
          description="Chạm micro để thu âm, chạm lần nữa để dừng. Bấm nút Gửi để AI chấm điểm."
        />


      </ScrollView>
    </SafeAreaView>
  );
}

const PHONEME_GUIDES: Record<string, string> = {
  // Vowels
  '/iː/':  'Kéo dài âm "i" như trong "see". Môi mỉm cười nhẹ, lưỡi chạm vào hàm ếch trước.',
  '/ɪ/':   'Âm "i" ngắn, thư giãn như trong "sit". Môi không cần mỉm cười, lưỡi ở giữa.',
  '/e/':   'Âm "e" như trong "bed". Miệng mở vừa phải, lưỡi ở vị trí giữa-trước.',
  '/æ/':   'Âm "a" mở như trong "cat". Miệng mở rộng, hàm hạ thấp, lưỡi phẳng ở trước.',
  '/ɑː/':  'Âm "a" dài như trong "father". Hàm mở rộng, lưỡi nằm phẳng phía sau miệng.',
  '/ɒ/':   'Âm "o" tròn ngắn như trong "hot". Môi tròn nhẹ, hàm hạ thấp.',
  '/ɔː/':  'Âm "o" dài như trong "law". Môi tròn và nhô ra trước, lưỡi ở phía sau.',
  '/ʊ/':   'Âm "u" ngắn như trong "book". Môi tròn nhẹ, lưỡi nâng phía sau.',
  '/uː/':  'Âm "u" dài như trong "food". Môi tròn và chặt, lưỡi nâng cao phía sau.',
  '/ʌ/':   'Âm giống "ă" như trong "cup". Miệng mở vừa, lưỡi ở vị trí giữa thấp.',
  '/ɜː/':  'Âm "ơ" dài như trong "bird". Môi thư giãn, lưỡi ở chính giữa miệng.',
  '/ə/':   'Âm trung hòa (schwa) như âm đệm trong "about". Miệng thư giãn hoàn toàn.',
  // Diphthongs
  '/eɪ/':  'Trượt từ "e" sang "i" như trong "day". Bắt đầu mở miệng rồi thu hẹp dần.',
  '/aɪ/':  'Trượt từ "a" sang "i" như trong "my". Hàm hạ xuống rồi nâng lên.',
  '/ɔɪ/':  'Trượt từ "o" sang "i" như trong "boy". Môi tròn rồi mỉm cười.',
  '/aʊ/':  'Trượt từ "a" sang "u" như trong "now". Miệng mở rồi thu tròn lại.',
  '/əʊ/':  'Trượt từ schwa sang "u" như trong "go". Bắt đầu thư giãn rồi thu tròn môi.',
  '/ɪə/':  'Trượt từ "i" sang schwa như trong "near".',
  '/eə/':  'Trượt từ "e" sang schwa như trong "hair".',
  '/ʊə/':  'Trượt từ "u" sang schwa như trong "pure".',
  // Consonants — stops
  '/p/':   'Âm bật hơi không có tiếng. Khép môi lại, giữ hơi, rồi bật ra mạnh.',
  '/b/':   'Giống /p/ nhưng có tiếng. Khép môi, rung dây thanh, rồi bật ra.',
  '/t/':   'Đặt đầu lưỡi sau răng cửa trên, giữ hơi rồi bật ra. Không có tiếng.',
  '/d/':   'Giống /t/ nhưng rung dây thanh khi phát âm.',
  '/k/':   'Gốc lưỡi chạm vào vòm miệng mềm phía sau, giữ hơi rồi bật ra.',
  '/ɡ/':   'Giống /k/ nhưng rung dây thanh. Gốc lưỡi chạm vòm mềm.',
  // Consonants — fricatives
  '/f/':   'Răng trên chạm nhẹ môi dưới, thổi hơi qua khe hẹp. Không có tiếng.',
  '/v/':   'Giống /f/ nhưng rung dây thanh khi thổi hơi.',
  '/θ/':   'Đặt đầu lưỡi nhẹ giữa hai hàm răng, thổi hơi qua. Không rung (như "think").',
  '/ð/':   'Giống /θ/ nhưng rung dây thanh (như "this"). Lưỡi nhẹ giữa răng.',
  '/s/':   'Lưỡi gần vòm trước, thổi hơi qua khe giữa răng. Không có tiếng.',
  '/z/':   'Giống /s/ nhưng rung dây thanh.',
  '/ʃ/':   'Môi hơi tròn, lưỡi nâng lên, thổi hơi rộng hơn /s/. Như "sh" trong "ship".',
  '/ʒ/':   'Giống /ʃ/ nhưng rung dây thanh. Như "s" trong "measure".',
  '/h/':   'Thổi hơi tự do qua thanh quản mở. Không rung dây thanh.',
  // Consonants — affricates
  '/tʃ/':  'Bắt đầu như /t/ rồi trượt sang /ʃ/. Như "ch" trong "church".',
  '/dʒ/':  'Bắt đầu như /d/ rồi trượt sang /ʒ/. Như "j" trong "jump".',
  // Consonants — nasals
  '/m/':   'Khép môi, rung dây thanh và thở qua mũi. Như âm "m" trong tiếng Việt.',
  '/n/':   'Đầu lưỡi chạm vòm sau răng cửa trên, thở qua mũi.',
  '/ŋ/':   'Gốc lưỡi chạm vòm mềm, thở qua mũi. Như "ng" cuối từ trong tiếng Việt.',
  // Consonants — approximants
  '/l/':   'Đầu lưỡi chạm vòm sau răng cửa trên, hơi thoát ra hai bên. Rung dây thanh.',
  '/r/':   'Lưỡi uốn nhẹ ra sau, không chạm vòm. Môi hơi tròn. Không rung như /r/ tiếng Việt.',
  '/j/':   'Giống "y" tiếng Việt. Lưỡi nâng lên hướng về vòm trước. Như "y" trong "yes".',
  '/w/':   'Môi tròn chặt rồi nhanh chóng mở ra. Như "w" trong "water".',
};


const styles = StyleSheet.create({
  // ── shared roots ──────────────────────────────────────────────────────────
  desktopRoot: {
    flex: 1,
    backgroundColor: '#faf8ff',
    overflow: 'hidden',
  },
  mobileRoot: {
    flex: 1,
    backgroundColor: '#faf8ff',
  },

  // ── select screen ─────────────────────────────────────────────────────────
  selectScroll: {
    flexGrow: 1,
    alignItems: 'center',
  },
  selectContainer: {
    width: '100%',
    maxWidth: 600,
    paddingHorizontal: 20,
    paddingTop: 32,
    paddingBottom: 40,
  },
  selectTitle: {
    color: '#191b23',
    fontSize: 28,
    fontWeight: '900',
    marginBottom: 6,
  },
  selectSubtitle: {
    color: '#737686',
    fontSize: 14,
    lineHeight: 21,
    marginBottom: 28,
  },
  sectionLabel: {
    color: '#434655',
    fontSize: 13,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 10,
  },
  topicScroll: {
    marginBottom: 28,
  },
  topicChips: {
    flexDirection: 'row',
    gap: 8,
    paddingRight: 20,
  },
  topicChip: {
    borderRadius: 999,
    borderWidth: 1.5,
    borderColor: '#c3c6d7',
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: '#ffffff',
  },
  topicChipSelected: {
    backgroundColor: '#004ac6',
    borderColor: '#004ac6',
  },
  topicChipText: {
    color: '#434655',
    fontSize: 13,
    fontWeight: '600',
  },
  topicChipTextSelected: {
    color: '#ffffff',
  },
  setsLoader: {
    marginTop: 12,
    marginBottom: 28,
  },
  emptySets: {
    color: '#737686',
    fontSize: 14,
    marginBottom: 28,
  },
  setCard: {
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: '#e7e7f3',
    padding: 16,
    marginBottom: 10,
    backgroundColor: '#ffffff',
  },
  setCardSelected: {
    borderColor: '#004ac6',
    backgroundColor: '#eef2ff',
  },
  setTitle: {
    color: '#191b23',
    fontSize: 16,
    fontWeight: '700',
  },
  setTitleSelected: {
    color: '#004ac6',
  },
  setDesc: {
    color: '#737686',
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },
  setCount: {
    color: '#737686',
    fontSize: 12,
    marginTop: 6,
    fontWeight: '500',
  },
  startPracticeButton: {
    marginTop: 24,
    minHeight: 56,
    borderRadius: 12,
    backgroundColor: '#004ac6',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    shadowColor: '#004ac6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 5,
  },
  startBtnDisabled: {
    opacity: 0.72,
    shadowOpacity: 0,
    elevation: 0,
  },
  startSpinner: {
    marginRight: 8,
  },
  startBtnText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
  },

  // ── practice screen ───────────────────────────────────────────────────────
  practiceRoot: {
    flex: 1,
    backgroundColor: colors.background,
  },
  practiceScroll: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.xl,
  },

  // Toast container floats at the very top of the SafeAreaView
  toastContainer: {
    position: 'absolute',
    top: spacing.md,
    left: spacing.lg,
    right: spacing.lg,
    zIndex: 100,
    gap: spacing.sm,
  },

  navRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
  },
  backBtn: {
    backgroundColor: colors.surfaceContainerHigh,
    borderRadius: radius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
  },
  backBtnText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.onSurfaceVariant,
  },
  recordingPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.errorContainer,
    borderRadius: radius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
  },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.error,
  },
  recTimerText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.error,
  },

  phraseHeader: {
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  badge: {
    backgroundColor: `${colors.secondaryContainer}33`,
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.secondary,
    textTransform: 'uppercase',
    letterSpacing: 2,
  },
  phraseText: {
    ...typography.h1,
    color: colors.onSurface,
    textAlign: 'center',
  },

  // ── practice area ─────────────────────────────────────────────────────────
  // Desktop: row — mic absolute-centered over full width, guide floated right
  // Mobile:  column — mic first (full-width), guide stacked below
  practiceArea: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 240,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  practiceAreaMobile: {
    flexDirection: 'column',
    alignItems: 'stretch',
    minHeight: 0,
  },
  micCol: {
    height: 240,
    alignItems: 'center',
    justifyContent: 'center',
  },
  micColDesktop: {
    position: 'absolute',
    left: 0,
    right: 0,
  },
  micColMobile: {
    width: '100%',
  },
  waveformAbsolute: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Shared card shell
  guideCol: {
    justifyContent: 'center',
    backgroundColor: colors.surfaceLowest,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.slate100,
    padding: spacing.md,
    gap: spacing.sm,
  },
  // Desktop: right-edge card ~250px
  guideColDesktop: {
    marginLeft: 'auto',
    width: 250,
    alignSelf: 'center',
  },
  // Mobile: full-width below mic
  guideColMobile: {
    alignSelf: 'stretch',
    marginTop: spacing.md,
  },
  guideIpa: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.outline,
    letterSpacing: 0.5,
  },
  guideMeaning: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.onSurface,
    lineHeight: 18,
  },
  guideSample: {
    fontSize: 12,
    color: colors.onSurfaceVariant,
    fontStyle: 'italic',
    lineHeight: 19,
  },
  guideBody: {
    fontSize: 12,
    color: colors.onSurfaceVariant,
    lineHeight: 19,
  },
  wordSampleBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingHorizontal: spacing.sm,
    height: 40,
  },
  wordSampleBtnDisabled: {
    opacity: 0.45,
  },
  wordSampleBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.white,
  },
  guideAttr: {
    fontSize: 10,
    color: colors.slate400,
    fontStyle: 'italic',
  },

  hintText: {
    fontSize: 14,
    color: colors.onSurfaceVariant,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  readyText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.success,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },

  processingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    marginBottom: spacing.sm,
  },
  processingText: {
    fontSize: 14,
    color: colors.onSurfaceVariant,
  },
  failedText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.error,
    textAlign: 'center',
  },

  audioWarningBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: colors.warningLight,
    borderRadius: radius.lg,
    borderLeftWidth: 4,
    borderLeftColor: colors.warning,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  audioWarningContent: {
    flex: 1,
    gap: 4,
  },
  audioWarningTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.tertiary,
  },
  audioWarningText: {
    fontSize: 13,
    color: colors.onTertiaryFixedVariant,
    lineHeight: 19,
  },

  errorWrap: {
    marginBottom: spacing.md,
  },

  // Replay + Gửi | divider | Bỏ qua — centered as one group
  circleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  circleMainBtns: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  circleDividerArea: {
    paddingHorizontal: 24,
    alignSelf: 'stretch',
    justifyContent: 'center',
    alignItems: 'center',
  },
  circleDivider: {
    width: 1,
    height: 36,
    backgroundColor: colors.outlineVariant,
  },

});
