import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
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
  fetchVocabularyItems,
  fetchVocabularySets,
  fetchVocabularySetDetail,
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

  // ── select-screen state ──────────────────────────────────────────────────
  const [mode, setMode] = useState<'select' | 'practice'>('select');
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [selectedSetId, setSelectedSetId] = useState<string | null>(null);
  const [sets, setSets] = useState<VocabularySet[]>([]);
  const [setsLoading, setSetsLoading] = useState(true);
  const [startLoading, setStartLoading] = useState(false);

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

  // ── phoneme modal state ──────────────────────────────────────────────────
  const [phonemeModalVisible, setPhonemeModalVisible] = useState(false);

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

  useEffect(() => {
    return () => {
      if (sound) void sound.unloadAsync();
    };
  }, [sound]);

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

    // Update streak using a ref to always have the latest value.
    const score = job.score ?? 0;
    const newStreak = score > 80 ? streakRef.current + 1 : 0;
    streakRef.current = newStreak;

    const doNavigate = () => {
      router.push({
        pathname: '/practice/result' as any,
        params: {
          score: String(job.score ?? 0),
          word: practiceTarget.word,
          phonetic: practiceTarget.phonetic ?? '',
          audio_url: job.audio_url ?? '',
          problem_phonemes: JSON.stringify(job.problem_phonemes ?? []),
          feedback: JSON.stringify(job.feedback ?? null),
        },
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

      if (selectedSetId) {
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
    if (timerRef.current)    { clearInterval(timerRef.current);  timerRef.current = null; }
    if (pollingRef.current)  { clearInterval(pollingRef.current); pollingRef.current = null; }
    if (navTimerRef.current) { clearTimeout(navTimerRef.current); navTimerRef.current = null; }
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
          {recording && (
            <View style={styles.recordingPill}>
              <View style={styles.recDot} />
              <Text style={styles.recTimerText}>{mm}:{ss}</Text>
            </View>
          )}
        </View>

        {/* Phrase header */}
        <View style={styles.phraseHeader}>
          <View style={styles.badge}>
            <Text style={styles.badgeText}>MASTER THIS PHRASE</Text>
          </View>
          <Text style={styles.phraseText}>"{practiceTarget?.word ?? ''}"</Text>
          {practiceTarget?.phonetic ? (
            <View style={styles.ipaRow}>
              <Pressable
                accessibilityRole="button"
                onPress={() => setPhonemeModalVisible(true)}
                style={styles.ipaPill}
              >
                <Text style={styles.ipaText}>{practiceTarget.phonetic}</Text>
                <MaterialIcons name="info-outline" size={14} color={colors.outline} style={{ marginLeft: 4 }} />
              </Pressable>
            </View>
          ) : null}
        </View>

        {/* Central mic module — waveform behind MicButton */}
        <View style={styles.micModule}>
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

        {/* Status hint below mic */}
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
          <View style={styles.processingRow}>
            <Text style={styles.failedText}>Chấm điểm thất bại. Vui lòng thử lại.</Text>
          </View>
        )}

        {/* Error */}
        {error ? (
          <View style={styles.errorWrap}>
            <ErrorState title="Cần kiểm tra lại" message={error} />
          </View>
        ) : null}

        {/* Replay + Submit row */}
        <View style={styles.circleRow}>
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

        {/* Teacher Tip */}
        <TipCard
          variant="dashed"
          icon={<MaterialIcons name="auto-awesome" size={24} color={colors.primary} />}
          title="Teacher Tip"
          description="Chạm micro để thu âm, chạm lần nữa để dừng. Bấm nút Gửi để AI chấm điểm."
        />

        {/* Skip link */}
        <Pressable style={styles.skipLink} onPress={handleBackToSelect}>
          <Text style={styles.skipLinkText}>Bỏ qua từ này</Text>
        </Pressable>
      </ScrollView>
      <PhonemeDetailModal
        visible={phonemeModalVisible}
        phoneme={practiceTarget?.phonetic ?? null}
        word={practiceTarget?.word ?? null}
        onClose={() => setPhonemeModalVisible(false)}
      />
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

function PhonemeDetailModal({
  visible,
  phoneme,
  word,
  onClose,
}: {
  visible: boolean;
  phoneme: string | null;
  word: string | null;
  onClose: () => void;
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const soundRef = useRef<Audio.Sound | null>(null);

  useEffect(() => {
    return () => { void soundRef.current?.unloadAsync(); };
  }, []);

  const guideText = phoneme ? (PHONEME_GUIDES[phoneme] ?? `Chưa có hướng dẫn chi tiết cho âm ${phoneme}.`) : '';

  const playReference = async (url: string) => {
    if (isPlaying) {
      await soundRef.current?.pauseAsync();
      setIsPlaying(false);
      return;
    }
    try {
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
      if (!soundRef.current) {
        const { sound } = await Audio.Sound.createAsync({ uri: url }, {}, (s) => {
          if (s.isLoaded && s.didJustFinish) setIsPlaying(false);
        });
        soundRef.current = sound;
      }
      await soundRef.current.playAsync();
      setIsPlaying(true);
    } catch { setIsPlaying(false); }
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable style={phonemeModalStyles.backdrop} onPress={onClose}>
        <Pressable style={phonemeModalStyles.sheet} onPress={() => undefined}>
          {/* Handle */}
          <View style={phonemeModalStyles.handle} />

          {/* Header */}
          <View style={phonemeModalStyles.header}>
            <View>
              {word ? <Text style={phonemeModalStyles.wordLabel}>{word}</Text> : null}
              {phoneme ? <Text style={phonemeModalStyles.phonemeLabel}>{phoneme}</Text> : null}
            </View>
            <Pressable accessibilityRole="button" onPress={onClose} style={phonemeModalStyles.closeBtn}>
              <MaterialIcons name="close" size={20} color={colors.onSurfaceVariant} />
            </Pressable>
          </View>

          {/* Guide section */}
          <View style={phonemeModalStyles.section}>
            <Text style={phonemeModalStyles.sectionTitle}>HƯỚNG DẪN PHÁT ÂM</Text>
            <Text style={phonemeModalStyles.guideText}>{guideText}</Text>
          </View>

          {/* Reference audio section */}
          <View style={phonemeModalStyles.section}>
            <Text style={phonemeModalStyles.sectionTitle}>ÂM MẪU</Text>
            <View style={phonemeModalStyles.audioRow}>
              <Pressable
                style={[phonemeModalStyles.playBtn, phonemeModalStyles.playBtnDisabled]}
                disabled
              >
                <MaterialIcons name="play-arrow" size={22} color="#ffffff" />
              </Pressable>
              <View style={phonemeModalStyles.audioMeta}>
                <Text style={phonemeModalStyles.audioLabel}>Phát âm chuẩn</Text>
                <Text style={phonemeModalStyles.audioNote}>Âm mẫu chưa khả dụng — sẽ cập nhật sau</Text>
              </View>
            </View>
          </View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const phonemeModalStyles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: colors.surfaceLowest,
    borderTopLeftRadius: radius.xxl,
    borderTopRightRadius: radius.xxl,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl,
    paddingTop: spacing.sm,
    gap: spacing.md,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.outlineVariant,
    alignSelf: 'center',
    marginBottom: spacing.xs,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
  },
  wordLabel: {
    fontSize: 22,
    fontWeight: '700',
    color: colors.onSurface,
  },
  phonemeLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: colors.outline,
    marginTop: 2,
  },
  closeBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.surfaceContainerHigh,
    alignItems: 'center',
    justifyContent: 'center',
  },
  section: {
    gap: spacing.xs,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.slate400,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  guideText: {
    fontSize: 15,
    lineHeight: 24,
    color: colors.onSurface,
  },
  audioRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.surfaceContainerHigh,
    borderRadius: radius.lg,
    padding: spacing.sm,
  },
  playBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  playBtnDisabled: {
    opacity: 0.4,
  },
  audioMeta: {
    flex: 1,
    gap: 2,
  },
  audioLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.onSurface,
  },
  audioNote: {
    fontSize: 12,
    color: colors.outlineVariant,
  },
});

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
    marginBottom: spacing.md,
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
  ipaRow: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  ipaPill: {
    backgroundColor: colors.surfaceContainerHigh,
    borderRadius: radius.md,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
  },
  ipaText: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.outline,
  },

  micModule: {
    width: '100%',
    height: 260,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: spacing.xl,
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

  errorWrap: {
    marginBottom: spacing.md,
  },

  // Replay + Submit side by side, centred
  circleRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.xxl,
    marginBottom: spacing.lg,
  },

  skipLink: {
    alignSelf: 'center',
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
  },
  skipLinkText: {
    fontSize: 13,
    color: colors.onSurfaceVariant,
    textDecorationLine: 'underline',
    textAlign: 'center',
  },
});
