import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { Redirect, useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useMemo, useRef, useState } from 'react';
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
import { AppButton, ErrorState, LoadingState, colors } from '../../../components/AppUI';
import { createPracticeJob, fetchPracticeStatus, uploadPracticeAudio } from '../../../lib/api';
import { useAuth } from '../../../lib/auth';
import {
  formatFeedbackLines,
  formatProblemPhonemes,
  getScoreTone,
} from '../../../lib/practiceFormatters';
import type { PracticeJob, PracticeJobStatus } from '../../../types';

type MistakeId =
  | 'final-sounds'
  | 'word-stress'
  | 'vowel-length'
  | 'consonant-clusters'
  | 'linking-sounds';

type PracticeTarget = {
  word: string;
  ipa: string;
  meaning: string;
  highlightLabel: string;
  highlight: string;
  parts: string[];
};

type MistakePracticeConfig = {
  title: string;
  subtitle: string;
  badge: string;
  coachTitle: string;
  coachTip: string;
  featureLabel: string;
  drillTitle: string;
  drillSteps: string[];
  accent: string;
  softAccent: string;
  targets: PracticeTarget[];
};

const waveformBars = [14, 24, 38, 28, 48, 36, 30, 18, 34, 22];

const practiceConfigs: Record<MistakeId, MistakePracticeConfig> = {
  'final-sounds': {
    title: 'Luyện Âm cuối',
    subtitle: 'Giữ âm đuôi rõ ràng để từ không bị mất nghĩa.',
    badge: 'Phụ âm cuối',
    coachTitle: 'AI Coach Tip',
    coachTip: 'Đừng kết thúc từ quá sớm. Bật nhẹ âm cuối, ngắn và rõ, nhưng không thêm nguyên âm phía sau.',
    featureLabel: 'Âm cuối mục tiêu',
    drillTitle: 'Tách thân từ và âm cuối',
    drillSteps: ['Đọc phần thân từ chậm rãi.', 'Dừng rất ngắn trước âm cuối.', 'Bật âm cuối gọn, không kéo dài.'],
    accent: '#00796B',
    softAccent: '#E0F7F4',
    targets: [
      {
        word: 'cat',
        ipa: '/kæt/',
        meaning: 'Con mèo',
        highlightLabel: 'Âm /t/ cuối từ',
        highlight: 't',
        parts: ['ca', 't'],
      },
      {
        word: 'five',
        ipa: '/faɪv/',
        meaning: 'Số năm',
        highlightLabel: 'Âm /v/ cuối từ',
        highlight: 'v',
        parts: ['fi', 've'],
      },
    ],
  },
  'word-stress': {
    title: 'Luyện Trọng âm từ',
    subtitle: 'Nhấn đúng âm tiết để từ dài nghe tự nhiên hơn.',
    badge: 'Trọng âm từ',
    coachTitle: 'AI Coach',
    coachTip: 'Hãy đọc âm tiết được nhấn to hơn, dài hơn và cao hơn các âm tiết còn lại.',
    featureLabel: 'Âm tiết nhấn mạnh',
    drillTitle: 'Cấu trúc nhịp điệu',
    drillSteps: ['Đọc từng âm tiết đều nhau.', 'Nhấn mạnh âm tiết mục tiêu.', 'Ghép lại thành một từ liền mạch.'],
    accent: '#0052CC',
    softAccent: '#E8F0FF',
    targets: [
      {
        word: 'important',
        ipa: '/ɪmˈpɔːr.tənt/',
        meaning: 'Quan trọng',
        highlightLabel: 'Nhấn âm "por"',
        highlight: 'por',
        parts: ['im', 'por', 'tant'],
      },
      {
        word: 'photograph',
        ipa: '/ˈfoʊ.tə.ɡræf/',
        meaning: 'Bức ảnh',
        highlightLabel: 'Nhấn âm "pho"',
        highlight: 'pho',
        parts: ['pho', 'to', 'graph'],
      },
    ],
  },
  'vowel-length': {
    title: 'Luyện Độ dài nguyên âm',
    subtitle: 'Phân biệt nguyên âm ngắn và dài trong các cặp tối thiểu.',
    badge: 'Nguyên âm dài/ngắn',
    coachTitle: 'Mẹo từ AI Coach',
    coachTip: 'Âm dài cần được giữ lâu hơn và miệng căng hơn. Hãy kéo dài âm lượng khoảng gấp đôi so với âm ngắn.',
    featureLabel: 'Độ dài mục tiêu',
    drillTitle: 'So sánh cặp tối thiểu',
    drillSteps: ['Đọc từ ngắn trước để cảm nhận nhịp.', 'Giữ nguyên âm dài lâu hơn.', 'So sánh lại hai từ và tránh đổi phụ âm cuối.'],
    accent: '#14B8A6',
    softAccent: '#DDFCF7',
    targets: [
      {
        word: 'sheep',
        ipa: '/ʃiːp/',
        meaning: 'Con cừu',
        highlightLabel: 'Âm /iː/ dài',
        highlight: 'ee',
        parts: ['sh', 'ee', 'p'],
      },
      {
        word: 'ship',
        ipa: '/ʃɪp/',
        meaning: 'Con tàu',
        highlightLabel: 'Âm /ɪ/ ngắn',
        highlight: 'i',
        parts: ['sh', 'i', 'p'],
      },
    ],
  },
  'consonant-clusters': {
    title: 'Luyện Cụm phụ âm',
    subtitle: 'Nối nhiều phụ âm liền nhau mà không chèn thêm nguyên âm.',
    badge: 'Cụm phụ âm',
    coachTitle: 'AI Coach Tip',
    coachTip: 'Hãy nối /s/, /t/ và /r/ thật sát nhau. Không thêm âm "ơ" giữa các phụ âm.',
    featureLabel: 'Cụm âm mục tiêu',
    drillTitle: 'Xây dựng âm',
    drillSteps: ['Bắt đầu với âm đầu tiên.', 'Thêm từng phụ âm vào cụm.', 'Đọc cả từ với cụm phụ âm liền mạch.'],
    accent: '#006B5F',
    softAccent: '#CCFBF1',
    targets: [
      {
        word: 'street',
        ipa: '/striːt/',
        meaning: 'Con đường',
        highlightLabel: 'Cụm /str/',
        highlight: 'str',
        parts: ['s', 'st', 'str', 'street'],
      },
      {
        word: 'crisps',
        ipa: '/krɪsps/',
        meaning: 'Khoai tây chiên giòn',
        highlightLabel: 'Cụm /sps/',
        highlight: 'sps',
        parts: ['kri', 'sp', 'sps'],
      },
    ],
  },
  'linking-sounds': {
    title: 'Luyện Nối âm',
    subtitle: 'Liên kết các từ để câu nói trôi chảy và tự nhiên.',
    badge: 'Connected speech',
    coachTitle: 'Mẹo từ AI Coach',
    coachTip: 'Hãy nối phụ âm cuối của từ trước với nguyên âm đầu của từ sau, đọc như một cụm liền mạch.',
    featureLabel: 'Điểm nối âm',
    drillTitle: 'Nối hai từ thành một cụm',
    drillSteps: ['Đọc riêng từng từ.', 'Giữ phụ âm cuối của từ trước.', 'Bắt ngay vào nguyên âm đầu của từ sau.'],
    accent: '#00796B',
    softAccent: '#DDF7F4',
    targets: [
      {
        word: 'pick it up',
        ipa: '/pɪk ɪt ʌp/',
        meaning: 'Nhặt nó lên',
        highlightLabel: 'Nối /k/ với /ɪ/',
        highlight: 'it',
        parts: ['pick', 'it', 'up'],
      },
      {
        word: 'turn off',
        ipa: '/tɝːn ɔːf/',
        meaning: 'Tắt đi',
        highlightLabel: 'Nối /n/ với /ɔː/',
        highlight: 'off',
        parts: ['turn', 'off'],
      },
    ],
  },
};

function isMistakeId(value: string | string[] | undefined): value is MistakeId {
  return typeof value === 'string' && value in practiceConfigs;
}

export default function MistakePracticeScreen() {
  const router = useRouter();
  const { mistakeId } = useLocalSearchParams();
  const { width } = useWindowDimensions();
  const { accessToken, loading: authLoading, session } = useAuth();
  const [targetIndex, setTargetIndex] = useState(0);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<PracticeJob | null>(null);
  const [jobStatus, setJobStatus] = useState<PracticeJobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const config = isMistakeId(mistakeId) ? practiceConfigs[mistakeId] : null;
  const target = config?.targets[targetIndex] ?? null;
  const isDesktop = width >= 900;
  const resultLines = useMemo(() => formatFeedbackLines(job?.feedback), [job?.feedback]);
  const problemLines = useMemo(() => formatProblemPhonemes(job?.problem_phonemes), [job?.problem_phonemes]);
  const scoreTone = getScoreTone(job?.score);

  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (sound) {
        void sound.unloadAsync();
      }
    };
  }, [sound]);

  useEffect(() => {
    setAudioUri(null);
    setJobId(null);
    setJob(null);
    setJobStatus(null);
    setError(null);
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }
  }, [targetIndex, mistakeId]);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }

    pollingRef.current = setInterval(async () => {
      try {
        const response = await fetchPracticeStatus(jobId, accessToken);
        setJob(response);
        setJobStatus(response.status);
        if (response.status === 'completed' || response.status === 'failed') {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không lấy được kết quả chấm.');
      }
    }, 3000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [jobId, accessToken]);

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

    const { recording: nextRecording } = await Audio.Recording.createAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY,
    );
    setRecording(nextRecording);
  };

  const stopRecording = async () => {
    if (!recording) {
      return;
    }
    await recording.stopAndUnloadAsync();
    setAudioUri(recording.getURI() ?? null);
    setRecording(null);
  };

  const playRecording = async () => {
    if (!audioUri) {
      return;
    }

    setError(null);
    if (sound) {
      await sound.replayAsync();
      return;
    }

    const { sound: nextSound } = await Audio.Sound.createAsync({ uri: audioUri });
    setSound(nextSound);
    await nextSound.playAsync();
  };

  const submitToAi = async () => {
    if (!target) {
      return;
    }
    if (!audioUri) {
      setError('Hãy ghi âm trước khi gửi AI chấm.');
      return;
    }
    if (!accessToken) {
      setError('Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const uploadResponse = await uploadPracticeAudio(audioUri, accessToken);
      const response = await createPracticeJob(
        {
          target_word: target.word,
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
      setSubmitting(false);
    }
  };

  const retry = () => {
    setAudioUri(null);
    setJobId(null);
    setJob(null);
    setJobStatus(null);
    setError(null);
  };

  const nextTarget = () => {
    if (!config) {
      return;
    }
    setTargetIndex((current) => (current + 1) % config.targets.length);
  };

  if (authLoading) {
    return (
      <View style={styles.centeredRoot}>
        <LoadingState title="Đang kiểm tra phiên đăng nhập" message="Vui lòng chờ trong giây lát." />
      </View>
    );
  }

  if (!session) {
    return <Redirect href="/welcome" />;
  }

  if (!config || !target) {
    return (
      <SafeAreaView style={styles.root}>
        <View style={styles.centeredRoot}>
          <ErrorState title="Không tìm thấy bài luyện" message="Bài luyện này chưa có trong thư viện lỗi." />
          <AppButton title="Quay lại Thư viện lỗi" onPress={() => router.push('/(tabs)/mistakes')} />
        </View>
      </SafeAreaView>
    );
  }

  const recordLabel = recording
    ? 'Đang thu âm... nhấn để dừng'
    : audioUri
      ? 'Bản ghi đã sẵn sàng'
      : 'Nhấn để thu âm';

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        <View style={styles.container}>
          <View style={styles.topBar}>
            <Pressable style={styles.backButton} onPress={() => router.push('/(tabs)/mistakes')}>
              <MaterialCommunityIcons name="arrow-left" size={20} color={colors.text} />
              <Text style={styles.backText}>Thư viện lỗi</Text>
            </Pressable>
            <View style={styles.lessonPill}>
              <Text style={styles.lessonPillText}>Bài {targetIndex + 1}/{config.targets.length}</Text>
            </View>
          </View>

          <View style={styles.header}>
            <Text style={[styles.badge, { backgroundColor: config.softAccent, color: config.accent }]}>
              {config.badge}
            </Text>
            <Text style={styles.title}>{config.title}</Text>
            <Text style={styles.subtitle}>{config.subtitle}</Text>
          </View>

          <View style={[styles.grid, isDesktop ? styles.desktopGrid : null]}>
            <View style={styles.leftColumn}>
              <View style={styles.targetCard}>
                <Text style={[styles.targetBadge, { backgroundColor: config.softAccent, color: config.accent }]}>
                  Từ vựng mục tiêu
                </Text>
                <Text style={styles.targetWord}>{target.word}</Text>
                <View style={styles.ipaRow}>
                  <Text style={[styles.ipa, { color: config.accent }]}>{target.ipa}</Text>
                  <Text style={styles.dot}>•</Text>
                  <Text style={styles.meaning}>{target.meaning}</Text>
                </View>
                <View style={styles.featureStrip}>
                  <Text style={styles.featureLabel}>{config.featureLabel}</Text>
                  <Text style={[styles.featureText, { color: config.accent }]}>{target.highlightLabel}</Text>
                </View>
              </View>

              <View style={styles.drillCard}>
                <View style={styles.cardHeaderRow}>
                  <MaterialCommunityIcons name="format-list-numbered" size={24} color={config.accent} />
                  <Text style={styles.cardTitle}>{config.drillTitle}</Text>
                </View>
                <View style={styles.partsRow}>
                  {target.parts.map((part, index) => {
                    const normalizedHighlight = target.highlight.toLowerCase().replace('_', '');
                    const isHighlighted = part.toLowerCase().includes(normalizedHighlight);
                    return (
                      <View key={`${part}-${index}`} style={styles.partWrap}>
                        <View
                          style={[
                            styles.partBubble,
                            isHighlighted ? { backgroundColor: config.softAccent, borderColor: config.accent } : null,
                          ]}
                        >
                          <Text style={[styles.partText, isHighlighted ? { color: config.accent } : null]}>
                            {part}
                          </Text>
                        </View>
                        {index < target.parts.length - 1 ? <Text style={styles.plus}>+</Text> : null}
                      </View>
                    );
                  })}
                </View>
                <View style={styles.stepList}>
                  {config.drillSteps.map((step, index) => (
                    <View key={step} style={styles.stepRow}>
                      <View style={[styles.stepNumber, { backgroundColor: config.softAccent }]}>
                        <Text style={[styles.stepNumberText, { color: config.accent }]}>{index + 1}</Text>
                      </View>
                      <Text style={styles.stepText}>{step}</Text>
                    </View>
                  ))}
                </View>
              </View>
            </View>

            <View style={styles.rightColumn}>
              <View style={[styles.coachCard, { backgroundColor: config.softAccent, borderColor: config.accent }]}>
                <View style={styles.coachIcon}>
                  <MaterialCommunityIcons name="lightbulb-on-outline" size={26} color={config.accent} />
                </View>
                <View style={styles.coachTextWrap}>
                  <Text style={[styles.coachTitle, { color: config.accent }]}>{config.coachTitle}</Text>
                  <Text style={styles.coachTip}>{config.coachTip}</Text>
                </View>
              </View>

              <View style={styles.recordCard}>
                <View style={styles.recordStatusRow}>
                  <View style={[styles.recordDot, recording ? styles.recordDotActive : null]} />
                  <Text style={styles.recordStatus}>{recordLabel}</Text>
                </View>
                <View style={styles.waveform}>
                  {waveformBars.map((bar, index) => (
                    <View
                      key={`${bar}-${index}`}
                      style={[
                        styles.waveBar,
                        { height: bar, backgroundColor: recording ? config.accent : '#B8C2D8' },
                      ]}
                    />
                  ))}
                </View>
                <Pressable
                  style={[styles.micButton, recording ? { backgroundColor: colors.error } : null]}
                  onPress={recording ? stopRecording : startRecording}
                >
                  <MaterialCommunityIcons name={recording ? 'stop' : 'microphone'} size={34} color="#FFFFFF" />
                </Pressable>
                <View style={styles.actionRow}>
                  <Pressable
                    style={[styles.secondaryAction, !audioUri ? styles.disabledAction : null]}
                    onPress={playRecording}
                    disabled={!audioUri}
                  >
                    <MaterialCommunityIcons name="replay" size={20} color={colors.primary} />
                    <Text style={styles.secondaryActionText}>Nghe lại</Text>
                  </Pressable>
                  <Pressable
                    style={[styles.primaryAction, (!audioUri || submitting) ? styles.disabledAction : null]}
                    onPress={submitToAi}
                    disabled={!audioUri || submitting}
                  >
                    {submitting ? (
                      <ActivityIndicator color="#FFFFFF" />
                    ) : (
                      <>
                        <MaterialCommunityIcons name="send" size={19} color="#FFFFFF" />
                        <Text style={styles.primaryActionText}>Gửi AI</Text>
                      </>
                    )}
                  </Pressable>
                </View>
              </View>

              {error ? <ErrorState title="Cần kiểm tra lại" message={error} /> : null}

              <View style={styles.resultCard}>
                <View style={styles.resultHeader}>
                  <Text style={styles.cardTitle}>Kết quả AI</Text>
                  {jobStatus ? (
                    <Text style={styles.resultStatus}>
                      {jobStatus === 'completed' ? 'Đã chấm' : jobStatus === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                    </Text>
                  ) : null}
                </View>
                {jobStatus ? (
                  <>
                    <View style={styles.scoreRow}>
                      <View style={[styles.scoreCircle, { borderColor: scoreTone.color, backgroundColor: scoreTone.softColor }]}>
                        <Text style={[styles.scoreText, { color: scoreTone.color }]}>
                          {job?.score != null ? `${job.score}%` : '...'}
                        </Text>
                      </View>
                      <View style={styles.scoreCopy}>
                        <Text style={[styles.scoreLabel, { color: scoreTone.color }]}>{scoreTone.label}</Text>
                        <Text style={styles.scoreSubcopy}>AI đang phân tích phát âm theo mục tiêu của bài.</Text>
                      </View>
                    </View>
                    <View style={styles.feedbackBlock}>
                      <Text style={styles.feedbackTitle}>Âm cần sửa</Text>
                      {problemLines.map((line, index) => (
                        <Text key={`${line}-${index}`} style={styles.feedbackLine}>{line}</Text>
                      ))}
                      <Text style={styles.feedbackTitle}>Nhận xét</Text>
                      {resultLines.map((line, index) => (
                        <Text key={`${line}-${index}`} style={styles.feedbackLine}>{line}</Text>
                      ))}
                    </View>
                  </>
                ) : (
                  <Text style={styles.emptyResult}>Ghi âm và gửi AI để xem điểm, âm cần sửa và gợi ý luyện tiếp.</Text>
                )}
                <View style={styles.resultActions}>
                  <Pressable style={styles.retryButton} onPress={retry}>
                    <MaterialCommunityIcons name="refresh" size={20} color={colors.primary} />
                    <Text style={styles.retryText}>Thử lại</Text>
                  </Pressable>
                  <Pressable style={styles.nextButton} onPress={nextTarget}>
                    <Text style={styles.nextText}>Từ tiếp theo</Text>
                    <MaterialCommunityIcons name="arrow-right" size={20} color="#FFFFFF" />
                  </Pressable>
                </View>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#FAF8FF' },
  scrollContent: { flexGrow: 1 },
  centeredRoot: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    padding: 24,
    backgroundColor: '#FAF8FF',
  },
  container: {
    width: '100%',
    maxWidth: 1180,
    alignSelf: 'center',
    paddingHorizontal: 20,
    paddingVertical: 24,
    gap: 18,
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  backButton: {
    minHeight: 42,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  backText: { color: colors.text, fontSize: 14, fontWeight: '800' },
  lessonPill: {
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  lessonPillText: { color: colors.primary, fontSize: 13, fontWeight: '900' },
  header: { gap: 8 },
  badge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    overflow: 'hidden',
    paddingHorizontal: 13,
    paddingVertical: 7,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  title: { color: colors.text, fontSize: 38, lineHeight: 44, fontWeight: '900' },
  subtitle: { color: colors.muted, fontSize: 17, lineHeight: 25 },
  grid: { gap: 22 },
  desktopGrid: { flexDirection: 'row', alignItems: 'flex-start' },
  leftColumn: { flex: 1.2, gap: 20 },
  rightColumn: { flex: 1, gap: 20 },
  targetCard: {
    minHeight: 250,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 28,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.05,
    shadowRadius: 18,
    elevation: 3,
  },
  targetBadge: {
    borderRadius: 999,
    overflow: 'hidden',
    paddingHorizontal: 13,
    paddingVertical: 7,
    fontSize: 12,
    fontWeight: '900',
  },
  targetWord: {
    color: colors.text,
    fontSize: 48,
    lineHeight: 56,
    fontWeight: '900',
    textAlign: 'center',
  },
  ipaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  ipa: { fontSize: 25, lineHeight: 34, fontWeight: '900' },
  dot: { color: '#C4CBD8', fontSize: 22, fontWeight: '900' },
  meaning: { color: colors.text, fontSize: 18, lineHeight: 26 },
  featureStrip: {
    marginTop: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: 14,
    width: '100%',
    alignItems: 'center',
    gap: 4,
  },
  featureLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  featureText: { fontSize: 17, fontWeight: '900' },
  drillCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 24,
    gap: 20,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.05,
    shadowRadius: 18,
    elevation: 3,
  },
  cardHeaderRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  cardTitle: { color: colors.text, fontSize: 21, fontWeight: '900' },
  partsRow: {
    minHeight: 110,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FBFCFF',
    padding: 16,
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  partWrap: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  partBubble: {
    minWidth: 70,
    minHeight: 58,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#D7DEF0',
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 14,
  },
  partText: { color: colors.text, fontSize: 25, fontWeight: '900' },
  plus: { color: '#B8C2D8', fontSize: 24, fontWeight: '900' },
  stepList: { gap: 12 },
  stepRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumberText: { fontSize: 14, fontWeight: '900' },
  stepText: { flex: 1, color: colors.text, fontSize: 15, lineHeight: 22 },
  coachCard: { borderRadius: 14, borderWidth: 1, padding: 22, flexDirection: 'row', gap: 16 },
  coachIcon: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  coachTextWrap: { flex: 1, gap: 6 },
  coachTitle: { fontSize: 17, fontWeight: '900' },
  coachTip: { color: colors.text, fontSize: 16, lineHeight: 25 },
  recordCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 24,
    alignItems: 'center',
    gap: 18,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.05,
    shadowRadius: 18,
    elevation: 3,
  },
  recordStatusRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  recordDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#D89292' },
  recordDotActive: { backgroundColor: colors.error },
  recordStatus: { color: colors.muted, fontSize: 14, fontWeight: '800' },
  waveform: {
    height: 58,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  waveBar: { width: 7, borderRadius: 999 },
  micButton: {
    width: 92,
    height: 92,
    borderRadius: 46,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.24,
    shadowRadius: 18,
    elevation: 6,
  },
  actionRow: { width: '100%', flexDirection: 'row', gap: 12 },
  secondaryAction: {
    flex: 1,
    minHeight: 52,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  primaryAction: {
    flex: 1,
    minHeight: 52,
    borderRadius: 10,
    backgroundColor: colors.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  disabledAction: { opacity: 0.55 },
  secondaryActionText: { color: colors.primary, fontSize: 14, fontWeight: '900' },
  primaryActionText: { color: '#FFFFFF', fontSize: 14, fontWeight: '900' },
  resultCard: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: 24,
    gap: 18,
    shadowColor: colors.text,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.05,
    shadowRadius: 18,
    elevation: 3,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  resultStatus: {
    borderRadius: 999,
    overflow: 'hidden',
    backgroundColor: colors.softBlue,
    color: colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 6,
    fontSize: 12,
    fontWeight: '900',
  },
  scoreRow: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  scoreCircle: {
    width: 84,
    height: 84,
    borderRadius: 42,
    borderWidth: 7,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreText: { fontSize: 20, fontWeight: '900' },
  scoreCopy: { flex: 1, gap: 4 },
  scoreLabel: { fontSize: 24, fontWeight: '900' },
  scoreSubcopy: { color: colors.muted, fontSize: 14, lineHeight: 20 },
  feedbackBlock: { borderTopWidth: 1, borderTopColor: colors.border, paddingTop: 14, gap: 8 },
  feedbackTitle: { color: colors.text, fontSize: 14, fontWeight: '900', marginTop: 4 },
  feedbackLine: { color: colors.muted, fontSize: 14, lineHeight: 21 },
  emptyResult: { color: colors.muted, fontSize: 15, lineHeight: 23 },
  resultActions: { flexDirection: 'row', gap: 12 },
  retryButton: {
    flex: 1,
    minHeight: 50,
    borderRadius: 10,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  retryText: { color: colors.primary, fontSize: 14, fontWeight: '900' },
  nextButton: {
    flex: 1,
    minHeight: 50,
    borderRadius: 10,
    backgroundColor: colors.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  nextText: { color: '#FFFFFF', fontSize: 14, fontWeight: '900' },
});
