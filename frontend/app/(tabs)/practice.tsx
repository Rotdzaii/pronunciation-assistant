import { useEffect, useRef, useState } from 'react';
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
import { Audio } from 'expo-av';
import { ErrorState, colors } from '../../components/AppUI';
import { createPracticeJob, fetchPracticeStatus } from '../../lib/api';
import { useAuth } from '../../lib/auth';
import type { PracticeStatusResponse } from '../../types';

const WAVEFORM_BARS = [12, 24, 40, 28, 46, 30, 34, 12];

const fallbackPracticeTarget = {
  word: 'Architecture',
  ipa: '/ˌɑːr.kɪ.tek.tʃɚ/',
  meaning: 'Kiến trúc',
  chips: ['Ar', 'chi', 'tec', 'ture'],
};

export default function PracticeScreen() {
  const { width, height } = useWindowDimensions();
  const isDesktop = width >= 768;
  const isWide = width >= 720;
  const { accessToken } = useAuth();
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<PracticeStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
    setRecording(recording);
  };

  const stopRecording = async () => {
    if (!recording) {
      return;
    }
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setAudioUri(uri ?? null);
    setRecording(null);
  };

  const handleCreateJob = async () => {
    if (!audioUri) {
      setError('Hãy ghi âm trước khi gửi.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await createPracticeJob(audioUri, accessToken);
      setJobId(response.job_id);
      setStatus({ status: 'queued' });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không tạo được lượt chấm.');
    } finally {
      setLoading(false);
    }
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
        setStatus(response);
        if (response.status === 'completed' || response.status === 'failed') {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không lấy được kết quả.');
      }
    }, 3000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [jobId, accessToken]);

  const recordStatus = recording
    ? 'Đang ghi âm... 00:03'
    : audioUri
      ? 'Bản ghi đã sẵn sàng'
      : 'Nhấn micro để bắt đầu';

  const content = (
    <View style={[styles.rootShell, isDesktop ? { height } : styles.mobileShell]}>
      <View style={styles.mainArea}>
        <View style={[styles.practiceContent, isDesktop ? styles.desktopContent : styles.mobileContent]}>
          <View style={styles.progressRow}>
            <View style={styles.progressPill}>
              <Text style={styles.progressText}>Từ 3 trên 10</Text>
            </View>
            <View style={styles.streakPill}>
              <Text style={styles.streakText}>● Chuỗi 5 ngày</Text>
            </View>
          </View>

          <View style={styles.wordCard}>
            <Text style={styles.wordLabel}>Từ mục tiêu</Text>

            <View style={styles.chipRow}>
              {fallbackPracticeTarget.chips.map((chip, index) => (
                <View
                  key={chip}
                  style={[
                    styles.chip,
                    index === 0 ? styles.chipTeal : null,
                    index === 1 ? styles.chipRed : null,
                    index > 1 ? styles.chipPlain : null,
                  ]}
                >
                  <Text
                    style={[
                      styles.chipText,
                      index === 0 ? styles.chipTextTeal : null,
                      index === 1 ? styles.chipTextRed : null,
                      index > 1 ? styles.chipTextPlain : null,
                    ]}
                  >
                    {chip}
                  </Text>
                </View>
              ))}
            </View>

            <Text style={styles.targetWord}>{fallbackPracticeTarget.word}</Text>

            <View style={styles.ipaRow}>
              <Text style={styles.ipaText}>{fallbackPracticeTarget.ipa}</Text>
              <Pressable style={styles.audioButton} onPress={playRecording} disabled={!audioUri}>
                <SpeakerIcon />
              </Pressable>
            </View>

            <Text style={styles.meaningLabel}>Nghĩa tiếng Việt</Text>
            <Text style={styles.meaningText}>"{fallbackPracticeTarget.meaning}"</Text>

            {status?.result ? (
              <View style={styles.resultStrip}>
                <Text style={styles.resultScore}>{status.result.score} điểm</Text>
                <Text style={styles.resultDetail}>
                  Âm cần sửa: {status.result.problem_phonemes.join(', ') || 'Chưa phát hiện lỗi nổi bật'}
                </Text>
              </View>
            ) : null}
          </View>

          {error ? (
            <View style={styles.errorWrap}>
              <ErrorState title="Cần kiểm tra lại" message={error} />
            </View>
          ) : null}

          <View style={styles.interactionArea}>
            <View style={styles.recordingStatusRow}>
              <View style={[styles.recordingDot, recording ? styles.recordingDotActive : null]} />
              <Text style={styles.recordingStatus}>{recordStatus}</Text>
            </View>

            <View style={styles.waveform}>
              {WAVEFORM_BARS.map((barHeight, index) => (
                <View
                  key={`${barHeight}-${index}`}
                  style={[
                    styles.waveBar,
                    { height: barHeight },
                    recording ? styles.waveBarActive : null,
                  ]}
                />
              ))}
            </View>

            <Pressable
              style={[styles.micButton, recording ? styles.micButtonRecording : null]}
              onPress={recording ? stopRecording : startRecording}
            >
              <MicIcon />
            </Pressable>

            <View style={[styles.actionRow, isWide ? styles.actionRowWide : null]}>
              <Pressable
                style={[styles.secondaryAction, !audioUri ? styles.actionDisabled : null]}
                onPress={playRecording}
                disabled={!audioUri}
              >
                <ReplayIcon disabled={!audioUri} />
                <Text style={[styles.secondaryActionText, !audioUri ? styles.actionTextDisabled : null]}>
                  Nghe lại
                </Text>
              </Pressable>

              <Pressable
                style={[styles.primaryAction, (!audioUri || loading) ? styles.primaryActionDisabled : null]}
                onPress={handleCreateJob}
                disabled={!audioUri || loading}
              >
                <Text style={styles.primaryActionIcon}>▷</Text>
                <Text style={styles.primaryActionText}>
                  {loading ? 'Đang gửi...' : 'Gửi AI chấm điểm'}
                </Text>
              </Pressable>
            </View>

            {loading ? <ActivityIndicator color={colors.primary} /> : null}
            <Text style={styles.skipText}>Bỏ qua từ</Text>
          </View>
        </View>
      </View>
    </View>
  );

  if (isDesktop) {
    return <View style={[styles.desktopRoot, { height }]}>{content}</View>;
  }

  return (
    <SafeAreaView style={styles.mobileRoot}>
      <ScrollView
        contentContainerStyle={styles.mobileScrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {content}
      </ScrollView>
    </SafeAreaView>
  );
}

function SpeakerIcon() {
  return (
    <View style={styles.speakerIcon}>
      <View style={styles.speakerBox} />
      <View style={styles.speakerWave} />
    </View>
  );
}

function MicIcon() {
  return (
    <View style={styles.micGlyph}>
      <View style={styles.micCapsule} />
      <View style={styles.micStem} />
      <View style={styles.micBase} />
    </View>
  );
}

function ReplayIcon({ disabled }: { disabled: boolean }) {
  return (
    <View style={[styles.replayIcon, disabled ? styles.replayIconDisabled : null]}>
      <View style={styles.replayArc} />
      <Text style={styles.replayArrow}>‹</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  desktopRoot: {
    flex: 1,
    backgroundColor: '#faf8ff',
    overflow: 'hidden',
  },
  mobileRoot: {
    flex: 1,
    backgroundColor: '#faf8ff',
  },
  mobileScrollContent: {
    flexGrow: 1,
  },
  rootShell: {
    width: '100%',
    flexDirection: 'row',
    backgroundColor: '#faf8ff',
    overflow: 'hidden',
  },
  mobileShell: {
    minHeight: 820,
    flexDirection: 'column',
    paddingHorizontal: 20,
    paddingVertical: 24,
  },
  mainArea: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#faf8ff',
    overflow: 'hidden',
  },
  practiceContent: {
    width: '100%',
    maxWidth: 600,
    flex: 1,
    position: 'relative',
  },
  desktopContent: {
    height: '100%',
  },
  mobileContent: {
    minHeight: 780,
  },
  progressRow: {
    position: 'absolute',
    top: 24,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  progressPill: {
    backgroundColor: '#e7e7f3',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  progressText: {
    color: '#434655',
    fontSize: 12,
    fontWeight: '500',
  },
  streakPill: {
    backgroundColor: '#bc4800',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  streakText: {
    color: '#ffede6',
    fontSize: 12,
    fontWeight: '800',
  },
  wordCard: {
    position: 'absolute',
    top: 64,
    left: 0,
    right: 0,
    minHeight: 304,
    backgroundColor: '#ffffff',
    borderColor: 'rgba(195,198,215,0.3)',
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 32,
    paddingVertical: 32,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 12,
    elevation: 3,
  },
  wordLabel: {
    color: '#737686',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 16,
  },
  chipRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 16,
  },
  chip: {
    minHeight: 32,
    borderRadius: 6,
    paddingHorizontal: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chipPlain: {
    backgroundColor: 'transparent',
  },
  chipTeal: {
    backgroundColor: 'rgba(0,107,95,0.1)',
  },
  chipRed: {
    backgroundColor: 'rgba(255,218,214,0.5)',
  },
  chipText: {
    fontSize: 16,
    fontWeight: '400',
  },
  chipTextTeal: {
    color: '#6df5e1',
  },
  chipTextRed: {
    color: '#ba1a1a',
  },
  chipTextPlain: {
    color: '#191b23',
  },
  targetWord: {
    color: '#191b23',
    fontSize: 32,
    lineHeight: 40,
    fontWeight: '800',
    marginBottom: 8,
  },
  ipaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 24,
  },
  ipaText: {
    color: '#004ac6',
    fontSize: 18,
    lineHeight: 28,
  },
  audioButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#ededf9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  speakerIcon: {
    width: 22,
    height: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  speakerBox: {
    width: 9,
    height: 13,
    borderRadius: 3,
    backgroundColor: '#004ac6',
  },
  speakerWave: {
    position: 'absolute',
    right: 0,
    width: 10,
    height: 17,
    borderRightWidth: 3,
    borderColor: '#004ac6',
    borderRadius: 9,
  },
  meaningLabel: {
    color: '#737686',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 4,
  },
  meaningText: {
    color: '#434655',
    fontSize: 16,
    lineHeight: 24,
    fontStyle: 'italic',
  },
  resultStrip: {
    marginTop: 10,
    backgroundColor: '#faf8ff',
    borderColor: '#c3c6d7',
    borderWidth: 1,
    borderRadius: 10,
    padding: 8,
    width: '100%',
  },
  resultScore: {
    color: '#191b23',
    fontSize: 12,
    fontWeight: '800',
    textAlign: 'center',
  },
  resultDetail: {
    color: '#434655',
    fontSize: 11,
    textAlign: 'center',
  },
  errorWrap: {
    position: 'absolute',
    top: 374,
    left: 0,
    right: 0,
  },
  interactionArea: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 504,
    alignItems: 'center',
  },
  recordingStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ba1a1a',
  },
  recordingDotActive: {
    backgroundColor: '#ba1a1a',
  },
  recordingStatus: {
    color: '#737686',
    fontSize: 14,
    fontWeight: '600',
  },
  waveform: {
    height: 42,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
    marginBottom: 20,
    opacity: 0.7,
  },
  waveBar: {
    width: 6,
    borderRadius: 999,
    backgroundColor: '#006b5f',
  },
  waveBarActive: {
    opacity: 1,
  },
  micButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#e1e2ed',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 28,
  },
  micButtonRecording: {
    backgroundColor: '#e1e2ed',
  },
  micGlyph: {
    alignItems: 'center',
  },
  micCapsule: {
    width: 16,
    height: 27,
    borderRadius: 8,
    borderWidth: 3,
    borderColor: '#434655',
  },
  micStem: {
    width: 3,
    height: 11,
    backgroundColor: '#434655',
  },
  micBase: {
    width: 22,
    height: 3,
    borderRadius: 2,
    backgroundColor: '#434655',
  },
  actionRow: {
    width: '100%',
    gap: 16,
  },
  actionRowWide: {
    flexDirection: 'row',
  },
  secondaryAction: {
    flex: 1,
    minHeight: 56,
    borderRadius: 8,
    backgroundColor: '#e7e7f3',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 6,
    elevation: 2,
  },
  secondaryActionText: {
    color: '#191b23',
    fontSize: 14,
    fontWeight: '700',
  },
  actionDisabled: {
    opacity: 0.72,
  },
  actionTextDisabled: {
    color: '#434655',
  },
  replayIcon: {
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  replayIconDisabled: {
    opacity: 0.7,
  },
  replayArc: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderLeftWidth: 2,
    borderBottomWidth: 2,
    borderColor: '#191b23',
  },
  replayArrow: {
    position: 'absolute',
    left: 0,
    top: -4,
    color: '#191b23',
    fontSize: 18,
    fontWeight: '900',
  },
  primaryAction: {
    flex: 1,
    minHeight: 56,
    borderRadius: 8,
    backgroundColor: '#004ac6',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 10,
    shadowColor: '#004ac6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 5,
  },
  primaryActionDisabled: {
    opacity: 0.72,
    shadowOpacity: 0,
    elevation: 0,
  },
  primaryActionIcon: {
    color: '#ffffff',
    fontSize: 21,
    fontWeight: '700',
  },
  primaryActionText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '800',
  },
  skipText: {
    color: '#c3c6d7',
    fontSize: 12,
    fontWeight: '600',
    marginTop: 22,
    paddingBottom: 8,
  },
});
