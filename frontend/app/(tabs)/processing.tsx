import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useLocalSearchParams } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, colors } from '../../components/AppUI';
import { BackButton } from '../../components/BackButton';
import type { PracticeJobStatus } from '../../types';

type StepState = 'done' | 'active' | 'idle' | 'error';

const processingSteps = [
  'Đang tải audio',
  'Đang phân tích đặc trưng giọng nói',
  'Đang phát hiện lỗi phát âm',
  'Đang tạo góp ý bằng tiếng Việt',
];

const validStatuses: PracticeJobStatus[] = ['queued', 'processing', 'completed', 'failed'];

export default function ProcessingScreen() {
  const params = useLocalSearchParams<{ jobId?: string; status?: string }>();
  const jobId = normalizeParam(params.jobId);
  const status = normalizeStatus(normalizeParam(params.status));
  const hasJob = Boolean(jobId);
  const activeIndex = getActiveStep(status, hasJob);

  return (
    <AppScreen maxWidth={680}>
      <BackButton label="Quay lại luyện tập" fallbackHref="/(tabs)/practice" />

      <View style={styles.centerStage}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>AI Analyst</Text>
        </View>

        <Text style={styles.title}>AI đang phân tích</Text>
        <Text style={styles.subtitle}>Hệ thống đang nghe kỹ phát âm của bạn...</Text>

        <View style={styles.ringStage}>
          <View style={styles.outerRing} />
          <View style={styles.middleRing} />
          <View style={styles.innerPulse}>
            <MaterialCommunityIcons name="waveform" size={38} color="#004AC6" />
          </View>
          <View style={styles.waveBars}>
            {[18, 24, 30, 22, 28].map((height, index) => (
              <View key={`${height}-${index}`} style={[styles.waveBar, { height }]} />
            ))}
          </View>
        </View>

        <AppCard style={styles.stepsCard}>
          {!hasJob ? (
            <View style={styles.emptyNotice}>
              <MaterialCommunityIcons name="information-outline" size={20} color={colors.primary} />
              <Text style={styles.emptyText}>Chưa có lượt chấm nào đang xử lý.</Text>
            </View>
          ) : null}

          {processingSteps.map((step, index) => (
            <StepRow
              key={step}
              label={step}
              state={getStepState(index, activeIndex, status, hasJob)}
            />
          ))}

          {status ? (
            <Text style={styles.statusText}>Trạng thái hiện tại: {translateStatus(status)}</Text>
          ) : null}
        </AppCard>

        <Text style={styles.waitingText}>
          Kết quả sẽ sẵn sàng sau khi hệ thống hoàn tất phân tích.
        </Text>
      </View>
    </AppScreen>
  );
}

function StepRow({ label, state }: { label: string; state: StepState }) {
  const active = state === 'active';
  const done = state === 'done';
  const error = state === 'error';

  return (
    <View style={styles.stepRow}>
      <View
        style={[
          styles.stepIcon,
          done ? styles.stepIconDone : null,
          active ? styles.stepIconActive : null,
          error ? styles.stepIconError : null,
        ]}
      >
        <MaterialCommunityIcons
          name={done ? 'check' : active ? 'sync' : error ? 'alert' : 'circle-small'}
          size={done || active || error ? 18 : 22}
          color={done || active || error ? '#FFFFFF' : '#94A3B8'}
        />
      </View>
      <Text
        style={[
          styles.stepText,
          done ? styles.stepTextDone : null,
          active ? styles.stepTextActive : null,
          error ? styles.stepTextError : null,
        ]}
      >
        {label}
      </Text>
    </View>
  );
}

function getActiveStep(status: PracticeJobStatus | null, hasJob: boolean) {
  if (!hasJob) {
    return -1;
  }

  if (status === 'completed') {
    return processingSteps.length;
  }

  if (status === 'failed') {
    return 2;
  }

  if (status === 'queued') {
    return 0;
  }

  return 1;
}

function getStepState(
  index: number,
  activeIndex: number,
  status: PracticeJobStatus | null,
  hasJob: boolean,
): StepState {
  if (!hasJob) {
    return 'idle';
  }

  if (status === 'failed' && index === activeIndex) {
    return 'error';
  }

  if (index < activeIndex || status === 'completed') {
    return 'done';
  }

  if (index === activeIndex) {
    return 'active';
  }

  return 'idle';
}

function translateStatus(status: PracticeJobStatus) {
  switch (status) {
    case 'queued':
      return 'Đang chờ xử lý';
    case 'processing':
      return 'Đang phân tích';
    case 'completed':
      return 'Hoàn tất';
    case 'failed':
      return 'Không thành công';
    default:
      return status;
  }
}

function normalizeStatus(value: string | null): PracticeJobStatus | null {
  if (value && validStatuses.includes(value as PracticeJobStatus)) {
    return value as PracticeJobStatus;
  }

  return null;
}

function normalizeParam(value: string | string[] | undefined) {
  if (Array.isArray(value)) {
    return value[0] || null;
  }

  return value || null;
}

const styles = StyleSheet.create({
  centerStage: {
    minHeight: 720,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    paddingVertical: 28,
  },
  badge: {
    borderRadius: 999,
    backgroundColor: '#DBE1FF',
    paddingHorizontal: 12,
    paddingVertical: 5,
    marginBottom: 4,
  },
  badgeText: {
    color: '#003EA8',
    fontSize: 12,
    fontWeight: '900',
  },
  title: {
    color: '#191B23',
    fontSize: 29,
    fontWeight: '900',
    lineHeight: 36,
    textAlign: 'center',
  },
  subtitle: {
    color: '#434655',
    fontSize: 17,
    lineHeight: 25,
    textAlign: 'center',
  },
  ringStage: {
    width: 260,
    height: 260,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 4,
  },
  outerRing: {
    position: 'absolute',
    width: 220,
    height: 220,
    borderRadius: 110,
    borderWidth: 5,
    borderColor: '#B7F4EC',
  },
  middleRing: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
    borderWidth: 5,
    borderColor: '#E1E7FF',
  },
  innerPulse: {
    width: 112,
    height: 112,
    borderRadius: 56,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E1E2ED',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#0F172A',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.12,
    shadowRadius: 18,
    elevation: 5,
  },
  waveBars: {
    position: 'absolute',
    bottom: 22,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 7,
  },
  waveBar: {
    width: 7,
    borderRadius: 999,
    backgroundColor: '#79A7F2',
  },
  stepsCard: {
    width: '100%',
    maxWidth: 560,
    borderRadius: 12,
    backgroundColor: '#F3F3FE',
    borderColor: '#DDE1F2',
    paddingVertical: 22,
    paddingHorizontal: 28,
    gap: 18,
  },
  emptyNotice: {
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#DBE1FF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
  },
  emptyText: {
    flex: 1,
    color: '#434655',
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '700',
  },
  stepRow: {
    minHeight: 38,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  stepIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#E1E2ED',
    borderWidth: 2,
    borderColor: '#C3C6D7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepIconDone: {
    backgroundColor: '#007A6C',
    borderColor: '#007A6C',
  },
  stepIconActive: {
    backgroundColor: '#004AC6',
    borderColor: '#004AC6',
  },
  stepIconError: {
    backgroundColor: '#BA1A1A',
    borderColor: '#BA1A1A',
  },
  stepText: {
    flex: 1,
    color: '#191B23',
    fontSize: 16,
    lineHeight: 23,
    fontWeight: '700',
  },
  stepTextDone: {
    color: '#191B23',
  },
  stepTextActive: {
    color: '#004AC6',
    fontWeight: '900',
  },
  stepTextError: {
    color: '#BA1A1A',
    fontWeight: '900',
  },
  statusText: {
    color: colors.muted,
    fontSize: 13,
    textAlign: 'center',
    fontWeight: '700',
  },
  waitingText: {
    color: '#434655',
    fontSize: 14,
    lineHeight: 21,
    textAlign: 'center',
    marginTop: 28,
  },
});
