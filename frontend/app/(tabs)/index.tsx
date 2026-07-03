import { MaterialCommunityIcons } from '@expo/vector-icons';
import { type Href, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, ErrorState, LoadingState, StatusBadge, colors } from '../../components/AppUI';
import { fetchStudentAssignments, fetchStudentClassDetail, fetchStudentClasses } from '../../lib/api';
import type { ClassDetail, ClassSummary, StudentAssignment } from '../../types';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;

type QuickAction = {
  label: string;
  icon: IconName;
  href: Href<string>;
};

type LearningMode = {
  title: string;
  description: string;
  icon: IconName;
  href: Href<string>;
  tags: string[];
  badge?: string;
  visualTone: 'blue' | 'teal' | 'orange' | 'red';
};

const quickActions: QuickAction[] = [
  { label: 'Luyện tập', icon: 'microphone-outline', href: '/(tabs)/practice' },
  { label: 'Thư viện lỗi', icon: 'alert-circle-outline', href: '/(tabs)/mistakes' },
  { label: 'Lịch sử', icon: 'history', href: '/(tabs)/history' },
  { label: 'Hồ sơ', icon: 'account-outline', href: '/(tabs)/profile' },
];

const learningModes: LearningMode[] = [
  {
    title: 'Luyện phát âm theo từ',
    description:
      'Ghi âm một từ mục tiêu, nghe lại bản ghi và nhận góp ý phát âm bằng tiếng Việt.',
    icon: 'microphone-outline',
    href: '/(tabs)/practice',
    tags: ['#Speaking', '#AI Coach'],
    badge: 'Ưu tiên',
    visualTone: 'blue',
  },
  {
    title: 'Luyện theo chủ đề',
    description:
      'Chọn chủ đề thực tế như nhà hàng, sân bay hoặc phỏng vấn để luyện nhóm từ liên quan.',
    icon: 'layers-outline',
    href: '/(tabs)/practice-mode',
    tags: ['#Vocabulary', '#Real-life'],
    visualTone: 'teal',
  },
  {
    title: 'Thư viện lỗi phổ biến',
    description:
      'Ôn lại các lỗi người Việt thường gặp: âm cuối, trọng âm, cụm phụ âm và nối âm.',
    icon: 'alert-circle-outline',
    href: '/(tabs)/mistakes',
    tags: ['#Phoneme', '#Vietnamese learners'],
    visualTone: 'red',
  },
  {
    title: 'Lịch sử và tiến độ',
    description:
      'Xem lại điểm số, phản hồi AI và các âm cần chú ý sau từng lượt luyện.',
    icon: 'chart-line',
    href: '/(tabs)/history',
    tags: ['#Progress', '#Feedback'],
    visualTone: 'orange',
  },
];

export default function HomeScreen() {
  const router = useRouter();
  const [classes, setClasses] = useState<ClassSummary[]>([]);
  const [selectedClass, setSelectedClass] = useState<ClassDetail | null>(null);
  const [classLoading, setClassLoading] = useState(true);
  const [classDetailLoading, setClassDetailLoading] = useState(false);
  const [classError, setClassError] = useState<string | null>(null);
  const [assignments, setAssignments] = useState<StudentAssignment[]>([]);

  useEffect(() => {
    console.debug('[StudentHome] class list effect');
    const loadClasses = async () => {
      setClassLoading(true);
      setClassError(null);
      try {
        const response = await fetchStudentClasses();
        setClasses(response);
        if (response[0]) {
          const detail = await fetchStudentClassDetail(response[0].id);
          setSelectedClass(detail);
        }
      } catch (err) {
        setClassError(err instanceof Error ? err.message : 'Chưa thể tải danh sách lớp.');
      } finally {
        setClassLoading(false);
      }
    };

    void loadClasses();
  }, []);

  useEffect(() => {
    fetchStudentAssignments().then(setAssignments).catch(() => undefined);
  }, []);

  const openClassDetail = async (classId: string) => {
    setClassDetailLoading(true);
    setClassError(null);
    try {
      setSelectedClass(await fetchStudentClassDetail(classId));
    } catch (err) {
      setClassError(err instanceof Error ? err.message : 'Chưa thể tải chi tiết lớp.');
    } finally {
      setClassDetailLoading(false);
    }
  };

  return (
    <AppScreen maxWidth={1200}>
      <View style={styles.dashboardGrid}>
        <View style={styles.mainColumn}>
          <View style={styles.heroCard}>
            <View style={styles.heroGlow} />
            <View style={styles.heroContent}>
              <Text style={styles.heroEyebrow}>Chào buổi sáng!</Text>
              <Text style={styles.heroTitle}>Hôm nay luyện 5 phút nhé?</Text>
              <Text style={styles.heroText}>
                Tiếp tục hành trình phát âm tiếng Anh của bạn. Ghi âm một bài ngắn,
                nhận góp ý rõ ràng và cải thiện từng âm một cách có mục tiêu.
              </Text>
              <View style={styles.heroActions}>
                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push('/(tabs)/practice')}
                  style={styles.primaryAction}
                >
                  <Text style={styles.primaryActionText}>Luyện phát âm ngay</Text>
                  <MaterialCommunityIcons name="arrow-right" size={18} color="#FFFFFF" />
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  onPress={() => router.push('/(tabs)/practice-mode')}
                  style={styles.secondaryAction}
                >
                  <Text style={styles.secondaryActionText}>Chọn chủ đề</Text>
                  <MaterialCommunityIcons name="shape-outline" size={18} color={colors.text} />
                </Pressable>
              </View>
            </View>
          </View>

          <View style={styles.quickActionGrid}>
            {quickActions.map((action) => (
              <Pressable
                key={action.label}
                accessibilityRole="button"
                onPress={() => router.push(action.href)}
                style={styles.quickAction}
              >
                <View style={styles.quickActionIcon}>
                  <MaterialCommunityIcons name={action.icon} size={22} color={colors.primary} />
                </View>
                <Text style={styles.quickActionText}>{action.label}</Text>
              </Pressable>
            ))}
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Chế độ học</Text>
            <Pressable onPress={() => router.push('/(tabs)/practice-mode')}>
              <Text style={styles.sectionLink}>Xem tất cả</Text>
            </Pressable>
          </View>

          <View style={styles.learningList}>
            {learningModes.map((mode) => (
              <LearningModeCard key={mode.title} mode={mode} onPress={() => router.push(mode.href)} />
            ))}
          </View>

          <AppCard style={styles.aiTipCard}>
            <View style={styles.aiTipIcon}>
              <MaterialCommunityIcons name="head-cog-outline" size={24} color="#FFFFFF" />
            </View>
            <View style={styles.aiTipCopy}>
              <Text style={styles.aiTipTitle}>Gợi ý hôm nay</Text>
              <Text style={styles.aiTipText}>
                Sau khi có kết quả luyện tập thật, khu vực này sẽ ưu tiên âm cần sửa tiếp theo.
                Bạn có thể bắt đầu bằng một bài luyện từ ngắn để tạo dữ liệu đầu tiên.
              </Text>
              <Pressable onPress={() => router.push('/(tabs)/practice')} style={styles.inlineLink}>
                <Text style={styles.inlineLinkText}>Luyện ngay</Text>
                <MaterialCommunityIcons name="arrow-right" size={16} color={colors.primary} />
              </Pressable>
            </View>
          </AppCard>
        </View>

          <View style={styles.sideColumn}>
          <StudentClassesCard
            classes={classes}
            selectedClass={selectedClass}
            loading={classLoading}
            detailLoading={classDetailLoading}
            error={classError}
            onSelectClass={openClassDetail}
          />

          <AppCard style={styles.profileCard}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>PA</Text>
            </View>
            <View style={styles.profileCopy}>
              <Text style={styles.profileName}>Người học</Text>
              <Text style={styles.profileMeta}>Sẵn sàng luyện phát âm</Text>
            </View>
            <MaterialCommunityIcons name="dots-vertical" size={22} color={colors.muted} />
          </AppCard>

          <View style={styles.statGrid}>
            <SmallStatCard icon="fire" value="--" label="Ngày liên tiếp" tone="orange" />
            <SmallStatCard icon="target" value="--" label="Độ chuẩn xác" tone="blue" />
          </View>

          <AppCard style={styles.progressCard}>
            <View style={styles.progressHeader}>
              <Text style={styles.progressKicker}>Mục tiêu tuần</Text>
              <StatusBadge label="Chưa có dữ liệu" tone="processing" style={styles.progressBadge} />
            </View>
            <Text style={styles.progressTitle}>Bắt đầu tuần học mới</Text>
            <ProgressLine label="Thời gian học" value="0/30p" percent={0} />
            <ProgressLine
              label="Bài tập hoàn thành"
              value={assignments.length > 0
                ? `${assignments.filter((a) => a.progress_status === 'completed').length}/${assignments.length}`
                : '0/0'}
              percent={assignments.length > 0
                ? Math.round((assignments.filter((a) => a.progress_status === 'completed').length / assignments.length) * 100)
                : 0}
            />
            {assignments.length === 0 ? (
              <View style={styles.progressNoteBox}>
                <Text style={styles.progressNoteText}>
                  Hoàn thành bài luyện đầu tiên để cập nhật tiến độ thật từ hệ thống.
                </Text>
              </View>
            ) : null}
          </AppCard>

          <AppCard style={styles.goalCard}>
            <View style={styles.goalIcon}>
              <MaterialCommunityIcons name="microphone-outline" size={24} color={colors.primary} />
            </View>
            <Text style={styles.goalTitle}>Mục tiêu nói hôm nay</Text>
            <Text style={styles.goalText}>
              Ghi âm ít nhất một từ hoặc một câu ngắn. Việc luyện đều mỗi ngày giúp bạn tự tin hơn.
            </Text>
            <Pressable onPress={() => router.push('/(tabs)/practice')} style={styles.outlineButton}>
              <Text style={styles.outlineButtonText}>Xem chi tiết</Text>
            </Pressable>
          </AppCard>

          <AppCard style={styles.tipCard}>
            <Text style={styles.tipTitle}>Mẹo nhỏ</Text>
            <Text style={styles.tipText}>
              Khi luyện âm cuối, hãy giữ hơi thêm nửa nhịp trước khi kết thúc từ.
            </Text>
          </AppCard>
        </View>
      </View>
    </AppScreen>
  );
}

function LearningModeCard({
  mode,
  onPress,
}: {
  mode: LearningMode;
  onPress: () => void;
}) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.learningCard}>
      <View style={styles.learningCopy}>
        <View style={[styles.learningIcon, visualToneStyles[mode.visualTone]]}>
          <MaterialCommunityIcons
            name={mode.icon}
            size={24}
            color={mode.visualTone === 'orange' ? '#FFFFFF' : colors.primary}
          />
        </View>
        <Text style={styles.learningTitle}>{mode.title}</Text>
        <Text style={styles.learningDescription}>{mode.description}</Text>
        <View style={styles.tagRow}>
          {mode.tags.map((tag) => (
            <Text key={tag} style={styles.tag}>
              {tag}
            </Text>
          ))}
        </View>
      </View>
      <View style={[styles.learningVisual, visualToneStyles[mode.visualTone]]}>
        <View style={styles.visualCircleLarge} />
        <View style={styles.visualCircleSmall} />
        <MaterialCommunityIcons name={mode.icon} size={46} color={colors.primary} />
        {mode.badge ? (
          <View style={styles.popularBadge}>
            <Text style={styles.popularBadgeText}>{mode.badge}</Text>
          </View>
        ) : null}
      </View>
    </Pressable>
  );
}

function StudentClassesCard({
  classes,
  selectedClass,
  loading,
  detailLoading,
  error,
  onSelectClass,
}: {
  classes: ClassSummary[];
  selectedClass: ClassDetail | null;
  loading: boolean;
  detailLoading: boolean;
  error: string | null;
  onSelectClass: (classId: string) => void;
}) {
  if (loading) {
    return <LoadingState title="Đang tải lớp của tôi" message="Hệ thống đang lấy các lớp học mà bạn đang tham gia." />;
  }

  return (
    <AppCard style={styles.classCard}>
      <View style={styles.classHeader}>
        <View>
          <Text style={styles.classKicker}>Lớp của tôi</Text>
          <Text style={styles.classTitle}>Các lớp học mà bạn đang tham gia.</Text>
        </View>
        <StatusBadge label={`${classes.length} lớp`} tone="primary" />
      </View>

      {error ? <ErrorState title="Chưa thể tải lớp" message={error} /> : null}

      <View style={styles.classList}>
        {classes.length > 0 ? (
          classes.map((classItem) => {
            const active = selectedClass?.id === classItem.id;
            return (
              <Pressable
                key={classItem.id}
                accessibilityRole="button"
                onPress={() => onSelectClass(classItem.id)}
                style={[styles.classRow, active ? styles.classRowActive : null]}
              >
                <View style={styles.classRowMain}>
                  <Text style={styles.classCode}>{classItem.code ?? 'Chưa có mã'}</Text>
                  <Text style={styles.className}>{classItem.name}</Text>
                  {classItem.description ? (
                    <Text style={styles.classDescription} numberOfLines={2}>{classItem.description}</Text>
                  ) : null}
                  <View style={styles.classMetaRow}>
                    <Text style={styles.classMetaText}>{classItem.student_count} học sinh</Text>
                    <Text style={styles.classMetaText}>{classItem.teacher_count} giảng viên</Text>
                  </View>
                  <View style={styles.teacherList}>
                    {(classItem.teachers ?? []).slice(0, 3).map((teacher) => (
                      <View key={teacher.id} style={styles.teacherPill}>
                        <MaterialCommunityIcons name="account-tie-outline" size={15} color={colors.primary} />
                        <Text style={styles.teacherPillText}>{teacher.display_name}</Text>
                      </View>
                    ))}
                  </View>
                </View>
                <View style={styles.classAction}>
                  <Text style={styles.classActionText}>Xem chi tiết</Text>
                  <MaterialCommunityIcons name="chevron-right" size={18} color={colors.primary} />
                </View>
              </Pressable>
            );
          })
        ) : (
          <Text style={styles.classEmptyText}>Bạn chưa tham gia lớp nào. Vui lòng liên hệ giảng viên để được thêm vào lớp.</Text>
        )}
      </View>

      {detailLoading ? (
        <Text style={styles.classEmptyText}>Đang tải chi tiết lớp...</Text>
      ) : selectedClass ? (
        <View style={styles.classDetailBox}>
          <Text style={styles.classDetailKicker}>Thông tin lớp</Text>
          <Text style={styles.classDetailTitle}>{selectedClass.name}</Text>
          {selectedClass.description ? (
            <Text style={styles.classDetailText}>{selectedClass.description}</Text>
          ) : null}
          <View style={styles.classStatRow}>
            <ClassMiniStat label="Học sinh" value={selectedClass.student_count} />
            <ClassMiniStat label="Giảng viên" value={selectedClass.teacher_count} />
          </View>
          <Text style={styles.classDetailKicker}>Giảng viên phụ trách</Text>
          <View style={styles.teacherList}>
            {(selectedClass.teachers ?? []).slice(0, 3).map((teacher) => (
              <View key={teacher.id} style={styles.teacherPill}>
                <MaterialCommunityIcons name="account-tie-outline" size={15} color={colors.primary} />
                <Text style={styles.teacherPillText}>{teacher.display_name}</Text>
              </View>
            ))}
          </View>
        </View>
      ) : null}
    </AppCard>
  );
}

function ClassMiniStat({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.classMiniStat}>
      <Text style={styles.classMiniValue}>{value}</Text>
      <Text style={styles.classMiniLabel}>{label}</Text>
    </View>
  );
}

function SmallStatCard({
  icon,
  value,
  label,
  tone,
}: {
  icon: IconName;
  value: string;
  label: string;
  tone: 'blue' | 'orange';
}) {
  return (
    <AppCard style={styles.smallStatCard}>
      <MaterialCommunityIcons
        name={icon}
        size={28}
        color={tone === 'orange' ? colors.accent : colors.primary}
      />
      <Text style={styles.smallStatValue}>{value}</Text>
      <Text style={styles.smallStatLabel}>{label}</Text>
    </AppCard>
  );
}

function ProgressLine({
  label,
  value,
  percent,
}: {
  label: string;
  value: string;
  percent: number;
}) {
  return (
    <View style={styles.progressLine}>
      <View style={styles.progressLineTop}>
        <Text style={styles.progressLineLabel}>{label}</Text>
        <Text style={styles.progressLineValue}>{value}</Text>
      </View>
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${percent}%` }]} />
      </View>
    </View>
  );
}

const visualToneStyles = StyleSheet.create({
  blue: {
    backgroundColor: colors.softBlue,
  },
  teal: {
    backgroundColor: colors.softTeal,
  },
  orange: {
    backgroundColor: colors.accent,
  },
  red: {
    backgroundColor: colors.softRed,
  },
});

const styles = StyleSheet.create({
  dashboardGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 24,
  },
  mainColumn: {
    flexGrow: 2,
    flexBasis: 620,
    gap: 22,
  },
  sideColumn: {
    flexGrow: 1,
    flexBasis: 300,
    gap: 18,
  },
  classCard: {
    borderRadius: 20,
    gap: 14,
  },
  classHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 12,
  },
  classKicker: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  classTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
    marginTop: 3,
  },
  classList: {
    gap: 8,
  },
  classRow: {
    minHeight: 68,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    gap: 10,
  },
  classRowActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  classRowMain: {
    flex: 1,
    gap: 3,
  },
  classCode: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  className: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  classDescription: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  classMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  classMetaText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  classAction: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  classActionText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  classEmptyText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  classDetailBox: {
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: '#F8FBFF',
    padding: 14,
    gap: 10,
  },
  classDetailTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: '900',
  },
  classDetailKicker: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  classDetailText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  classStatRow: {
    flexDirection: 'row',
    gap: 8,
  },
  classMiniStat: {
    flex: 1,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: colors.border,
    padding: 10,
  },
  classMiniValue: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '900',
  },
  classMiniLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  teacherList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  teacherPill: {
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  teacherPillText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  heroCard: {
    minHeight: 318,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: '#D8E3FB',
    backgroundColor: '#EEF4FF',
    padding: 32,
    overflow: 'hidden',
  },
  heroGlow: {
    position: 'absolute',
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: '#CFFAFE',
    opacity: 0.7,
    right: -54,
    top: -64,
  },
  heroContent: {
    maxWidth: 680,
    gap: 16,
  },
  heroEyebrow: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  heroTitle: {
    color: colors.text,
    fontSize: 44,
    lineHeight: 54,
    fontWeight: '900',
  },
  heroText: {
    color: colors.muted,
    fontSize: 17,
    lineHeight: 27,
    maxWidth: 650,
  },
  heroActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
    marginTop: 8,
  },
  primaryAction: {
    minHeight: 56,
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
    elevation: 5,
  },
  primaryActionText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '900',
  },
  secondaryAction: {
    minHeight: 56,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#C3C6D7',
    backgroundColor: colors.surface,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 22,
  },
  secondaryActionText: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  quickActionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  quickAction: {
    flexGrow: 1,
    flexBasis: 150,
    minHeight: 76,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 14,
  },
  quickActionIcon: {
    width: 40,
    height: 40,
    borderRadius: 14,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickActionText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 12,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 28,
    lineHeight: 35,
    fontWeight: '900',
  },
  sectionLink: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
  learningList: {
    gap: 22,
  },
  learningCard: {
    minHeight: 245,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    overflow: 'hidden',
    flexDirection: 'row',
  },
  learningCopy: {
    flex: 1.45,
    padding: 24,
    justifyContent: 'center',
    gap: 12,
  },
  learningIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  learningTitle: {
    color: colors.text,
    fontSize: 25,
    lineHeight: 32,
    fontWeight: '900',
  },
  learningDescription: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 23,
  },
  tagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 4,
  },
  tag: {
    color: colors.primary,
    backgroundColor: colors.softBlue,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
    fontSize: 12,
    fontWeight: '900',
  },
  learningVisual: {
    flex: 1,
    minWidth: 180,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  visualCircleLarge: {
    position: 'absolute',
    width: 170,
    height: 170,
    borderRadius: 85,
    backgroundColor: 'rgba(255,255,255,0.58)',
    right: -40,
    top: -30,
  },
  visualCircleSmall: {
    position: 'absolute',
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: 'rgba(255,255,255,0.72)',
    left: 22,
    bottom: 28,
  },
  popularBadge: {
    position: 'absolute',
    right: 18,
    bottom: 18,
    borderRadius: 999,
    backgroundColor: colors.primary,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  popularBadgeText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '900',
  },
  aiTipCard: {
    borderRadius: 20,
    borderColor: '#99F6E4',
    backgroundColor: '#DDFDF8',
    flexDirection: 'row',
    gap: 16,
    padding: 24,
  },
  aiTipIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: '#007A6C',
    alignItems: 'center',
    justifyContent: 'center',
  },
  aiTipCopy: {
    flex: 1,
    gap: 8,
  },
  aiTipTitle: {
    color: colors.text,
    fontSize: 17,
    fontWeight: '900',
  },
  aiTipText: {
    color: colors.text,
    fontSize: 15,
    lineHeight: 23,
  },
  inlineLink: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  inlineLinkText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '900',
  },
  profileCard: {
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  avatar: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
  },
  profileCopy: {
    flex: 1,
    gap: 3,
  },
  profileName: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  profileMeta: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  statGrid: {
    flexDirection: 'row',
    gap: 14,
  },
  smallStatCard: {
    flex: 1,
    minHeight: 148,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  smallStatValue: {
    color: colors.text,
    fontSize: 34,
    fontWeight: '900',
    lineHeight: 38,
  },
  smallStatLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
    textAlign: 'center',
  },
  progressCard: {
    borderRadius: 20,
    gap: 16,
  },
  progressHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  progressKicker: {
    color: colors.text,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  progressBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  progressTitle: {
    color: colors.text,
    fontSize: 24,
    lineHeight: 30,
    fontWeight: '900',
  },
  progressLine: {
    gap: 8,
  },
  progressLineTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
  },
  progressLineLabel: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '700',
  },
  progressLineValue: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  progressTrack: {
    height: 9,
    borderRadius: 999,
    backgroundColor: '#D8E3FB',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 999,
    backgroundColor: colors.primary,
  },
  progressNoteBox: {
    borderRadius: 14,
    backgroundColor: colors.softTeal,
    borderColor: '#99F6E4',
    borderWidth: 1,
    padding: 14,
  },
  progressNoteText: {
    color: '#006B5F',
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
    fontWeight: '800',
  },
  goalCard: {
    borderRadius: 20,
    gap: 12,
  },
  goalIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  goalTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  goalText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 22,
  },
  outlineButton: {
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#C3C6D7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  outlineButtonText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  tipCard: {
    borderRadius: 20,
    backgroundColor: colors.softOrange,
    borderColor: '#FED7AA',
  },
  tipTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  tipText: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
});
