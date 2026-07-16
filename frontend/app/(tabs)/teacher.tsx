import { useEffect, useMemo, useRef, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, TouchableWithoutFeedback, useWindowDimensions, View } from 'react-native';
import {
  AppCard,
  AppScreen,
  ErrorState,
  LoadingState,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import {
  createAssignment,
  fetchTeacherAnalytics,
  fetchTeacherAssignments,
  fetchAssignmentGradebook,
  overrideAssignmentGrade,
  fetchTeacherClassDetail,
  fetchTeacherClassScores,
  fetchTeacherClassStudents,
  fetchTeacherClasses,
  fetchTeacherReviewRequestDetail,
  fetchTeacherReviewRequests,
  fetchTeacherStudentScores,
  fetchVocabularySets,
  addTeacherReviewNote,
  requestTeacherReviewReanalysis,
  resolveTeacherReviewRequest,
} from '../../lib/api';
import {
  WeekInMonth,
  formatDateRange,
  getCurrentWeekOfMonth,
  getWeeksInMonth,
  startOfMondayWeek,
} from '../../lib/dateRange';
import { exportCsv } from '../../lib/exportCsv';
import { useAuth } from '../../lib/auth';
import {
  formatPracticeStatus,
  formatReviewReason,
  formatReviewSeverity,
  formatReviewSource,
  formatReviewStatus,
} from '../../lib/format';
import type {
  Assignment,
  AssignmentGradebookItem,
  ClassDetail,
  ClassStudent,
  ClassSummary,
  TeacherAnalyticsResponse,
  TeacherClassScoreStudent,
  TeacherClassScoresResponse,
  TeacherCommonError,
  TeacherReviewRequest,
  TeacherReviewRequestDetail,
  TeacherReviewRequestsResponse,
  TeacherStudentScoresResponse,
  TeacherStudentSummary,
  VocabularySet,
} from '../../types';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;
type TeacherSection = 'overview' | 'classes' | 'reports' | 'settings' | 'support';
type ClassWorkspaceTab = 'students' | 'scores' | 'assignments' | 'reviews';
type ReportPeriod = 'week' | 'month';
type ReportKind = 'learning' | 'progress' | 'errors' | 'assignments' | 'ai_review';

const TEACHER_SECTION_EVENT = 'phoenix:teacher-section';

const ZERO_ANALYTICS: Required<TeacherAnalyticsResponse> = {
  total_students: 0,
  total_practice_sessions: 0,
  average_score: 0,
  avg_score: 0,
  active_jobs: 0,
  need_support_count: 0,
  good_progress_count: 0,
  improving_count: 0,
  common_errors: [],
  progress_distribution: {
    need_support: 0,
    improving: 0,
    good_progress: 0,
  },
  students: [],
};

const SECTION_COPY: Record<TeacherSection, { title: string; subtitle: string }> = {
  overview: {
    title: 'Tổng quan',
    subtitle: 'Những việc giáo viên cần chú ý trong hôm nay và tình hình chung của các lớp đang dạy.',
  },
  classes: {
    title: 'Lớp học',
    subtitle: 'Chọn một lớp để xem học viên, điểm luyện tập, bài giao và lỗi phát âm.',
  },
  reports: {
    title: 'Báo cáo',
    subtitle: 'Xuất báo cáo CSV theo tuần, tháng và từng nhóm dữ liệu.',
  },
  settings: {
    title: 'Cài đặt',
    subtitle: 'Các tuỳ chọn dành cho workspace giáo viên.',
  },
  support: {
    title: 'Hỗ trợ',
    subtitle: 'Tài nguyên hỗ trợ sử dụng Phoenix trong lớp học.',
  },
};

const assignmentStatuses = [
  'Chưa nộp',
  'Đã nộp',
  'Đang chẩn đoán',
  'Đã có kết quả',
  'Cần giáo viên kiểm tra',
  'Xử lý lỗi',
];

const reportKinds: Array<{ key: ReportKind; label: string }> = [
  
  { key: 'progress', label: 'Tiến độ luyện tập' },
  { key: 'errors', label: 'Lỗi phát âm' },
  { key: 'assignments', label: 'Bài luyện đã giao' },
  { key: 'ai_review', label: 'Kết quả AI cần kiểm tra' },
];

export default function TeacherScreen() {
  const { accessToken } = useAuth();
  const [activeSection, setActiveSection] = useState<TeacherSection>(() => getPendingTeacherSection());
  const [data, setData] = useState<TeacherAnalyticsResponse | null>(null);
  const [classes, setClasses] = useState<ClassSummary[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);
  const [selectedClass, setSelectedClass] = useState<ClassDetail | null>(null);
  const [classStudents, setClassStudents] = useState<ClassStudent[]>([]);
  const [classWorkspaceTab, setClassWorkspaceTab] = useState<ClassWorkspaceTab>('students');
  const [selectedWeek, setSelectedWeek] = useState(() => getCurrentWeekOfMonth());
  const [reportPeriod, setReportPeriod] = useState<ReportPeriod>('week');
  const [reportKind, setReportKind] = useState<ReportKind>('learning');
  const [notice, setNotice] = useState<string | null>(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(true);
  const [loadingClasses, setLoadingClasses] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classError, setClassError] = useState<string | null>(null);
  const [reviewData, setReviewData] = useState<TeacherReviewRequestsResponse | null>(null);
  const [reviewDetail, setReviewDetail] = useState<TeacherReviewRequestDetail | null>(null);
  const [reviewSearch, setReviewSearch] = useState('');
  const [loadingReviews, setLoadingReviews] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [selectedScoreStudent, setSelectedScoreStudent] = useState<ClassStudent | null>(null);
  const [studentScores, setStudentScores] = useState<TeacherStudentScoresResponse | null>(null);
  const [classScores, setClassScores] = useState<TeacherClassScoresResponse | null>(null);
  const [loadingScores, setLoadingScores] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);

  useEffect(() => {
    const pendingSection = getPendingTeacherSection();
    if (pendingSection !== activeSection) {
      setActiveSection(pendingSection);
    }

    const onSectionChange = (event: Event) => {
      const detail = normalizeTeacherSection((event as CustomEvent<string>).detail);
      if (detail) {
        setActiveSection(detail);
      }
    };

    if (typeof window !== 'undefined') {
      window.addEventListener(TEACHER_SECTION_EVENT, onSectionChange);
      return () => window.removeEventListener(TEACHER_SECTION_EVENT, onSectionChange);
    }

    return undefined;
  }, [activeSection]);

  useEffect(() => {
    const loadAnalytics = async () => {
      setLoadingAnalytics(true);
      setError(null);
      try {
        setData(await fetchTeacherAnalytics(accessToken));
      } catch (err) {
        setData(null);
        setError(err instanceof Error ? err.message : 'Chưa thể cập nhật số liệu lớp học.');
      } finally {
        setLoadingAnalytics(false);
      }
    };

    void loadAnalytics();
  }, [accessToken]);

  useEffect(() => {
    const loadClasses = async () => {
      setLoadingClasses(true);
      setClassError(null);
      try {
        const response = await fetchTeacherClasses();
        setClasses(response);
        setSelectedClassId((current) => current ?? response[0]?.id ?? null);
      } catch (err) {
        setClassError(err instanceof Error ? err.message : 'Chưa thể tải danh sách lớp.');
      } finally {
        setLoadingClasses(false);
      }
    };

    void loadClasses();
  }, []);

  useEffect(() => {
    if (!selectedClassId) {
      setSelectedClass(null);
      setClassStudents([]);
      return;
    }

    const loadClassDetail = async () => {
      setLoadingDetail(true);
      setClassError(null);
      try {
        const [detail, students] = await Promise.all([
          fetchTeacherClassDetail(selectedClassId),
          fetchTeacherClassStudents(selectedClassId),
        ]);
        setSelectedClass(detail);
        setClassStudents(students);
      } catch (err) {
        setClassError(err instanceof Error ? err.message : 'Chưa thể tải chi tiết lớp.');
      } finally {
        setLoadingDetail(false);
      }
    };

    void loadClassDetail();
  }, [selectedClassId]);

  useEffect(() => {
    if (activeSection !== 'overview' && activeSection !== 'classes') return;

    const loadReviews = async () => {
      setLoadingReviews(true);
      setReviewError(null);
      try {
        const response = await fetchTeacherReviewRequests({
          class_id: activeSection === 'classes' ? selectedClassId : undefined,
          q: reviewSearch,
          limit: 50,
          offset: 0,
        });
        setReviewData(response);
      } catch (err) {
        setReviewData(null);
        setReviewError(err instanceof Error ? err.message : 'Chưa thể tải yêu cầu xem lại.');
      } finally {
        setLoadingReviews(false);
      }
    };

    void loadReviews();
  }, [activeSection, reviewSearch, selectedClassId]);

  useEffect(() => {
    if (activeSection !== 'classes' || !selectedScoreStudent) return;

    const loadStudentScores = async () => {
      setLoadingScores(true);
      setScoreError(null);
      try {
        setStudentScores(await fetchTeacherStudentScores(selectedScoreStudent.id, { class_id: selectedClassId }));
      } catch (err) {
        setStudentScores(null);
        setScoreError(err instanceof Error ? err.message : 'Chưa thể tải điểm học sinh.');
      } finally {
        setLoadingScores(false);
      }
    };

    void loadStudentScores();
  }, [activeSection, selectedClassId, selectedScoreStudent]);

  useEffect(() => {
    if ((activeSection !== 'overview' && activeSection !== 'classes') || !selectedClassId) return;

    const loadClassScores = async () => {
      try {
        setClassScores(await fetchTeacherClassScores(selectedClassId));
      } catch {
        setClassScores(null);
      }
    };

    void loadClassScores();
  }, [activeSection, selectedClassId]);

  const analytics = useMemo(() => normalizeAnalytics(data), [data]);
  const classStats = useMemo(() => summarizeClasses(classes), [classes]);
  const aiReviewItems = useMemo(() => reviewData?.items ?? [], [reviewData]);
  const classNotificationCount = reviewData?.pending_count ?? 0;
  const hasPracticeData = analytics.total_practice_sessions > 0;
  const sectionCopy = SECTION_COPY[activeSection];

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new CustomEvent('phoenix:teacher-notifications', {
      detail: { classes: classNotificationCount },
    }));
  }, [classNotificationCount]);

  const setSection = (section: TeacherSection) => {
    setNotice(null);
    setActiveSection(section);
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent(TEACHER_SECTION_EVENT, { detail: section }));
    }
  };

  const exportReport = (kind: ReportKind = reportKind) => {
    const result = exportCsv(
      `phoenix_teacher_${kind}.csv`,
      ['Loại báo cáo', 'Kỳ báo cáo', 'Chỉ số', 'Giá trị', 'Ghi chú'],
      buildReportRows(kind, reportPeriod, analytics, classStats),
    );
    setNotice(result.message);
  };

  return (
    <AppScreen maxWidth={1180}>
      <View style={styles.headerBar}>
        <View>
          <Text style={styles.appTitle}>Bảng điều khiển giáo viên</Text>
          <Text style={styles.appSubtitle}>Phoenix Teacher Workspace</Text>
        </View>
        <StatusBadge label={`${classes.length} lớp`} tone="primary" />
      </View>

      <View style={styles.dashboardHeader}>
        <View style={styles.dashboardCopy}>
          <Text style={styles.sectionTitle}>{sectionCopy.title}</Text>
          <Text style={styles.sectionSubtitle}>{sectionCopy.subtitle}</Text>
        </View>
        <HeaderActions
          section={activeSection}
          selectedWeek={selectedWeek}
          onWeekChange={setSelectedWeek}
          onReports={() => setSection('reports')}
          onExport={exportReport}
        />
      </View>

      {notice ? <Notice message={notice} /> : null}
      {loadingAnalytics || loadingClasses ? (
        <LoadingState title="Đang tải dữ liệu giáo viên" message="Hệ thống đang lấy lớp, học sinh và thống kê luyện tập." />
      ) : null}
      {error ? <ErrorState title="Chưa thể cập nhật dữ liệu luyện tập" message={error} /> : null}
      {classError ? <ErrorState title="Chưa thể tải dữ liệu lớp" message={classError} /> : null}

      {activeSection === 'overview' ? (
        <OverviewSection
          classStats={classStats}
          analytics={analytics}
          classes={classes}
          selectedClassId={selectedClassId}
          classScores={classScores}
          reviewData={reviewData}
          hasPracticeData={hasPracticeData}
          onViewClass={(classId) => {
            setSelectedClassId(classId);
            setSection('classes');
          }}
        />
      ) : null}

      {activeSection === 'classes' ? (
        <StudentsSection
          classes={classes}
          selectedClass={selectedClass}
          selectedClassId={selectedClassId}
          students={classStudents}
          loading={loadingDetail}
          workspaceTab={classWorkspaceTab}
          selectedWeek={selectedWeek}
          onWeekChange={setSelectedWeek}
          classScores={classScores}
          reviewData={reviewData}
          reviewDetail={reviewDetail}
          selectedScoreStudent={selectedScoreStudent}
          studentScores={studentScores}
          loadingScores={loadingScores}
          scoreError={scoreError}
          onChangeTab={setClassWorkspaceTab}
          onSelectClass={(classId) => {
            setSelectedClassId(classId);
            setSelectedScoreStudent(null);
            setStudentScores(null);
            setReviewDetail(null);
          }}
          onSelectStudent={setSelectedScoreStudent}
          onCloseStudentScores={() => { setSelectedScoreStudent(null); setStudentScores(null); }}
          onOpenReviewDetail={async (requestId) => {
            setReviewError(null);
            try {
              setReviewDetail(await fetchTeacherReviewRequestDetail(requestId));
            } catch (err) {
              setReviewError(err instanceof Error ? err.message : 'Chưa thể mở chi tiết yêu cầu.');
            }
          }}
          onCloseReviewDetail={() => setReviewDetail(null)}
          onSaveReviewNote={async (requestId, teacherNote) => {
            setReviewError(null);
            try {
              setReviewDetail(await addTeacherReviewNote(requestId, { teacher_note: teacherNote }));
              setReviewData(await fetchTeacherReviewRequests({ class_id: selectedClassId, q: reviewSearch, limit: 50, offset: 0 }));
            } catch (err) {
              setReviewError(err instanceof Error ? err.message : 'Chưa thể lưu ghi chú.');
            }
          }}
          onResolveReview={async (requestId, nextStatus, resolution, teacherNote) => {
            setReviewError(null);
            try {
              setReviewDetail(await resolveTeacherReviewRequest(requestId, {
                status: nextStatus,
                teacher_resolution: resolution,
                teacher_note: teacherNote,
              }));
              setReviewData(await fetchTeacherReviewRequests({ class_id: selectedClassId, q: reviewSearch, limit: 50, offset: 0 }));
            } catch (err) {
              setReviewError(err instanceof Error ? err.message : 'Chưa thể cập nhật yêu cầu.');
            }
          }}
          onRequestReviewReanalysis={async (requestId) => {
            setReviewError(null);
            try {
              setReviewDetail(await requestTeacherReviewReanalysis(requestId));
              setReviewData(await fetchTeacherReviewRequests({ class_id: selectedClassId, q: reviewSearch, limit: 50, offset: 0 }));
            } catch (err) {
              setReviewError(err instanceof Error ? err.message : 'Chưa thể yêu cầu chấm lại.');
            }
          }}
        />
      ) : null}

      {activeSection === 'reports' ? (
        <ReportsSection
          analytics={analytics}
          classStats={classStats}
          reportPeriod={reportPeriod}
          reportKind={reportKind}
          selectedWeek={selectedWeek}
          onChangePeriod={setReportPeriod}
          onChangeKind={setReportKind}
          onExport={() => exportReport(reportKind)}
        />
      ) : null}

      {activeSection === 'settings' ? (
        <UtilitySection
          icon="cog-outline"
          title="Cài đặt đang được hoàn thiện"
          message="Các tuỳ chọn nâng cao cho giáo viên sẽ được đặt tại đây. Hồ sơ cá nhân vẫn nằm trong mục Hồ sơ riêng."
        />
      ) : null}

      {activeSection === 'support' ? (
        <UtilitySection
          icon="lifebuoy"
          title="Hỗ trợ giáo viên"
          message="Tài liệu hướng dẫn và kênh hỗ trợ sẽ được bổ sung sau. Batch này không thêm dữ liệu giả cho phần hỗ trợ."
        />
      ) : null}
    </AppScreen>
  );
}

function HeaderActions({
  section,
  selectedWeek,
  onWeekChange,
  onReports,
  onExport,
}: {
  section: TeacherSection;
  selectedWeek: WeekInMonth;
  onWeekChange: (week: WeekInMonth) => void;
  onReports: () => void;
  onExport: (kind?: ReportKind) => void;
}) {
  if (section === 'classes' || section === 'settings' || section === 'support') return null;

  if (section === 'overview') {
    return (
      <View style={styles.headerActions}>
        <WeekPickerButton selectedWeek={selectedWeek} onChange={onWeekChange} />
        <ActionButton icon="file-eye-outline" label="Xem báo cáo" primary onPress={onReports} />
      </View>
    );
  }

  if (section === 'reports') {
    return (
      <View style={styles.headerActions}>
        <WeekPickerButton selectedWeek={selectedWeek} onChange={onWeekChange} />
      </View>
    );
  }

  return null;
}

function OverviewSection({
  classStats,
  analytics,
  classes,
  selectedClassId,
  classScores,
  reviewData,
  hasPracticeData,
  onViewClass,
}: {
  classStats: ReturnType<typeof summarizeClasses>;
  analytics: Required<TeacherAnalyticsResponse>;
  classes: ClassSummary[];
  selectedClassId: string | null;
  classScores: TeacherClassScoresResponse | null;
  reviewData: TeacherReviewRequestsResponse | null;
  hasPracticeData: boolean;
  onViewClass: (classId: string) => void;
}) {
  const pendingReviewCount = reviewData?.pending_count ?? 0;
  const recentRequests = reviewData?.items.slice(0, 5) ?? [];
  const selectedClassScores = selectedClassId === classScores?.class_id ? classScores : null;

  return (
    <View style={styles.sectionStack}>
      <View style={styles.metricsGrid}>
        <MetricCard icon="school-outline" label="Lớp đang dạy" value={String(classStats.totalClasses)} tone="blue" />
        <MetricCard icon="account-group-outline" label="Học sinh phụ trách" value={String(classStats.totalStudents)} tone="teal" />
        <MetricCard icon="comment-alert-outline" label="Yêu cầu chờ xử lý" value={String(pendingReviewCount)} tone="orange" />
        <MetricCard icon="microphone-outline" label="Lượt luyện tập tuần này" value={String(analytics.total_practice_sessions)} tone="red" />
      </View>

      <View style={styles.contentGrid}>
        <AppCard style={[styles.panel, styles.mainColumn]}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.cardTitle}>Cần chú ý hôm nay</Text>
              <Text style={styles.cardDescription}>Các việc có thể cần giáo viên can thiệp trước, không hiển thị chi tiết quản lý lớp tại màn tổng quan.</Text>
            </View>
          </View>
          <View style={styles.studentRows}>
            {pendingReviewCount > 0 ? (
              <View style={styles.classSummaryRow}>
                <Text style={styles.className}>{pendingReviewCount} yêu cầu xem lại đang chờ</Text>
                <Text style={styles.mutedText}>Vào Lớp học, tab Yêu cầu xem lại để xem chi tiết, ghi chú hoặc đánh dấu đã xử lý.</Text>
              </View>
            ) : null}
            {!hasPracticeData ? (
              <View style={styles.classSummaryRow}>
                <Text style={styles.className}>Chưa có dữ liệu luyện tập mới</Text>
                <Text style={styles.mutedText}>Khi học sinh hoàn thành bài luyện, hoạt động gần đây và điểm lớp sẽ hiển thị tại đây.</Text>
              </View>
            ) : null}
            {pendingReviewCount === 0 && hasPracticeData ? (
              <EmptyPanel icon="check-circle-outline" title="Chưa có việc cần xử lý ngay" message="Các yêu cầu xem lại, bài sắp đến hạn hoặc cảnh báo dữ liệu sẽ xuất hiện tại đây khi có dữ liệu thật." />
            ) : null}
          </View>
        </AppCard>

        <View style={styles.sideColumn}>
          <AppCard style={styles.panel}>
            <Text style={styles.cardTitle}>Hoạt động gần đây</Text>
            {recentRequests.length > 0 ? (
              <View style={styles.studentRows}>
                {recentRequests.map((item) => (
                  <View key={item.id} style={styles.classSummaryRow}>
                    <Text style={styles.className}>{item.student_name}</Text>
                    <Text style={styles.mutedText}>{formatReviewReason(item.reason)} - {formatReviewStatus(item.status)}</Text>
                  </View>
                ))}
              </View>
            ) : (
              <EmptyPanel icon="history" title="Chưa có hoạt động gần đây" message="Hệ thống sẽ hiển thị 3-5 hoạt động mới nhất khi có dữ liệu thật." />
            )}
          </AppCard>
        </View>
      </View>

      <AppCard style={styles.panel}>
        <View style={styles.panelHeader}>
          <View>
            <Text style={styles.cardTitle}>Tóm tắt lớp</Text>
            <Text style={styles.cardDescription}>Mỗi lớp có một card ngắn. Chi tiết học viên, điểm, bài giao và lỗi phát âm nằm trong mục Lớp học.</Text>
          </View>
        </View>
        <View style={styles.classRosterGrid}>
          {classes.map((classItem) => {
            const isSelectedClass = selectedClassId === classItem.id;
            const averageScore = isSelectedClass
              ? selectedClassScores?.students.find((student) => student.average_score !== null)?.average_score ?? null
              : null;
            const practiceCount = isSelectedClass ? selectedClassScores?.completed_count ?? 0 : null;
            return (
              <View key={classItem.id} style={styles.classCard}>
                <Text style={styles.classCode}>{classItem.code ?? 'Chưa có mã'}</Text>
                <Text style={styles.className}>{classItem.name}</Text>
                <Text style={styles.mutedText}>{classItem.student_count} học sinh</Text>
                <Text style={styles.mutedText}>Điểm TB: {averageScore === null ? 'Chưa có điểm' : `${Math.round(averageScore)}/100`}</Text>
                <Text style={styles.mutedText}>Lượt luyện: {practiceCount === null ? 'Chưa tải' : String(practiceCount)}</Text>
                <Pressable accessibilityRole="button" onPress={() => onViewClass(classItem.id)} style={styles.smallButton}>
                  <Text style={styles.smallButtonText}>Xem lớp</Text>
                </Pressable>
              </View>
            );
          })}
          {classes.length === 0 ? (
            <EmptyPanel icon="school-outline" title="Bạn chưa phụ trách lớp nào" message="Khi được thêm vào lớp, tóm tắt lớp sẽ hiển thị tại đây." />
          ) : null}
        </View>
      </AppCard>
    </View>
  );
}

function TeacherReviewRequestsSection({
  data,
  detail,
  search,
  loading,
  error,
  onSearch,
  onOpenDetail,
  onCloseDetail,
  onSaveNote,
  onResolve,
  onRequestReanalysis,
}: {
  data: TeacherReviewRequestsResponse | null;
  detail: TeacherReviewRequestDetail | null;
  search: string;
  loading: boolean;
  error: string | null;
  onSearch: (value: string) => void;
  onOpenDetail: (requestId: string) => void;
  onCloseDetail: () => void;
  onSaveNote: (requestId: string, teacherNote: string) => void;
  onResolve: (requestId: string, status: 'resolved' | 'rejected', resolution: string, teacherNote?: string | null) => void;
  onRequestReanalysis: (requestId: string) => void;
}) {
  const items = data?.items ?? [];

  return (
    <View style={styles.sectionStack}>
      <View style={styles.metricsGrid}>
        <MetricCard icon="clock-alert-outline" label="Chờ xử lý" value={String(data?.pending_count ?? 0)} tone="orange" />
        <MetricCard icon="account-alert-outline" label="Học sinh báo cáo" value={String(data?.student_report_count ?? 0)} tone="blue" />
        <MetricCard icon="robot-outline" label="Hệ thống đánh dấu" value={String(data?.system_flag_count ?? 0)} tone="teal" />
        <MetricCard icon="check-decagram-outline" label="Đã xử lý tuần này" value={String(data?.resolved_this_week_count ?? 0)} tone="red" />
      </View>

      <View style={styles.contentGrid}>
        <AppCard style={[styles.panel, styles.mainColumn]}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.cardTitle}>Yêu cầu xem lại kết quả AI</Text>
              <Text style={styles.cardDescription}>AI vẫn tự chấm bình thường. Giáo viên chỉ xử lý các kết quả được báo cáo hoặc bị đánh dấu bất thường.</Text>
            </View>
            <StatusBadge label={`${data?.total ?? 0} yêu cầu`} tone="primary" />
          </View>

          <View style={styles.filterRow}>
            <TextInput
              value={search}
              onChangeText={onSearch}
              placeholder="Tìm học sinh, lớp hoặc nội dung luyện tập"
              placeholderTextColor="#94A3B8"
              style={styles.searchInput}
            />
          </View>

          {loading ? <Text style={styles.mutedText}>Đang tải yêu cầu xem lại...</Text> : null}
          {error ? <ErrorState title="Chưa thể tải yêu cầu xem lại" message={error} /> : null}

          <View style={styles.aiReviewRows}>
            {items.map((item) => (
              <ReviewRequestRow key={item.id} item={item} onOpen={() => onOpenDetail(item.id)} />
            ))}
            {!loading && !error && items.length === 0 ? (
              <EmptyPanel
                icon="comment-check-outline"
                title="Hiện chưa có yêu cầu xem lại nào"
                message="Kết quả AI gốc không bị ghi đè. Khi học sinh báo cáo hoặc hệ thống đánh dấu bất thường, yêu cầu sẽ xuất hiện ở đây."
              />
            ) : null}
          </View>
        </AppCard>

        <ReviewDetailPanel
          detail={detail}
          onClose={onCloseDetail}
          onSaveNote={onSaveNote}
          onResolve={onResolve}
          onRequestReanalysis={onRequestReanalysis}
        />
      </View>
    </View>
  );
}

function ReviewRequestRow({ item, onOpen }: { item: TeacherReviewRequest; onOpen: () => void }) {
  return (
    <View style={styles.aiReviewRow}>
      <View style={styles.aiReviewMain}>
        <Text style={styles.studentName}>{item.student_name}</Text>
        <Text style={styles.mutedText}>{item.class_code ?? item.class_name ?? 'Chưa rõ lớp'} - {item.target_word ?? 'Bài luyện'}</Text>
        <Text style={styles.reviewReason}>{formatReviewReason(item.reason)} - {formatReviewSeverity(item.severity)}</Text>
        <Text style={styles.mutedText}>{formatReviewSource(item.source)} - {formatReviewStatus(item.status)}</Text>
      </View>
      <View style={styles.reportValueWrap}>
        <Text style={styles.reportValue}>{item.ai_score === null ? 'Chưa có điểm' : `${Math.round(item.ai_score)}%`}</Text>
        <Pressable accessibilityRole="button" onPress={onOpen} style={styles.smallButton}>
          <Text style={styles.smallButtonText}>Xem chi tiết</Text>
        </Pressable>
      </View>
    </View>
  );
}

function ReviewDetailPanel({
  detail,
  onClose,
  onSaveNote,
  onResolve,
  onRequestReanalysis,
}: {
  detail: TeacherReviewRequestDetail | null;
  onClose: () => void;
  onSaveNote: (requestId: string, teacherNote: string) => void;
  onResolve: (requestId: string, status: 'resolved' | 'rejected', resolution: string, teacherNote?: string | null) => void;
  onRequestReanalysis: (requestId: string) => void;
}) {
  const [teacherNote, setTeacherNote] = useState('');
  const [resolution, setResolution] = useState('');

  if (!detail) {
    return (
      <AppCard style={[styles.panel, styles.sideColumn]}>
        <Text style={styles.cardTitle}>Chi tiết yêu cầu</Text>
        <Text style={styles.cardDescription}>Chọn một yêu cầu để xem đề bài, kết quả AI gốc, ghi chú học sinh và audit trail.</Text>
      </AppCard>
    );
  }

  return (
    <AppCard style={[styles.panel, styles.sideColumn]}>
      <View style={styles.panelHeader}>
        <Text style={styles.cardTitle}>Chi tiết yêu cầu</Text>
        <Pressable accessibilityRole="button" onPress={onClose} style={styles.smallButton}>
          <Text style={styles.smallButtonText}>Đóng</Text>
        </Pressable>
      </View>
      <ReportRow label="Học sinh" value={detail.student_name} />
      <ReportRow label="Lớp" value={detail.class_code ?? detail.class_name ?? 'Chưa rõ'} />
      <ReportRow label="Nội dung" value={detail.target_word ?? 'Bài luyện'} />
      <ReportRow label="Điểm AI gốc" value={detail.ai_score === null ? 'Chưa có điểm' : `${Math.round(detail.ai_score)}%`} />
      <ReportRow label="Trạng thái AI" value={formatPracticeStatus(detail.practice_status)} />
      <ReportRow label="Nguồn" value={formatReviewSource(detail.source)} />
      <ReportRow label="Lý do" value={formatReviewReason(detail.reason)} />
      {detail.student_note ? <Text style={styles.cardDescription}>Ghi chú học sinh: {detail.student_note}</Text> : null}
      {detail.teacher_note ? <Text style={styles.cardDescription}>Ghi chú giáo viên: {detail.teacher_note}</Text> : null}
      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Ghi chú giáo viên</Text>
        <TextInput
          value={teacherNote}
          onChangeText={setTeacherNote}
          placeholder="Nhập ghi chú riêng cho yêu cầu này"
          placeholderTextColor="#94A3B8"
          style={styles.searchInput}
        />
      </View>
      <View style={styles.inputGroup}>
        <Text style={styles.inputLabel}>Kết luận xử lý</Text>
        <TextInput
          value={resolution}
          onChangeText={setResolution}
          placeholder="Nhập kết luận trước khi đánh dấu đã xử lý hoặc từ chối"
          placeholderTextColor="#94A3B8"
          style={styles.searchInput}
        />
      </View>
      <Text style={styles.cardTitle}>Audit trail</Text>
      {detail.audit_logs.length > 0 ? detail.audit_logs.map((log) => (
        <View key={log.id} style={styles.classSummaryRow}>
          <Text style={styles.className}>{log.action_type}</Text>
          <Text style={styles.mutedText}>{formatDate(log.created_at)}</Text>
        </View>
      )) : <Text style={styles.mutedText}>Chưa có audit log.</Text>}
      <View style={styles.disabledActionRow}>
        <ActionButton
          icon="content-save-outline"
          label="Lưu ghi chú"
          disabled={!teacherNote.trim()}
          onPress={() => onSaveNote(detail.id, teacherNote.trim())}
        />
        <ActionButton
          icon="close-circle-outline"
          label="Từ chối yêu cầu"
          disabled={!resolution.trim()}
          onPress={() => onResolve(detail.id, 'rejected', resolution.trim(), teacherNote.trim() || null)}
        />
        <ActionButton
          icon="check-circle-outline"
          label="Đánh dấu đã xử lý"
          disabled={!resolution.trim()}
          onPress={() => onResolve(detail.id, 'resolved', resolution.trim(), teacherNote.trim() || null)}
        />
        <ActionButton icon="reload" label="Yêu cầu chấm lại" onPress={() => onRequestReanalysis(detail.id)} />
      </View>
      <Text style={styles.mutedText}>Các thao tác chỉ lưu teacher review/audit log riêng, không ghi đè kết quả AI gốc.</Text>
    </AppCard>
  );
}

function StudentsSection({
  classes,
  selectedClass,
  selectedClassId,
  students,
  loading,
  workspaceTab,
  selectedWeek,
  onWeekChange,
  classScores,
  reviewData,
  reviewDetail,
  selectedScoreStudent,
  studentScores,
  loadingScores,
  scoreError,
  onChangeTab,
  onSelectClass,
  onSelectStudent,
  onCloseStudentScores,
  onOpenReviewDetail,
  onCloseReviewDetail,
  onSaveReviewNote,
  onResolveReview,
  onRequestReviewReanalysis,
}: {
  classes: ClassSummary[];
  selectedClass: ClassDetail | null;
  selectedClassId: string | null;
  students: ClassStudent[];
  loading: boolean;
  workspaceTab: ClassWorkspaceTab;
  selectedWeek: WeekInMonth;
  onWeekChange: (week: WeekInMonth) => void;
  classScores: TeacherClassScoresResponse | null;
  reviewData: TeacherReviewRequestsResponse | null;
  reviewDetail: TeacherReviewRequestDetail | null;
  selectedScoreStudent: ClassStudent | null;
  studentScores: TeacherStudentScoresResponse | null;
  loadingScores: boolean;
  scoreError: string | null;
  onChangeTab: (tab: ClassWorkspaceTab) => void;
  onSelectClass: (classId: string) => void;
  onSelectStudent: (student: ClassStudent) => void;
  onCloseStudentScores: () => void;
  onOpenReviewDetail: (requestId: string) => void;
  onCloseReviewDetail: () => void;
  onSaveReviewNote: (requestId: string, teacherNote: string) => void;
  onResolveReview: (requestId: string, status: 'resolved' | 'rejected', resolution: string, teacherNote?: string | null) => void;
  onRequestReviewReanalysis: (requestId: string) => void;
}) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const isCompact = windowWidth < 600;
  const [studentSearch, setStudentSearch] = useState('');
  const selectedClassName = selectedClass?.name ?? classes.find((item) => item.id === selectedClassId)?.name ?? 'Lớp học';
  const selectedClassCode = selectedClass?.code ?? classes.find((item) => item.id === selectedClassId)?.code ?? 'Phoenix';
  const scoredStudents = classScores?.students.filter((student) => student.average_score !== null) ?? [];
  const averageScore = scoredStudents.length
    ? Math.round(scoredStudents.reduce((sum, student) => sum + (student.average_score ?? 0), 0) / scoredStudents.length)
    : null;
  const filteredStudents = students.filter((student) => {
    const needle = studentSearch.trim().toLowerCase();
    if (!needle) return true;
    return student.display_name.toLowerCase().includes(needle) || (student.email ?? '').toLowerCase().includes(needle);
  });

  return (
    <View style={styles.classWorkspace}>
      <View style={styles.classWorkspaceTopbar}>
        {classes.length > 0 ? (
          <ClassPicker
            classes={classes}
            selectedClassId={selectedClassId}
            pendingReviewCount={reviewData?.pending_count ?? 0}
            onSelectClass={onSelectClass}
          />
        ) : (
          <EmptyPanel icon="school-outline" title="Bạn chưa phụ trách lớp nào" message="Khi bạn được thêm vào lớp, workspace lớp học sẽ hiển thị tại đây." />
        )}
      </View>

      <View style={styles.classBreadcrumb}>
        <Text style={styles.breadcrumbText}>Lớp học</Text>
        <Text style={styles.breadcrumbSeparator}>›</Text>
        <Text style={styles.breadcrumbTextActive}>{selectedClassCode}</Text>
      </View>

      <View style={styles.classWorkspaceHeader}>
        <View>
          <Text style={styles.workspaceTitle}>Quản lý lớp học</Text>
          <Text style={styles.workspaceSubtitle}>{selectedClassName}</Text>
        </View>
        {selectedClassId ? (
          <View style={styles.classHeaderStats}>
            <Text style={styles.classHeaderStatLine}>
              {students.length} học sinh · Điểm TB: {averageScore === null ? 'Chưa có điểm' : `${averageScore}/100`} · {reviewData?.pending_count ?? 0} yêu cầu chờ
            </Text>
          </View>
        ) : null}
      </View>

      {selectedClassId ? (
        <View style={styles.classTabBar}>
          <ClassTabButton icon="account-group-outline" label="Học viên" active={workspaceTab === 'students'} onPress={() => onChangeTab('students')} />
          <ClassTabButton icon="star-outline" label="Điểm" active={workspaceTab === 'scores'} onPress={() => onChangeTab('scores')} />
          <ClassTabButton icon="clipboard-text-outline" label="Bài giao" active={workspaceTab === 'assignments'} onPress={() => onChangeTab('assignments')} />
          <ClassTabButton
            icon="comment-alert-outline"
            label="Yêu cầu xem lại"
            active={workspaceTab === 'reviews'}
            badgeCount={reviewData?.pending_count ?? 0}
            onPress={() => onChangeTab('reviews')}
          />
        </View>
      ) : null}

      {selectedClassId && workspaceTab === 'students' ? (
        <View style={styles.studentTabStack}>
          <View style={styles.studentToolbar}>
            <TextInput
              value={studentSearch}
              onChangeText={setStudentSearch}
              placeholder="Tìm theo tên hoặc email"
              placeholderTextColor="#94A3B8"
              style={styles.studentSearchInput}
            />
            <StatusBadge label={`${filteredStudents.length}/${students.length} học sinh`} tone="primary" />
          </View>

          <AppCard style={[styles.panel, styles.studentListPanel]}>
            <View style={styles.panelHeader}>
              <View>
                <Text style={styles.cardTitle}>Học viên</Text>
                <Text style={styles.cardDescription}>Danh sách học sinh trong lớp đang chọn.</Text>
              </View>
              <StatusBadge label={`${students.length} học sinh`} tone="primary" />
            </View>
            {loading ? <Text style={styles.mutedText}>Đang tải học sinh...</Text> : null}
            <View style={styles.studentRows}>
              {filteredStudents.map((student) => (
                <StudentRosterRow
                  key={student.id}
                  student={student}
                  scoreSummary={classScores?.students.find((item) => item.student_id === student.id)}
                  onViewScores={() => onSelectStudent(student)}
                />
              ))}
              {!loading && filteredStudents.length === 0 ? (
                <EmptyPanel icon="account-search-outline" title="Chưa có học sinh trong lớp" message="Khi lớp có thành viên hoạt động, danh sách học sinh sẽ hiển thị tại đây." />
              ) : null}
            </View>
          </AppCard>
        </View>
      ) : null}

      {selectedClassId && workspaceTab === 'scores' ? (
        <View style={styles.sectionStack}>
          <ClassWorkspaceOverview
            students={students}
            classScores={classScores}
            classId={selectedClassId}
          />
        </View>
      ) : null}

      {selectedClassId && workspaceTab === 'assignments' ? (
        <AssignmentSection
          classId={selectedClassId}
          className={selectedClass?.name || ''}
          students={students}
          selectedWeek={selectedWeek}
          onWeekChange={onWeekChange}
          onExport={() => undefined}
        />
      ) : null}

      {selectedClassId && workspaceTab === 'reviews' ? (
        <View style={styles.contentGrid}>
          <AppCard style={[styles.panel, styles.mainColumn]}>
            <Text style={styles.cardTitle}>Yêu cầu xem lại của lớp</Text>
            <Text style={styles.cardDescription}>Chỉ hiển thị request thuộc lớp đang chọn.</Text>
            <View style={styles.aiReviewRows}>
              {(reviewData?.items ?? []).map((item) => (
                <ReviewRequestRow key={item.id} item={item} onOpen={() => onOpenReviewDetail(item.id)} />
              ))}
              {(reviewData?.items ?? []).length === 0 ? (
                <EmptyPanel icon="comment-check-outline" title="Lớp này chưa có yêu cầu xem lại nào." message="Khi học sinh báo cáo kết quả AI, yêu cầu sẽ xuất hiện tại đây." />
              ) : null}
            </View>
          </AppCard>
          <ReviewDetailPanel
            detail={reviewDetail}
            onClose={onCloseReviewDetail}
            onSaveNote={onSaveReviewNote}
            onResolve={onResolveReview}
            onRequestReanalysis={onRequestReviewReanalysis}
          />
        </View>
      ) : null}

      <Modal visible={!!selectedScoreStudent} transparent animationType="fade" onRequestClose={onCloseStudentScores}>
        <KeyboardAvoidingView style={styles.modalKeyboardRoot} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <TouchableWithoutFeedback onPress={onCloseStudentScores}>
            <View style={styles.studentDetailBackdrop}>
              <TouchableWithoutFeedback>
                <View style={[styles.studentDetailModal, {
                  width: Math.max(0, Math.min(600, windowWidth - 24)),
                  maxHeight: Math.max(0, windowHeight - 24),
                  padding: isCompact ? 16 : 24,
                }]}>
                  <View style={styles.modalHeader}>
                    <Text style={styles.cardTitle}>{selectedScoreStudent?.display_name}</Text>
                    <Pressable accessibilityRole="button" accessibilityLabel="Đóng chi tiết học sinh" onPress={onCloseStudentScores} style={styles.modalCloseBtn}>
                      <MaterialCommunityIcons name="close" size={20} color={colors.muted} />
                    </Pressable>
                  </View>
                  {!isInternalEmail(selectedScoreStudent?.email) ? <Text style={styles.cardDescription}>{selectedScoreStudent?.email ?? 'Chưa có email'}</Text> : null}
                  <ScrollView keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
                    {loadingScores ? <Text style={styles.mutedText}>Đang tải kết quả luyện tập...</Text> : null}
                    {scoreError ? <ErrorState title="Chưa thể tải điểm" message={scoreError} /> : null}
                    {studentScores ? (
                      <View style={styles.reportRows}>
                        <ReportRow label="Tổng lượt luyện" value={String(studentScores.summary.total_attempts)} />
                        <ReportRow label="Đã có kết quả" value={String(studentScores.summary.completed_attempts)} />
                        <ReportRow label="Điểm trung bình" value={studentScores.summary.average_score === null ? 'Chưa có điểm' : `${Math.round(studentScores.summary.average_score)}/100`} />
                        <ReportRow label="Lần luyện gần nhất" value={formatDate(studentScores.summary.last_practice_at)} />
                      </View>
                    ) : null}
                    {studentScores ? (
                      <View style={styles.assignmentDetailSection}>
                        <Text style={styles.cardTitle}>Lỗi phát âm cần chú ý</Text>
                        {buildStudentPronunciationIssues(studentScores).length > 0 ? (
                          <View style={styles.errorRows}>
                            {buildStudentPronunciationIssues(studentScores).map((issue) => (
                              <View key={issue.label} style={styles.errorRow}>
                                <View style={styles.pronunciationIssueCopy}>
                                  <Text style={styles.errorLabel}>{formatMistakeLabel(issue.label)}</Text>
                                  <Text style={styles.mutedText}>{issue.count} lần xuất hiện</Text>
                                  {issue.examples.length > 0 ? <Text style={styles.mutedText}>Ví dụ: {issue.examples.slice(0, 2).join(', ')}</Text> : null}
                                  <Text style={styles.mutedText}>Nên luyện lại với tốc độ chậm và nghe mẫu trước khi ghi âm.</Text>
                                </View>
                                <StatusBadge label={mistakeSeverity(issue.count)} tone={issue.count >= 10 ? 'error' : issue.count >= 4 ? 'warning' : 'primary'} />
                              </View>
                            ))}
                          </View>
                        ) : (
                          <EmptyPanel icon="microphone-off" title="Chưa có đủ dữ liệu lỗi phát âm." message="Phoenix sẽ hiển thị lỗi khi học sinh có kết quả luyện tập hoàn tất." />
                        )}
                      </View>
                    ) : null}
                    {studentScores?.items && studentScores.items.length > 0 ? (
                      <View style={styles.studentRows}>
                        {studentScores.items.map((item) => (
                          <View key={item.practice_history_id} style={styles.classSummaryRow}>
                            <Text style={styles.className}>{item.target_word ?? 'Bài luyện'}</Text>
                            <Text style={styles.mutedText}>{formatPracticeStatus(item.status)}</Text>
                            <Text style={styles.reportValue}>{item.score === null ? 'Chưa có điểm' : `${Math.round(item.score)}/100`}</Text>
                            {item.needs_review ? <StatusBadge label="Cần xem lại" tone="warning" /> : null}
                          </View>
                        ))}
                      </View>
                    ) : null}
                  </ScrollView>
                </View>
              </TouchableWithoutFeedback>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </Modal>
    </View>
  );
}

function ClassWorkspaceOverview({
  students,
  classScores,
  classId,
}: {
  students: ClassStudent[];
  classScores: TeacherClassScoresResponse | null;
  classId: string;
}) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const useScrollableScoreTable = windowWidth < 1024;
  const isCompact = windowWidth < 600;
  const [modalStudent, setModalStudent] = useState<ClassStudent | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalScores, setModalScores] = useState<TeacherStudentScoresResponse | null>(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  useEffect(() => {
    if (!modalStudent) return;
    setModalScores(null);
    setModalError(null);
    setModalLoading(true);
    fetchTeacherStudentScores(modalStudent.id, { class_id: classId })
      .then(setModalScores)
      .catch((err) => setModalError(err instanceof Error ? err.message : 'Chưa thể tải điểm.'))
      .finally(() => setModalLoading(false));
  }, [modalStudent, classId]);

  const handleCloseModal = () => {
    setModalVisible(false);
    setModalStudent(null);
    setModalScores(null);
    setModalError(null);
  };

  const scoredStudents = classScores?.students.filter((student) => student.average_score !== null) ?? [];
  const averageScore = scoredStudents.length
    ? Math.round(scoredStudents.reduce((sum, student) => sum + (student.average_score ?? 0), 0) / scoredStudents.length)
    : null;
  const needsImprovementCount = scoredStudents.filter((student) => (student.average_score ?? 0) < 70).length;
  const notPracticedCount = (classScores?.students ?? []).filter((s) => s.completed_attempts === 0).length;
  const studentsById = new Map(students.map((student) => [student.id, student]));

  return (
    <AppCard style={styles.panel}>
      <View style={styles.panelHeader}>
        <View>
          <Text style={styles.cardTitle}>Điểm học sinh</Text>
          <Text style={styles.cardDescription}>Theo dõi điểm phát âm đã có kết quả của từng học sinh trong lớp.</Text>
        </View>
      </View>
      <View style={styles.classScoreSummary}>
        <View style={[styles.scoreSummaryItem, styles.scoreSummaryItemLarge]}>
          <Text style={styles.metricValue}>{averageScore === null ? '--' : `${averageScore}/100`}</Text>
          <Text style={styles.metricLabel}>Điểm TB lớp</Text>
        </View>
        <View style={[styles.scoreSummaryItem, styles.scoreSummaryItemSmall, needsImprovementCount > 0 ? styles.scoreSummaryItemRed : null]}>
          <Text style={[styles.metricValue, needsImprovementCount > 0 ? styles.metricValueRed : null]}>{needsImprovementCount}</Text>
          <Text style={[styles.metricLabel, needsImprovementCount > 0 ? styles.metricLabelRed : null]}>Cần cải thiện</Text>
        </View>
        <View style={[styles.scoreSummaryItem, styles.scoreSummaryItemSmall, notPracticedCount > 0 ? styles.scoreSummaryItemOrange : null]}>
          <Text style={[styles.metricValue, notPracticedCount > 0 ? styles.metricValueOrange : null]}>{notPracticedCount}</Text>
          <Text style={[styles.metricLabel, notPracticedCount > 0 ? styles.metricLabelOrange : null]}>Chưa luyện</Text>
        </View>
      </View>
      <Text style={styles.cardTitle}>Bảng điểm theo học sinh</Text>
      <ScrollView
        horizontal={useScrollableScoreTable}
        showsHorizontalScrollIndicator={useScrollableScoreTable}
        contentContainerStyle={useScrollableScoreTable ? styles.scoreTableScrollContent : undefined}
      >
      <View style={[styles.studentRows, useScrollableScoreTable ? styles.scoreTableScrollable : null]}>
        <View style={styles.scoreTableHeader}>
          <View style={{ width: 40 }} />
          <View style={styles.scoreColName}>
            <Text style={styles.scoreTableHeaderCell}>Học sinh</Text>
          </View>
          <View style={styles.scoreColScore}>
            <Text style={[styles.scoreTableHeaderCell, { textAlign: 'center' }]}>Điểm TB</Text>
          </View>
          <View style={styles.scoreColComment}>
            <Text style={[styles.scoreTableHeaderCell, { textAlign: 'center' }]}>Nhận xét</Text>
          </View>
          <View style={styles.scoreColBtn} />
        </View>
        {(classScores?.students ?? []).map((student) => (
          <View key={student.student_id} style={styles.scoreTableRow}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{student.student_name.slice(0, 1).toUpperCase()}</Text>
            </View>
            <View style={styles.scoreColName}>
              <Text style={styles.studentName}>{student.student_name}</Text>
              {!isInternalEmail(student.student_email) ? <Text style={styles.mutedText}>{student.student_email ?? 'Chưa có email'}</Text> : null}
            </View>
            <View style={styles.scoreColScore}>
              <View style={[
                styles.scorePill,
                student.average_score === null ? styles.scorePillNeutral
                  : student.average_score >= 80 ? styles.scorePillGreen
                  : student.average_score >= 60 ? styles.scorePillYellow
                  : styles.scorePillRed,
              ]}>
                <Text style={[styles.scorePillText, student.average_score !== null && student.average_score < 60 ? styles.scorePillTextRed : null]}>
                  {student.average_score === null ? '--' : `${Math.round(student.average_score)}/100`}
                </Text>
              </View>
            </View>
            <View style={styles.scoreColComment}>
              <Text style={[styles.reportValue, { textAlign: 'center' }]}>{scoreComment(student.average_score)}</Text>
            </View>
            <View style={styles.scoreColBtn}>
              <Pressable
                accessibilityRole="button"
                onPress={() => {
                  const matchedStudent = studentsById.get(student.student_id);
                  if (matchedStudent) { setModalStudent(matchedStudent); setModalVisible(true); }
                }}
                disabled={!studentsById.has(student.student_id)}
                style={[styles.smallButton, !studentsById.has(student.student_id) ? styles.disabledAction : null]}
              >
                <Text style={styles.smallButtonText}>Xem chi tiết</Text>
              </Pressable>
            </View>
          </View>
        ))}
        {!classScores || classScores.students.length === 0 ? (
          <EmptyPanel icon="chart-box-outline" title="Chưa có dữ liệu điểm của lớp" message="Khi học sinh hoàn thành bài luyện và Phoenix trả score, kết quả sẽ hiển thị tại đây." />
        ) : null}
      </View>
      </ScrollView>
      <Modal visible={modalVisible} transparent animationType="fade" onRequestClose={handleCloseModal}>
        <KeyboardAvoidingView style={styles.modalKeyboardRoot} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <TouchableWithoutFeedback onPress={handleCloseModal}>
            <View style={styles.studentDetailBackdrop}>
              <TouchableWithoutFeedback>
                <View style={[styles.studentDetailModal, {
                  width: Math.max(0, Math.min(600, windowWidth - 24)),
                  maxHeight: Math.max(0, windowHeight - 24),
                  padding: isCompact ? 16 : 24,
                }]}>
                <View style={styles.modalHeader}>
                  <Text style={styles.cardTitle}>{modalStudent?.display_name}</Text>
                  <Pressable accessibilityRole="button" accessibilityLabel="Đóng chi tiết học sinh" onPress={handleCloseModal} style={styles.modalCloseBtn}>
                    <MaterialCommunityIcons name="close" size={20} color={colors.muted} />
                  </Pressable>
                </View>
                {!isInternalEmail(modalStudent?.email) ? <Text style={styles.cardDescription}>{modalStudent?.email ?? 'Chưa có email'}</Text> : null}
                <ScrollView keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
                  {modalLoading ? <Text style={styles.mutedText}>Đang tải kết quả luyện tập...</Text> : null}
                  {modalError ? <ErrorState title="Chưa thể tải điểm" message={modalError} /> : null}
                  {modalScores ? (
                    <View style={styles.reportRows}>
                      <ReportRow label="Tổng lượt luyện" value={String(modalScores.summary.total_attempts)} />
                      <ReportRow label="Đã có kết quả" value={String(modalScores.summary.completed_attempts)} />
                      <ReportRow label="Điểm trung bình" value={modalScores.summary.average_score === null ? 'Chưa có điểm' : `${Math.round(modalScores.summary.average_score)}/100`} />
                      <ReportRow label="Lần luyện gần nhất" value={formatDate(modalScores.summary.last_practice_at)} />
                    </View>
                  ) : null}
                  {modalScores ? (
                    <View style={styles.assignmentDetailSection}>
                      <Text style={styles.cardTitle}>Lỗi phát âm cần chú ý</Text>
                      {buildStudentPronunciationIssues(modalScores).length > 0 ? (
                        <View style={styles.errorRows}>
                          {buildStudentPronunciationIssues(modalScores).map((issue) => (
                            <View key={issue.label} style={styles.errorRow}>
                              <View style={styles.pronunciationIssueCopy}>
                                <Text style={styles.errorLabel}>{formatMistakeLabel(issue.label)}</Text>
                                <Text style={styles.mutedText}>{issue.count} lần xuất hiện</Text>
                                {issue.examples.length > 0 ? <Text style={styles.mutedText}>Ví dụ: {issue.examples.slice(0, 2).join(', ')}</Text> : null}
                              </View>
                              <StatusBadge label={mistakeSeverity(issue.count)} tone={issue.count >= 10 ? 'error' : issue.count >= 4 ? 'warning' : 'primary'} />
                            </View>
                          ))}
                        </View>
                      ) : (
                        <EmptyPanel icon="microphone-off" title="Chưa có đủ dữ liệu lỗi phát âm." message="Phoenix sẽ hiển thị lỗi khi học sinh có kết quả luyện tập hoàn tất." />
                      )}
                    </View>
                  ) : null}
                  {modalScores?.items && modalScores.items.length > 0 ? (
                    <View style={styles.studentRows}>
                      {modalScores.items.map((item) => (
                        <View key={item.practice_history_id} style={styles.classSummaryRow}>
                          <Text style={styles.className}>{item.target_word ?? 'Bài luyện'}</Text>
                          <Text style={styles.mutedText}>{formatPracticeStatus(item.status)}</Text>
                          <Text style={styles.reportValue}>{item.score === null ? 'Chưa có điểm' : `${Math.round(item.score)}/100`}</Text>
                          {item.needs_review ? <StatusBadge label="Cần xem lại" tone="warning" /> : null}
                        </View>
                      ))}
                    </View>
                  ) : null}
                </ScrollView>
                </View>
              </TouchableWithoutFeedback>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </Modal>
    </AppCard>
  );
}

function ClassAnalyticsSection({
  analytics,
  classStats,
  classes,
  classScores,
  aiReviewItems,
  hasPracticeData,
  selectedWeek,
}: {
  analytics: Required<TeacherAnalyticsResponse>;
  classStats: ReturnType<typeof summarizeClasses>;
  classes: ClassSummary[];
  classScores: TeacherClassScoresResponse | null;
  aiReviewItems: TeacherReviewRequest[];
  hasPracticeData: boolean;
  selectedWeek: WeekInMonth;
}) {
  return (
    <View style={styles.sectionStack}>
      <View style={styles.metricsGrid}>
        <MetricCard icon="school-outline" label="Tổng số lớp" value={String(classStats.totalClasses)} tone="blue" />
        <MetricCard icon="account-group-outline" label="Học sinh đang phụ trách" value={String(classStats.totalStudents)} tone="teal" />
        <MetricCard icon="microphone-outline" label="Dữ liệu luyện tập" value={String(analytics.total_practice_sessions)} tone="orange" />
        <MetricCard icon="alert-octagon-outline" label="AI cần kiểm tra" value={String(aiReviewItems.length)} tone="red" />
      </View>

      <View style={styles.contentGrid}>
        <View style={styles.mainColumn}>
          <AppCard style={styles.panel}>
            <View style={styles.panelHeader}>
              <View>
                <Text style={styles.cardTitle}>Phân tích AI Phoenix</Text>
                <Text style={styles.cardDescription}>Khoảng phân tích: {selectedWeek.label} - {formatDateRange(selectedWeek)}. Các chỉ số chỉ dùng dữ liệu luyện tập thật từ backend.</Text>
              </View>
            </View>
            {hasPracticeData ? <ProgressDistribution analytics={analytics} /> : (
              <EmptyPanel
                icon="brain"
                title="Chưa có dữ liệu chẩn đoán AI"
                message="Khi học sinh hoàn thành bài luyện, kết quả Phoenix sẽ được hiển thị tại đây."
              />
            )}
          </AppCard>
          <AiReviewSection items={aiReviewItems} />
        </View>
        <View style={styles.sideColumn}>
          <AppCard style={styles.panel}>
            <Text style={styles.cardTitle}>Theo lớp</Text>
            {classScores ? (
              <View style={styles.reportRows}>
                <ReportRow label="Đã có kết quả" value={String(classScores.completed_count)} />
                <ReportRow label="Đang chẩn đoán" value={String(classScores.pending_count)} />
                <ReportRow label="Xử lý lỗi" value={String(classScores.failed_count)} />
              </View>
            ) : null}
            <View style={styles.classRows}>
              {classes.map((classItem) => (
                <View key={classItem.id} style={styles.classSummaryRow}>
                  <Text style={styles.classCode}>{classItem.code ?? 'Chưa có mã'}</Text>
                  <Text style={styles.className}>{classItem.name}</Text>
                  <Text style={styles.mutedText}>{classItem.student_count} học sinh, {classItem.teacher_count} giảng viên</Text>
                </View>
              ))}
            </View>
          </AppCard>
          <AiReviewCriteriaCard />
        </View>
      </View>
    </View>
  );
}

function AiReviewSection({ items }: { items: TeacherReviewRequest[] }) {
  return (
    <AppCard style={styles.panel}>
      <View style={styles.panelHeader}>
        <View>
          <Text style={styles.cardTitle}>Kết quả AI cần kiểm tra</Text>
          <Text style={styles.cardDescription}>Teacher review là lớp nhận xét riêng, không ghi đè kết quả AI gốc.</Text>
        </View>
        <StatusBadge label={`${items.length} mục`} tone={items.length > 0 ? 'warning' : 'primary'} />
      </View>
      {items.length > 0 ? (
        <View style={styles.aiReviewRows}>
          {items.map((item) => (
            <View key={item.id} style={styles.aiReviewRow}>
              <View style={styles.aiReviewMain}>
                <Text style={styles.studentName}>{item.student_name}</Text>
                <Text style={styles.mutedText}>{item.class_code ?? item.class_name ?? 'Chưa rõ lớp'} - {item.target_word ?? 'Bài luyện'}</Text>
                <Text style={styles.mutedText}>{formatDate(item.created_at)} - {formatReviewStatus(item.status)}</Text>
                <Text style={styles.reviewReason}>{formatReviewReason(item.reason)}</Text>
              </View>
              <Pressable accessibilityRole="button" disabled style={[styles.smallButton, styles.disabledAction]}>
                <Text style={styles.smallButtonText}>Sắp có</Text>
              </Pressable>
            </View>
          ))}
        </View>
      ) : (
        <EmptyPanel
          icon="clipboard-search-outline"
          title="Chưa có dữ liệu chẩn đoán AI"
          message="Khi học sinh hoàn thành bài luyện, kết quả Phoenix sẽ được hiển thị tại đây."
        />
      )}
      <View style={styles.disabledActionRow}>
        <ActionButton icon="flag-outline" label="Đánh dấu cần kiểm tra" disabled onPress={() => undefined} />
        <ActionButton icon="note-edit-outline" label="Thêm ghi chú" disabled onPress={() => undefined} />
        <ActionButton icon="refresh" label="Yêu cầu phân tích lại" disabled onPress={() => undefined} />
      </View>
    </AppCard>
  );
}

function AiReviewCriteriaCard() {
  const reasons = [
    'confidence thấp',
    'job failed',
    'processing quá lâu',
    'điểm quá thấp bất thường',
    'audio lỗi hoặc thiếu',
    'text similarity thấp',
    'lỗi phát âm lặp lại nhiều lần',
  ];

  return (
    <AppCard style={styles.panel}>
      <Text style={styles.cardTitle}>Tiêu chí cần kiểm tra</Text>
      <View style={styles.pillList}>
        {reasons.map((reason) => (
          <View key={reason} style={styles.infoPill}>
            <Text style={styles.infoPillText}>{reason}</Text>
          </View>
        ))}
      </View>
    </AppCard>
  );
}

function PronunciationErrorsSection({ errors, onExport }: { errors: TeacherCommonError[]; onExport: () => void }) {
  const systemCategories = ['Âm cuối', 'Độ dài nguyên âm', 'Trọng âm từ', 'Cụm phụ âm', 'Nối âm'];

  return (
    <View style={styles.contentGrid}>
      <View style={styles.mainColumn}>
        <AppCard style={styles.panel}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.cardTitle}>Thống kê lỗi phát âm</Text>
              <Text style={styles.cardDescription}>Chỉ thống kê lỗi học sinh mắc phải, không duyệt từng bài AI tại trang này.</Text>
            </View>
            <ActionButton icon="download-outline" label="Xuất báo cáo lỗi" primary onPress={onExport} />
          </View>
          {errors.length > 0 ? <CommonErrors errors={errors} /> : (
            <EmptyPanel
              icon="alert-circle-outline"
              title="Chưa có dữ liệu lỗi phát âm"
              message="Các lỗi phổ biến sẽ xuất hiện sau khi học sinh luyện tập."
            />
          )}
        </AppCard>
      </View>
      <View style={styles.sideColumn}>
        <AppCard style={styles.panel}>
          <Text style={styles.cardTitle}>Danh mục lỗi hệ thống</Text>
          <View style={styles.pillList}>
            {systemCategories.map((label) => (
              <View key={label} style={styles.infoPill}>
                <Text style={styles.infoPillText}>{label}</Text>
              </View>
            ))}
          </View>
        </AppCard>
      </View>
    </View>
  );
}

function AssignmentSection({
  classId,
  className,
  students,
  selectedWeek,
  onWeekChange,
  onExport,
}: {
  classId: string;
  className: string;
  students: ClassStudent[];
  selectedWeek: WeekInMonth;
  onWeekChange: (week: WeekInMonth) => void;
  onExport: () => void;
}) {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const isCompact = windowWidth < 600;
  const stackAssignmentWorkspace = windowWidth < 1024;
  const [assignmentStudentSearch, setAssignmentStudentSearch] = useState('');
  const [selectedAssignmentStudentId, setSelectedAssignmentStudentId] = useState<string | null>(null);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [assignmentsLoading, setAssignmentsLoading] = useState(false);
  const [gradebookAssignment, setGradebookAssignment] = useState<Assignment | null>(null);
  const [gradebook, setGradebook] = useState<AssignmentGradebookItem[]>([]);
  const [gradebookLoading, setGradebookLoading] = useState(false);
  const [gradebookError, setGradebookError] = useState<string | null>(null);
  const [gradebookStudent, setGradebookStudent] = useState<AssignmentGradebookItem | null>(null);
  const [overrideScore, setOverrideScore] = useState('');
  const [overrideReason, setOverrideReason] = useState('');
  const [overrideLoading, setOverrideLoading] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [vocabSets, setVocabSets] = useState<VocabularySet[]>([]);
  const [vocabSetsLoading, setVocabSetsLoading] = useState(false);
  const [createTitle, setCreateTitle] = useState('');
  const [createSetId, setCreateSetId] = useState<string | null>(null);
  const [createTarget, setCreateTarget] = useState<'class' | 'student'>('class');
  const [createDueDate, setCreateDueDate] = useState('');
  const [createIsAssessment, setCreateIsAssessment] = useState(false);
  const [createTimer, setCreateTimer] = useState('60');
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState<string | null>(null);
  const [createAssessmentDate, setCreateAssessmentDate] = useState('');
  const [createAssessmentTime, setCreateAssessmentTime] = useState('23:59');

  useEffect(() => {
    void loadAssignments(selectedAssignmentStudentId);
  }, [classId]);

  useEffect(() => {
    if (!createSuccess) return;
    const timer = setTimeout(() => setCreateSuccess(null), 3000);
    return () => clearTimeout(timer);
  }, [createSuccess]);

  const loadAssignments = async (studentId?: string | null) => {
    setAssignmentsLoading(true);
    try {
      const [classData, studentData] = await Promise.all([
        fetchTeacherAssignments({ class_id: classId }),
        studentId ? fetchTeacherAssignments({ student_id: studentId }) : Promise.resolve({ items: [], total: 0 }),
      ]);
      const seen = new Set<string>();
      const merged = [...classData.items, ...studentData.items].filter((a) => {
        if (seen.has(a.id)) return false;
        seen.add(a.id);
        return true;
      });
      setAssignments(merged);
    } catch {
      // silently fail; teacher sees empty list
    } finally {
      setAssignmentsLoading(false);
    }
  };

  const openGradebook = async (assignment: Assignment) => {
    setGradebookAssignment(assignment);
    setGradebookStudent(null);
    setGradebookLoading(true);
    setGradebookError(null);
    try {
      setGradebook(await fetchAssignmentGradebook(assignment.id));
    } catch (err) {
      setGradebookError(err instanceof Error ? err.message : 'Không thể tải bảng điểm bài giao.');
    } finally {
      setGradebookLoading(false);
    }
  };

  const submitGradeOverride = async () => {
    if (!gradebookAssignment || !gradebookStudent || !overrideReason.trim()) return;
    const score = Number(overrideScore);
    if (!Number.isFinite(score) || score < 0 || score > 100) return;
    setOverrideLoading(true);
    try {
      await overrideAssignmentGrade(gradebookAssignment.id, gradebookStudent.student_id, score, overrideReason.trim());
      await openGradebook(gradebookAssignment);
      setGradebookStudent(null);
    } finally {
      setOverrideLoading(false);
    }
  };

  const openCreate = async () => {
    setShowCreate(true);
    setCreateTitle('');
    setCreateSetId(null);
    setCreateTarget(selectedAssignmentStudentId ? 'student' : 'class');
    setCreateDueDate('');
    setCreateIsAssessment(false);
    setCreateAssessmentDate('');
    setCreateAssessmentTime('');
    setCreateTimer('60');
    setCreateError(null);
    if (vocabSets.length === 0) {
      setVocabSetsLoading(true);
      try {
        const data = await fetchVocabularySets(50, 0);
        setVocabSets(data.items);
        if (data.items[0]) setCreateSetId(data.items[0].id);
      } catch {
        // silently fail
      } finally {
        setVocabSetsLoading(false);
      }
    } else {
      if (vocabSets[0]) setCreateSetId(vocabSets[0].id);
    }
  };

  const submitCreate = async () => {
    if (!createTitle.trim() || !createSetId) return;
    setCreateLoading(true);
    setCreateError(null);
    try {
      let deadline: string | null = null;
      if (createIsAssessment && createAssessmentDate) {
        const isoDate = parseDDMMYYYY(createAssessmentDate);
        const timeStr = createAssessmentTime.trim() || '23:59';
        if (!isoDate) {
          setCreateError('Ngày không hợp lệ. Vui lòng nhập theo định dạng DD/MM/YYYY.');
          return;
        }
        deadline = `${isoDate}T${timeStr}:00+07:00`;
      }
      const dueDateIso = !createIsAssessment && createDueDate.trim()
        ? parseDDMMYYYY(createDueDate.trim())
        : null;
      await createAssignment({
        title: createTitle.trim(),
        content_type: 'vocabulary_set',
        content_id: createSetId,
        class_id: createTarget === 'class' ? classId : null,
        student_id: createTarget === 'student' ? selectedAssignmentStudentId : null,
        due_date: dueDateIso,
        is_assessment: createIsAssessment,
        deadline,
        timer_per_word_seconds: createIsAssessment ? (parseInt(createTimer, 10) || 60) : 60,
      });
      setShowCreate(false);
      await loadAssignments(selectedAssignmentStudentId);
      const targetName = createTarget === 'class' ? className : (selectedStudent?.display_name || 'học sinh');
      setCreateSuccess(`Đã giao bài cho ${targetName}`);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Không thể tạo bài tập.');
    } finally {
      setCreateLoading(false);
    }
  };

  const filteredAssignmentStudents = students.filter((student) => {
    const needle = assignmentStudentSearch.trim().toLowerCase();
    if (!needle) return true;
    return student.display_name.toLowerCase().includes(needle) || (student.email ?? '').toLowerCase().includes(needle);
  });
  const selectedStudent = students.find((s) => s.id === selectedAssignmentStudentId) ?? null;

  const visibleAssignments = selectedAssignmentStudentId
    ? assignments.filter((a) => a.class_id === classId || a.student_id === selectedAssignmentStudentId)
    : assignments;

  return (
    <View style={styles.assignmentPanel}>
      <View style={styles.assignmentToolbar}>
        <View>
          <Text style={styles.cardTitle}>Bài giao</Text>
          <Text style={styles.cardDescription}>Chọn học sinh để xem bài đã giao và chuẩn bị bài luyện phù hợp.</Text>
        </View>
        <WeekPickerButton selectedWeek={selectedWeek} onChange={onWeekChange} />
        <View style={styles.assignmentToolbarActions}>
          <ActionButton icon="download-outline" label="Xuất danh sách" onPress={onExport} />
          <ActionButton icon="plus" label="Giao bài mới" primary onPress={openCreate} />
        </View>
      </View>

      {createSuccess ? (
        <View style={styles.successBanner}>
          <MaterialCommunityIcons name="check-circle" size={20} color={colors.success} />
          <Text style={styles.successBannerText}>{createSuccess}</Text>
        </View>
      ) : null}

      <View style={[styles.assignmentWorkspaceGrid, stackAssignmentWorkspace ? styles.assignmentWorkspaceGridStacked : null]}>
        <AppCard style={[styles.panel, styles.assignmentStudentColumn]}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.cardTitle}>Danh sách học sinh</Text>
              <Text style={styles.cardDescription}>Chọn một học sinh để xem bài đã giao.</Text>
            </View>
            <StatusBadge label={`${students.length} học sinh`} tone="primary" />
          </View>
          <TextInput
            value={assignmentStudentSearch}
            onChangeText={setAssignmentStudentSearch}
            placeholder="Tìm học sinh..."
            placeholderTextColor="#94A3B8"
            style={styles.assignmentSearchInput}
          />
          <View style={styles.assignmentStudentList}>
            {filteredAssignmentStudents.map((student) => {
              const count = assignments.filter(
                (a) => a.class_id === classId || a.student_id === student.id
              ).length;
              return (
                <Pressable
                  key={student.id}
                  accessibilityRole="button"
                  onPress={() => setSelectedAssignmentStudentId(student.id)}
                  style={[
                    styles.assignmentStudentRow,
                    selectedAssignmentStudentId === student.id ? styles.assignmentStudentRowActive : null,
                  ]}
                >
                  <View style={styles.avatar}>
                    <Text style={styles.avatarText}>{student.display_name.slice(0, 1).toUpperCase()}</Text>
                  </View>
                  <View style={styles.studentInfo}>
                    <Text style={styles.studentName}>{student.display_name}</Text>
                    <Text style={styles.mutedText}>{student.email ?? 'Chưa có email'}</Text>
                    <Text style={styles.mutedText}>{count} bài đã giao</Text>
                  </View>
                  <StatusBadge label={count > 0 ? `${count} bài` : 'Chưa giao'} tone={count > 0 ? 'primary' : 'idle'} />
                </Pressable>
              );
            })}
            {filteredAssignmentStudents.length === 0 ? (
              <EmptyPanel icon="account-search-outline" title="Không tìm thấy học sinh" message="Thử nhập tên hoặc email khác." />
            ) : null}
          </View>
        </AppCard>

        <AppCard style={[styles.panel, styles.assignmentDetailColumn]}>
          {!selectedStudent ? (
            <EmptyPanel icon="account-arrow-right-outline" title="Chọn một học sinh để xem bài đã giao." message="Danh sách bài đã giao sẽ hiển thị ở đây." />
          ) : (
            <>
              <View style={styles.panelHeader}>
                <View>
                  <Text style={styles.cardTitle}>{selectedStudent.display_name}</Text>
                  <Text style={styles.cardDescription}>{selectedStudent.email ?? 'Chưa có email'} — {visibleAssignments.length} bài đã giao</Text>
                </View>
                <StatusBadge label={selectedWeek.label} tone="primary" />
              </View>

              <View style={styles.assignmentDetailSection}>
                <Text style={styles.cardTitle}>Bài đã giao</Text>
                {assignmentsLoading ? (
                  <LoadingState title="" message="Đang tải bài đã giao..." />
                ) : visibleAssignments.length === 0 ? (
                  <EmptyPanel
                    icon="clipboard-text-outline"
                    title="Học sinh này chưa có bài được giao."
                    message="Nhấn 'Giao bài mới' để tạo bài tập đầu tiên."
                  />
                ) : (
                  <View style={styles.assignmentTopicList}>
                    {visibleAssignments.map((a) => (
                      <AssignmentRow key={a.id} assignment={a} onPress={() => void openGradebook(a)} />
                    ))}
                  </View>
                )}
              </View>
              {gradebookAssignment ? (
                <View style={styles.assignmentDetailSection}>
                  <View style={styles.panelHeader}>
                    <View><Text style={styles.cardTitle}>Bảng điểm: {gradebookAssignment.title}</Text><Text style={styles.cardDescription}>Điểm Assignment được tách khỏi điểm luyện tự do.</Text></View>
                    <Pressable onPress={() => setGradebookAssignment(null)} style={styles.smallButton}><Text style={styles.smallButtonText}>Đóng</Text></Pressable>
                  </View>
                  {gradebookLoading ? <LoadingState title="" message="Đang tải bảng điểm..." /> : null}
                  {gradebookError ? <ErrorState title="Chưa thể tải bảng điểm" message={gradebookError} /> : null}
                  {!gradebookLoading && !gradebookError ? <View style={styles.gradebookRows}>
                    {gradebook.map((row) => <Pressable key={row.student_id} onPress={() => { setGradebookStudent(row); setOverrideScore(row.final_score == null ? '' : String(row.final_score)); setOverrideReason(''); }} style={styles.gradebookRow}><View style={styles.gradebookName}><Text style={styles.studentName}>{row.student_name ?? 'Học sinh'}</Text><Text style={styles.mutedText}>{row.completed_items}/{row.total_items} từ · {row.submitted_at ? 'Đã nộp' : 'Chưa nộp'}</Text></View><StatusBadge label={formatAssignmentWorkStatus(row.work_status)} tone={row.is_locked ? 'idle' : row.work_status === 'completed' ? 'success' : 'primary'} /><StatusBadge label={formatAssignmentGrade(row.grading_status, row.final_score)} tone={row.grading_status === 'needs_review' ? 'warning' : row.grading_status === 'graded' ? 'success' : 'processing'} /></Pressable>)}
                    {gradebook.length === 0 ? <EmptyPanel icon="clipboard-text-outline" title="Chưa có học sinh nhận bài" message="Danh sách recipient sẽ hiển thị khi bài được giao." /> : null}
                  </View> : null}
                  {gradebookStudent ? <View style={styles.gradebookDetail}><Text style={styles.cardTitle}>Chi tiết: {gradebookStudent.student_name ?? 'Học sinh'}</Text><Text style={styles.mutedText}>AI: {gradebookStudent.auto_score == null ? 'Chưa có điểm tự động' : `${Math.round(gradebookStudent.auto_score)}/100`} · Điểm cuối: {gradebookStudent.final_score == null ? '—' : `${Math.round(gradebookStudent.final_score)}/100`}</Text><Text style={styles.mutedText}>Điểm từng từ: {Object.entries((gradebookStudent.details.best_scores as Record<string, number> | undefined) ?? {}).map(([itemId, score]) => `${itemId.slice(0, 8)}: ${Math.round(score)}`).join(' · ') || 'Chưa có kết quả AI'}</Text>{gradebookStudent.recordings.map((recording) => <Text key={recording.item_id} style={styles.mutedText}>Bản thu: {recording.word} · job {recording.practice_job_id?.slice(0, 8) ?? '—'}</Text>)}<TextInput value={overrideScore} onChangeText={setOverrideScore} keyboardType="numeric" placeholder="Điểm ghi đè 0–100" placeholderTextColor="#94A3B8" style={styles.createFormInput} /><TextInput value={overrideReason} onChangeText={setOverrideReason} placeholder="Lý do ghi đè (bắt buộc)" placeholderTextColor="#94A3B8" style={styles.createFormInput} /><Pressable onPress={() => void submitGradeOverride()} disabled={overrideLoading || !overrideReason.trim()} style={[styles.createSubmitBtn, (overrideLoading || !overrideReason.trim()) ? styles.disabledAction : null]}><Text style={styles.createSubmitBtnText}>{overrideLoading ? 'Đang lưu...' : 'Lưu điểm ghi đè'}</Text></Pressable></View> : null}
                </View>
              ) : null}
            </>
          )}
        </AppCard>
      </View>

      <Modal visible={showCreate} transparent animationType="fade" onRequestClose={() => setShowCreate(false)}>
        <KeyboardAvoidingView style={styles.modalKeyboardRoot} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <TouchableWithoutFeedback onPress={() => setShowCreate(false)}>
            <View style={styles.studentDetailBackdrop}>
              <TouchableWithoutFeedback onPress={() => undefined}>
                <View style={[styles.createAssignmentModal, {
                  width: Math.max(0, Math.min(520, windowWidth - 24)),
                  maxHeight: Math.max(0, windowHeight - 24),
                  padding: isCompact ? 16 : 24,
                }]}>
                <View style={styles.modalHeader}>
                  <Text style={styles.cardTitle}>Giao bài tập mới</Text>
                  <Pressable accessibilityRole="button" accessibilityLabel="Đóng tạo bài tập" onPress={() => setShowCreate(false)} style={styles.modalCloseBtn}>
                    <MaterialCommunityIcons name="close" size={20} color={colors.muted} />
                  </Pressable>
                </View>

                <ScrollView keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
                  <View style={styles.createFormGroup}>
                    <Text style={styles.createFormLabel}>Tiêu đề *</Text>
                    <TextInput
                      value={createTitle}
                      onChangeText={setCreateTitle}
                      placeholder="Nhập tiêu đề bài tập..."
                      placeholderTextColor="#94A3B8"
                      style={styles.createFormInput}
                    />
                  </View>

                  <View style={styles.createFormGroup}>
                    <Text style={styles.createFormLabel}>Bộ từ vựng *</Text>
                    {vocabSetsLoading ? (
                      <Text style={styles.mutedText}>Đang tải bộ từ...</Text>
                    ) : (
                      <View style={styles.setPickerList}>
                        {vocabSets.slice(0, 8).map((s) => (
                          <Pressable
                            key={s.id}
                            accessibilityRole="button"
                            onPress={() => setCreateSetId(s.id)}
                            style={[styles.setPickerItem, createSetId === s.id ? styles.setPickerItemActive : null]}
                          >
                            <Text style={[styles.setPickerText, createSetId === s.id ? styles.setPickerTextActive : null]}>
                              {s.title}
                            </Text>
                            {s.item_count != null ? (
                              <Text style={styles.mutedText}>{s.item_count} từ</Text>
                            ) : null}
                          </Pressable>
                        ))}
                        {vocabSets.length === 0 ? (
                          <Text style={styles.mutedText}>Không có bộ từ nào.</Text>
                        ) : null}
                      </View>
                    )}
                    {!vocabSetsLoading && vocabSets.length > 0 && !createSetId ? (
                      <Text style={styles.createHintText}>Vui lòng chọn bộ từ vựng</Text>
                    ) : null}
                  </View>

                  <View style={styles.createFormGroup}>
                    <Text style={styles.createFormLabel}>Giao cho</Text>
                  <View style={[styles.createTargetRow, isCompact ? styles.createTargetRowCompact : null]}>
                      <Pressable
                        accessibilityRole="button"
                        onPress={() => setCreateTarget('class')}
                        style={[styles.createTargetBtn, createTarget === 'class' ? styles.createTargetBtnActive : null]}
                      >
                        <Text style={[styles.createTargetBtnText, createTarget === 'class' ? styles.createTargetBtnTextActive : null]}>
                          Cả lớp
                        </Text>
                      </Pressable>
                      <Pressable
                        accessibilityRole="button"
                        onPress={() => setCreateTarget('student')}
                        disabled={!selectedAssignmentStudentId}
                        style={[
                          styles.createTargetBtn,
                          createTarget === 'student' ? styles.createTargetBtnActive : null,
                          !selectedAssignmentStudentId ? styles.disabledAction : null,
                        ]}
                      >
                        <Text style={[styles.createTargetBtnText, createTarget === 'student' ? styles.createTargetBtnTextActive : null]}>
                          {selectedStudent ? selectedStudent.display_name : 'Học sinh cụ thể'}
                        </Text>
                      </Pressable>
                    </View>
                  </View>

                  {!createIsAssessment ? (
                    <View style={styles.createFormGroup}>
                      <Text style={styles.createFormLabel}>Hạn nộp (DD/MM/YYYY, tùy chọn)</Text>
                      <TextInput
                        value={createDueDate}
                        onChangeText={setCreateDueDate}
                        placeholder="Chọn ngày"
                        placeholderTextColor="#94A3B8"
                        style={styles.createFormInput}
                      />
                    </View>
                  ) : null}

                  <View style={styles.createFormGroup}>
                    <Pressable
                      accessibilityRole="button"
                      onPress={() => setCreateIsAssessment(prev => !prev)}
                      style={[styles.createTargetBtn, createIsAssessment ? styles.createTargetBtnActive : null, { flex: 0, alignSelf: 'flex-start', paddingHorizontal: 16 }]}
                    >
                      <Text style={[styles.createTargetBtnText, createIsAssessment ? styles.createTargetBtnTextActive : null]}>
                        {createIsAssessment ? '✓ Chế độ kiểm tra' : 'Chế độ kiểm tra'}
                      </Text>
                    </Pressable>
                  </View>

                  {createIsAssessment ? (
                    <>
                      <View style={styles.createFormGroup}>
                        <Text style={styles.createFormLabel}>Ngày/Tháng/Năm (DD/MM/YYYY, bắt buộc)</Text>
                        <TextInput
                          value={createAssessmentDate}
                          onChangeText={setCreateAssessmentDate}
                          placeholder="Chọn ngày"
                          placeholderTextColor="#94A3B8"
                          style={styles.createFormInput}
                        />
                      </View>
                      <View style={styles.createFormGroup}>
                        <Text style={styles.createFormLabel}>Giờ : Phút (HH:MM, mặc định 23:59)</Text>
                        <TextInput
                          value={createAssessmentTime}
                          onChangeText={setCreateAssessmentTime}
                          placeholder="Chọn giờ"
                          placeholderTextColor="#94A3B8"
                          style={styles.createFormInput}
                        />
                      </View>
                      <View style={styles.createFormGroup}>
                        <Text style={styles.createFormLabel}>Thời gian mỗi từ (giây, mặc định 60)</Text>
                        <TextInput
                          value={createTimer}
                          onChangeText={setCreateTimer}
                          placeholder="60"
                          placeholderTextColor="#94A3B8"
                          keyboardType="numeric"
                          style={styles.createFormInput}
                        />
                      </View>
                    </>
                  ) : null}

                  {createError ? (
                    <Text style={styles.createErrorText}>{createError}</Text>
                  ) : null}

                  <View style={[styles.createModalActions, isCompact ? styles.createModalActionsCompact : null]}>
                    <Pressable accessibilityRole="button" onPress={() => setShowCreate(false)} style={styles.createCancelBtn}>
                      <Text style={styles.createCancelBtnText}>Hủy</Text>
                    </Pressable>
                    <Pressable
                      accessibilityRole="button"
                      onPress={submitCreate}
                      disabled={createLoading || !createTitle.trim() || !createSetId}
                      style={[
                        styles.createSubmitBtn,
                        (createLoading || !createTitle.trim() || !createSetId) ? styles.disabledAction : null,
                      ]}
                    >
                      <Text style={styles.createSubmitBtnText}>{createLoading ? 'Đang giao...' : 'Giao bài'}</Text>
                    </Pressable>
                  </View>
                </ScrollView>
                </View>
              </TouchableWithoutFeedback>
            </View>
          </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
      </Modal>
    </View>
  );
}

function AssignmentRow({ assignment, onPress }: { assignment: Assignment; onPress: () => void }) {
  const contentLabel = assignment.content_type === 'vocabulary_set' ? 'Từ vựng' : 'Câu';
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.assignmentRow}>
      <View style={styles.assignmentIcon}>
        <MaterialCommunityIcons name="book-open-page-variant-outline" size={22} color={colors.primary} />
      </View>
      <View style={styles.assignmentMain}>
        <View style={styles.assignmentTitleRow}>
          <StatusBadge label={contentLabel} tone="primary" />
          <Text style={styles.assignmentTitle}>{assignment.title}</Text>
        </View>
        <Text style={styles.mutedText}>
          {assignment.due_date
            ? `Hạn: ${assignment.due_date.slice(0, 10)}`
            : assignment.class_id
            ? 'Giao cho cả lớp'
            : 'Giao cho cá nhân'}
        </Text>
      </View>
    </Pressable>
  );
}

function formatAssignmentWorkStatus(status: AssignmentGradebookItem['work_status']): string {
  return ({ not_started: 'Chưa làm', in_progress: 'Đang làm', completed: 'Hoàn thành', submitted: 'Đã nộp', overdue: 'Quá hạn', locked: 'Đã khóa' } as const)[status];
}

function formatAssignmentGrade(status: AssignmentGradebookItem['grading_status'], score: number | null): string {
  if (score != null) return `${Math.round(score)}/100`;
  return ({ pending: 'Chờ chấm', provisional: 'Tạm tính', processing: 'Đang chấm', graded: 'Đã chấm', needs_review: 'Cần xem lại' } as const)[status];
}

function ReportsSection({
  analytics,
  classStats,
  reportPeriod,
  reportKind,
  selectedWeek,
  onChangePeriod,
  onChangeKind,
  onExport,
}: {
  analytics: Required<TeacherAnalyticsResponse>;
  classStats: ReturnType<typeof summarizeClasses>;
  reportPeriod: ReportPeriod;
  reportKind: ReportKind;
  selectedWeek: WeekInMonth;
  onChangePeriod: (period: ReportPeriod) => void;
  onChangeKind: (kind: ReportKind) => void;
  onExport: () => void;
}) {
  const rows = buildReportRows(reportKind, reportPeriod, analytics, classStats);

  return (
    <View style={styles.contentGrid}>
      <View style={styles.mainColumn}>
        <AppCard style={styles.panel}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.cardTitle}>Xuất báo cáo CSV</Text>
              <Text style={styles.cardDescription}>Kỳ đang chọn: {selectedWeek.label} - {formatDateRange(selectedWeek)}. CSV có BOM UTF-8; nếu chưa có dữ liệu thật, file vẫn ghi rõ trạng thái.</Text>
            </View>
            <ActionButton icon="download-outline" label="Xuất báo cáo CSV" primary onPress={onExport} />
          </View>
          <View style={styles.segmentRow}>
            <SegmentButton label="Theo tuần" active={reportPeriod === 'week'} onPress={() => onChangePeriod('week')} />
            <SegmentButton label="Theo tháng" active={reportPeriod === 'month'} onPress={() => onChangePeriod('month')} />
          </View>
          <View style={styles.reportKindGrid}>
            {reportKinds.map((kind) => (
              <SegmentButton key={kind.key} label={kind.label} active={reportKind === kind.key} onPress={() => onChangeKind(kind.key)} />
            ))}
          </View>
          <View style={styles.reportRows}>
            {rows.map((row) => (
              <ReportRow key={`${row[2]}-${row[3]}`} label={String(row[2])} value={String(row[3])} note={String(row[4])} />
            ))}
          </View>
        </AppCard>
      </View>
    </View>
  );
}

function UtilitySection({
  icon,
  title,
  message,
}: {
  icon: IconName;
  title: string;
  message: string;
}) {
  return (
    <AppCard style={styles.panel}>
      <EmptyPanel icon={icon} title={title} message={message} />
    </AppCard>
  );
}

function ClassPicker({
  classes,
  selectedClassId,
  pendingReviewCount = 0,
  onSelectClass,
}: {
  classes: ClassSummary[];
  selectedClassId: string | null;
  pendingReviewCount?: number;
  onSelectClass: (classId: string) => void;
}) {
  return (
    <View style={styles.classSelectorList}>
      {classes.map((classItem) => (
        <Pressable
          key={classItem.id}
          accessibilityRole="button"
          onPress={() => onSelectClass(classItem.id)}
          style={[styles.classChip, selectedClassId === classItem.id ? styles.classChipActive : null]}
        >
          <View style={styles.classChipText}>
            <Text style={styles.classCode}>{classItem.code ?? 'Chưa có mã'}</Text>
            <Text numberOfLines={1} style={styles.classChipName}>{classItem.name}</Text>
            <Text style={styles.mutedText}>{classItem.student_count} học sinh</Text>
          </View>
          {selectedClassId === classItem.id && pendingReviewCount > 0 ? (
            <StatusBadge label={`${pendingReviewCount} chờ`} tone="warning" />
          ) : null}
        </Pressable>
      ))}
      {classes.length === 0 ? <Text style={styles.mutedText}>Chưa có lớp đang phụ trách.</Text> : null}
    </View>
  );
}

function StudentRosterRow({
  student,
  scoreSummary,
  onViewScores,
}: {
  student: ClassStudent;
  scoreSummary?: TeacherClassScoreStudent;
  onViewScores?: () => void;
}) {
  return (
    <View style={styles.studentRow}>
      <View style={styles.avatar}>
        <Text style={styles.avatarText}>{student.display_name.slice(0, 1).toUpperCase()}</Text>
      </View>
      <View style={styles.studentInfo}>
        <Text style={styles.studentName}>{student.display_name}</Text>
        {!isInternalEmail(student.email) ? <Text style={styles.mutedText}>{student.email ?? 'Chưa có email'}</Text> : null}
        {scoreSummary ? (
          <Text style={styles.mutedText}>
            {scoreSummary.total_attempts} lượt luyện - {scoreSummary.average_score === null ? 'Chưa có điểm' : `${Math.round(scoreSummary.average_score)}/100`} - {formatDate(scoreSummary.last_practice_at)}
          </Text>
        ) : null}
      </View>
      {onViewScores ? (
        <Pressable accessibilityRole="button" onPress={onViewScores} style={styles.smallButton}>
          <Text style={styles.smallButtonText}>Xem điểm</Text>
        </Pressable>
      ) : null}
      {scoreSummary?.needs_review_count ? (
        <StatusBadge label="Cần xem lại" tone="warning" />
      ) : null}
    </View>
  );
}

function ProgressDistribution({ analytics }: { analytics: Required<TeacherAnalyticsResponse> }) {
  const rows = [
    { label: 'Cần hỗ trợ', value: analytics.progress_distribution.need_support, color: colors.error },
    { label: 'Đang cải thiện', value: analytics.progress_distribution.improving, color: colors.warning },
    { label: 'Tiến bộ tốt', value: analytics.progress_distribution.good_progress, color: colors.success },
  ];
  const maxValue = Math.max(...rows.map((row) => row.value), 1);

  return (
    <View style={styles.barChart}>
      {rows.map((row) => (
        <View key={row.label} style={styles.barColumn}>
          <Text style={styles.barValue}>{row.value}</Text>
          <View style={styles.barTrack}>
            <View style={[styles.barFill, { height: `${Math.max(6, (row.value / maxValue) * 100)}%`, backgroundColor: row.color }]} />
          </View>
          <Text style={styles.barLabel}>{row.label}</Text>
        </View>
      ))}
    </View>
  );
}

function CommonErrors({ errors }: { errors: TeacherCommonError[] }) {
  return (
    <View style={styles.errorRows}>
      {errors.map((error) => (
        <View key={error.label} style={styles.errorRow}>
          <View>
            <Text style={styles.errorLabel}>{formatMistakeLabel(error.label)}</Text>
            <Text style={styles.mutedText}>Mức độ: {error.count >= 10 ? 'Cao' : error.count >= 4 ? 'Trung bình' : 'Thấp'}</Text>
            <Text style={styles.mutedText}>Có thể giao bài luyện theo lỗi này.</Text>
          </View>
          <StatusBadge label={`${error.count} lần`} tone={error.count >= 10 ? 'error' : error.count >= 4 ? 'warning' : 'primary'} />
        </View>
      ))}
    </View>
  );
}

type StudentPronunciationIssue = {
  label: string;
  count: number;
  examples: string[];
};

function buildStudentPronunciationIssues(scores: TeacherStudentScoresResponse | null): StudentPronunciationIssue[] {
  if (!scores) return [];

  const issues = new Map<string, StudentPronunciationIssue>();
  const addIssue = (label: string | null | undefined, targetWord?: string | null, count = 1) => {
    const normalized = label?.trim();
    if (!normalized) return;
    if (targetWord && normalized.toLowerCase() === targetWord.trim().toLowerCase()) return;

    const key = normalized.toLowerCase();
    const existing = issues.get(key) ?? { label: normalized, count: 0, examples: [] };
    existing.count += count;
    if (targetWord && !existing.examples.includes(targetWord)) {
      existing.examples.push(targetWord);
    }
    issues.set(key, existing);
  };

  scores.summary.common_mistakes.forEach((error) => {
    addIssue(error.label, null, error.count);
  });

  scores.items.forEach((item) => {
    if (item.status !== 'completed') return;
    item.mistake_categories.forEach((category) => addIssue(category, item.target_word));
    extractFeedbackPhonemes(item.feedback).forEach((phoneme) => addIssue(phoneme, item.target_word));
  });

  return Array.from(issues.values()).sort((first, second) => second.count - first.count).slice(0, 6);
}

function extractFeedbackPhonemes(feedback: Record<string, unknown> | null): string[] {
  if (!feedback) return [];

  const values = [
    feedback.problem_phonemes,
    feedback.problemPhonemes,
    feedback.phonemes,
    feedback.errors,
  ];

  return values.flatMap((value) => extractPhonemeStrings(value));
}

function extractPhonemeStrings(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return value.flatMap((item) => {
    if (typeof item === 'string') {
      return isLikelyPhoneme(item) ? [item] : [];
    }
    if (!item || typeof item !== 'object') return [];

    const record = item as Record<string, unknown>;
    const candidates = [record.phoneme, record.symbol, record.label];
    return candidates.filter((candidate): candidate is string => typeof candidate === 'string' && isLikelyPhoneme(candidate));
  });
}

function formatMistakeLabel(label: string): string {
  const trimmed = label.trim();
  if (!trimmed) return 'Lỗi phát âm';
  const normalized = trimmed.toLowerCase().replace(/[_-]/g, ' ');
  const categoryLabels: Record<string, string> = {
    final: 'Âm cuối',
    'final sound': 'Âm cuối',
    vowel: 'Độ dài nguyên âm',
    'vowel length': 'Độ dài nguyên âm',
    stress: 'Trọng âm từ',
    'word stress': 'Trọng âm từ',
    cluster: 'Cụm phụ âm',
    'consonant cluster': 'Cụm phụ âm',
    linking: 'Nối âm',
  };
  if (categoryLabels[normalized]) return categoryLabels[normalized];
  if (isLikelyPhoneme(trimmed)) {
    return `Âm /${trimmed}/`;
  }
  return trimmed;
}

function isLikelyPhoneme(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed || trimmed.includes(' ')) return false;
  if (trimmed.length > 4) return false;
  return /^[a-zɑɔəɪʊʌæɛɜːʃʒθðŋɡ]+$/i.test(trimmed);
}

function mistakeSeverity(count: number): string {
  if (count >= 10) return 'Cao';
  if (count >= 4) return 'Trung bình';
  return 'Thấp';
}

function scoreComment(score: number | null): string {
  if (score === null) return 'Chưa có điểm';
  if (score >= 85) return 'Tốt';
  if (score >= 70) return 'Ổn định';
  if (score >= 50) return 'Cần luyện thêm';
  return 'Cần cải thiện';
}

function MetricCard({
  icon,
  label,
  value,
  tone,
}: {
  icon: IconName;
  label: string;
  value: string;
  tone: 'blue' | 'teal' | 'orange' | 'red';
}) {
  return (
    <AppCard style={[styles.metricCard, metricToneStyles[tone]]}>
      <MaterialCommunityIcons name={icon} size={24} color={colors.primary} />
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </AppCard>
  );
}

function ActionButton({
  icon,
  label,
  primary = false,
  disabled = false,
  onPress,
}: {
  icon: IconName;
  label: string;
  primary?: boolean;
  disabled?: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled}
      onPress={onPress}
      style={[styles.actionButton, primary ? styles.actionButtonPrimary : null, disabled ? styles.disabledAction : null]}
    >
      <MaterialCommunityIcons name={icon} size={17} color={primary ? '#FFFFFF' : colors.primary} />
      <Text style={[styles.actionButtonText, primary ? styles.actionButtonTextPrimary : null]}>{label}</Text>
    </Pressable>
  );
}

function SegmentButton({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={[styles.segmentButton, active ? styles.segmentButtonActive : null]}>
      <Text style={[styles.segmentButtonText, active ? styles.segmentButtonTextActive : null]}>{label}</Text>
    </Pressable>
  );
}

function ClassTabButton({
  icon,
  label,
  active,
  badgeCount = 0,
  onPress,
}: {
  icon: IconName;
  label: string;
  active: boolean;
  badgeCount?: number;
  onPress: () => void;
}) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={[styles.classTabButton, active ? styles.classTabButtonActive : null]}>
      <MaterialCommunityIcons name={icon} size={16} color={active ? colors.secondary : colors.muted} />
      <Text style={[styles.classTabText, active ? styles.classTabTextActive : null]}>{label}</Text>
      {badgeCount > 0 ? (
        <View style={styles.classTabBadge}>
          <Text style={styles.classTabBadgeText}>{badgeCount > 99 ? '99+' : String(badgeCount)}</Text>
        </View>
      ) : null}
    </Pressable>
  );
}

function ReportRow({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <View style={styles.reportRow}>
      <Text style={styles.reportLabel}>{label}</Text>
      <View style={styles.reportValueWrap}>
        <Text style={styles.reportValue}>{value}</Text>
        {note ? <Text style={styles.reportNote}>{note}</Text> : null}
      </View>
    </View>
  );
}

function EmptyPanel({ icon, title, message }: { icon: IconName; title: string; message: string }) {
  return (
    <View style={styles.emptyPanel}>
      <MaterialCommunityIcons name={icon} size={28} color={colors.primary} />
      <View style={styles.emptyCopy}>
        <Text style={styles.emptyTitle}>{title}</Text>
        <Text style={styles.emptyText}>{message}</Text>
      </View>
    </View>
  );
}

function Notice({ message }: { message: string }) {
  return (
    <View style={styles.notice}>
      <MaterialCommunityIcons name="information-outline" size={18} color={colors.primary} />
      <Text style={styles.noticeText}>{message}</Text>
    </View>
  );
}

function normalizeAnalytics(data: TeacherAnalyticsResponse | null): Required<TeacherAnalyticsResponse> {
  if (!data) return ZERO_ANALYTICS;
  const averageScore = data.average_score ?? data.avg_score ?? 0;
  const distribution = data.progress_distribution ?? ZERO_ANALYTICS.progress_distribution;

  return {
    total_students: data.total_students ?? 0,
    total_practice_sessions: data.total_practice_sessions ?? 0,
    average_score: averageScore,
    avg_score: data.avg_score ?? averageScore,
    active_jobs: data.active_jobs ?? 0,
    need_support_count: data.need_support_count ?? distribution.need_support ?? 0,
    good_progress_count: data.good_progress_count ?? distribution.good_progress ?? 0,
    improving_count: data.improving_count ?? distribution.improving ?? 0,
    common_errors: data.common_errors ?? [],
    progress_distribution: {
      need_support: distribution.need_support ?? 0,
      improving: distribution.improving ?? 0,
      good_progress: distribution.good_progress ?? 0,
    },
    students: data.students ?? [],
  };
}

function summarizeClasses(classes: ClassSummary[]) {
  return {
    totalClasses: classes.length,
    totalStudents: classes.reduce((sum, item) => sum + item.student_count, 0),
    totalTeachers: classes.reduce((sum, item) => sum + item.teacher_count, 0),
    activeClasses: classes.filter((item) => !item.status || item.status === 'active').length,
  };
}

function buildReportRows(
  kind: ReportKind,
  period: ReportPeriod,
  analytics: Required<TeacherAnalyticsResponse>,
  classStats: ReturnType<typeof summarizeClasses>,
): string[][] {
  const periodLabel = period === 'week' ? 'Theo tuần' : 'Theo tháng';
  const noDataNote = analytics.total_practice_sessions > 0 ? '' : 'Chưa có dữ liệu luyện tập thật';

  if (kind === 'errors') {
    const rows = analytics.common_errors.map((error) => ['Lỗi phát âm', periodLabel, error.label, String(error.count), '']);
    return rows.length > 0 ? rows : [['Lỗi phát âm', periodLabel, 'Trạng thái', 'Chưa có dữ liệu', noDataNote]];
  }

  if (kind === 'assignments') {
    return [['Bài luyện đã giao', periodLabel, 'Trạng thái', 'Chưa có dữ liệu', 'Chưa có assignment API']];
  }

  if (kind === 'ai_review') {
    return [['Kết quả AI cần kiểm tra', periodLabel, 'Trạng thái', 'Chưa có dữ liệu', 'Chưa có endpoint AI review']];
  }

  if (kind === 'progress') {
    return [
      ['Tiến độ luyện tập', periodLabel, 'Cần hỗ trợ', String(analytics.need_support_count), noDataNote],
      ['Tiến độ luyện tập', periodLabel, 'Đang cải thiện', String(analytics.improving_count), noDataNote],
      ['Tiến độ luyện tập', periodLabel, 'Tiến bộ tốt', String(analytics.good_progress_count), noDataNote],
    ];
  }

  return [
    ['Kết quả học tập', periodLabel, 'Lớp học', String(classStats.totalClasses), ''],
    ['Kết quả học tập', periodLabel, 'Học sinh', String(classStats.totalStudents), ''],
    ['Kết quả học tập', periodLabel, 'Lượt luyện tập', String(analytics.total_practice_sessions), noDataNote],
    ['Kết quả học tập', periodLabel, 'Điểm trung bình', analytics.total_practice_sessions > 0 ? `${Math.round(analytics.average_score)}%` : 'Chưa có dữ liệu', noDataNote],
  ];
}

function classStatusLabel(status: string | null | undefined) {
  if (!status || status === 'active') return 'Đang hoạt động';
  if (status === 'pending') return 'Đang chờ';
  if (status === 'inactive') return 'Không hoạt động';
  return status;
}

function statusToneForAssignment(status: string): 'idle' | 'processing' | 'success' | 'warning' | 'error' | 'primary' {
  if (status === 'Đang chẩn đoán') return 'processing';
  if (status === 'Đã nộp' || status === 'Đã có kết quả') return 'success';
  if (status === 'Cần giáo viên kiểm tra') return 'warning';
  if (status === 'Xử lý lỗi') return 'error';
  return 'idle';
}

function isTeacherSection(value: unknown): value is TeacherSection {
  return value === 'overview' || value === 'classes' || value === 'reports' || value === 'settings' || value === 'support';
}

function normalizeTeacherSection(value: unknown): TeacherSection | null {
  if (value === 'actions') return 'classes';
  return isTeacherSection(value) ? value : null;
}

function getPendingTeacherSection(): TeacherSection {
  if (typeof window === 'undefined') {
    return 'overview';
  }

  const pendingSection = (window as Window & { __phoenixTeacherSection?: string }).__phoenixTeacherSection;
  return normalizeTeacherSection(pendingSection) ?? 'overview';
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return 'Chưa có';
  try {
    const d = new Date(iso);
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    return `${day}/${month}/${d.getFullYear()}`;
  } catch {
    return iso;
  }
}

function parseDDMMYYYY(value: string): string | null {
  const match = value.trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!match) return null;
  const [, dd, mm, yyyy] = match;
  return `${yyyy}-${mm}-${dd}`;
}

function isInternalEmail(email: string | null | undefined): boolean {
  if (!email) return false;
  return email.includes('@phoenix-demo.local') || email.includes('@phoenix.edu.vn');
}

const WEEK_DAY_HEADERS = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];
const WEEK_DAY_MS = 86400000;

function WeekPickerButton({ selectedWeek, onChange }: { selectedWeek: WeekInMonth; onChange: (week: WeekInMonth) => void }) {
  const { width: windowWidth } = useWindowDimensions();
  const [isOpen, setIsOpen] = useState(false);
  const [viewMonth, setViewMonth] = useState(() => new Date(selectedWeek.start));
  const [btnRect, setBtnRect] = useState<{ top: number; right: number } | null>(null);
  const btnRef = useRef<View>(null);

  const weeks = useMemo(() => getWeeksInMonth(viewMonth), [viewMonth]);

  const calendarWeeks = useMemo(() => {
    const y = viewMonth.getFullYear();
    const m = viewMonth.getMonth();
    const firstMonday = startOfMondayWeek(new Date(y, m, 1));
    return weeks.map((week) => {
      const monday = new Date(firstMonday.getTime() + (week.index - 1) * 7 * WEEK_DAY_MS);
      const days = Array.from({ length: 7 }, (_, i) => {
        const d = new Date(monday.getTime() + i * WEEK_DAY_MS);
        return { n: d.getDate(), inMonth: d.getMonth() === m };
      });
      return { week, days };
    });
  }, [weeks, viewMonth]);

  const monthLabel = useMemo(() => {
    const m = viewMonth.getMonth() + 1;
    const y = viewMonth.getFullYear();
    return `Tháng ${m}/${y}`;
  }, [viewMonth]);

  const shiftMonth = (delta: number) => {
    setViewMonth((d) => new Date(d.getFullYear(), d.getMonth() + delta, 1));
  };

  const handleToggle = () => {
    if (!isOpen) {
      setViewMonth(new Date(selectedWeek.start));
      btnRef.current?.measure((_fx, _fy, w, h, px, py) => {
        const popupWidth = Math.min(320, Math.max(0, windowWidth - 24));
        const desiredRight = windowWidth - px - w;
        const right = Math.max(12, Math.min(windowWidth - popupWidth - 12, desiredRight));
        setBtnRect({ top: py + h + 4, right });
        setIsOpen(true);
      });
    } else {
      setIsOpen(false);
    }
  };

  const handleSelect = (week: WeekInMonth) => {
    onChange(week);
    setIsOpen(false);
  };

  const calendarContent = (
    <View style={[styles.weekPickerDropdown, { width: Math.min(320, Math.max(0, windowWidth - 24)) }, btnRect ? { top: btnRect.top, right: btnRect.right } : { top: 0, right: 0 }]}>
      <View style={styles.weekPickerHeader}>
        <Pressable accessibilityRole="button" accessibilityLabel="Tháng trước" onPress={() => shiftMonth(-1)} style={styles.weekPickerNav}>
          <MaterialCommunityIcons name="chevron-left" size={20} color={colors.primary} />
        </Pressable>
        <Text style={styles.weekPickerMonthLabel}>{monthLabel}</Text>
        <Pressable accessibilityRole="button" accessibilityLabel="Tháng sau" onPress={() => shiftMonth(1)} style={styles.weekPickerNav}>
          <MaterialCommunityIcons name="chevron-right" size={20} color={colors.primary} />
        </Pressable>
      </View>
      <View style={styles.weekPickerDayHeader}>
        <View style={styles.weekPickerWeekCell} />
        {WEEK_DAY_HEADERS.map((d) => (
          <Text key={d} style={styles.weekPickerDayLabel}>{d}</Text>
        ))}
      </View>
      {calendarWeeks.map(({ week, days }) => {
        const active = week.value === selectedWeek.value;
        return (
          <Pressable
            key={week.value}
            accessibilityRole="button"
            onPress={() => handleSelect(week)}
            style={[styles.weekPickerRow, active ? styles.weekPickerRowActive : null]}
          >
            <Text style={[styles.weekPickerWeekLabel, active ? styles.weekPickerWeekLabelActive : null]}>
              {week.label}
            </Text>
            {days.map(({ n, inMonth }, i) => (
              <Text
                key={i}
                style={[
                  styles.weekPickerDayNum,
                  !inMonth ? styles.weekPickerDayNumOut : null,
                  active ? styles.weekPickerDayNumActive : null,
                ]}
              >
                {n}
              </Text>
            ))}
          </Pressable>
        );
      })}
    </View>
  );

  return (
    <View ref={btnRef} style={styles.weekPickerWrap}>
      <Pressable accessibilityRole="button" onPress={handleToggle} style={[styles.actionButton, isOpen ? styles.actionButtonPrimary : null]}>
        <MaterialCommunityIcons name="calendar-week" size={17} color={isOpen ? '#FFFFFF' : colors.primary} />
        <Text style={[styles.actionButtonText, isOpen ? styles.actionButtonTextPrimary : null]}>
          {selectedWeek.label} ({formatDateRange(selectedWeek)})
        </Text>
        <MaterialCommunityIcons name={isOpen ? 'chevron-up' : 'chevron-down'} size={15} color={isOpen ? '#FFFFFF' : colors.muted} />
      </Pressable>
      <Modal visible={isOpen} transparent animationType="none" onRequestClose={() => setIsOpen(false)}>
        <TouchableWithoutFeedback onPress={() => setIsOpen(false)}>
          <View style={styles.weekPickerBackdrop}>
            <TouchableWithoutFeedback>
              {calendarContent}
            </TouchableWithoutFeedback>
          </View>
        </TouchableWithoutFeedback>
      </Modal>
    </View>
  );
}

const metricToneStyles = StyleSheet.create({
  blue: {
    backgroundColor: colors.softBlue,
    borderColor: '#BFDBFE',
  },
  teal: {
    backgroundColor: colors.softTeal,
    borderColor: '#99F6E4',
  },
  orange: {
    backgroundColor: colors.softOrange,
    borderColor: '#FED7AA',
  },
  red: {
    backgroundColor: colors.softRed,
    borderColor: '#FECACA',
  },
});

const styles = StyleSheet.create({
  headerBar: {
    minHeight: 64,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: -20,
    marginTop: -24,
    paddingHorizontal: 20,
    paddingVertical: 14,
  },
  appTitle: {
    color: colors.primary,
    fontSize: 20,
    fontWeight: '900',
  },
  appSubtitle: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  dashboardHeader: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 16,
  },
  dashboardCopy: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 320,
    gap: 4,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '900',
    lineHeight: 40,
  },
  sectionSubtitle: {
    color: colors.muted,
    fontSize: 16,
    lineHeight: 24,
  },
  headerActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    position: 'relative',
    zIndex: 9000,
    elevation: 20,
  },
  filterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  searchInput: {
    minHeight: 44,
    flexGrow: 1,
    flexBasis: 260,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 14,
    paddingHorizontal: 12,
  },
  inputGroup: {
    gap: 6,
  },
  inputLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  actionButton: {
    minHeight: 42,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    paddingHorizontal: 12,
  },
  actionButtonPrimary: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  actionButtonText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  actionButtonTextPrimary: {
    color: '#FFFFFF',
  },
  sectionStack: {
    gap: 16,
  },
  classWorkspace: {
    gap: 18,
  },
  classWorkspaceTopbar: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 12,
  },
  classBreadcrumb: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  breadcrumbText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  breadcrumbSeparator: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '900',
  },
  breadcrumbTextActive: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  classWorkspaceHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 12,
  },
  workspaceTitle: {
    color: colors.text,
    fontSize: 28,
    lineHeight: 34,
    fontWeight: '900',
  },
  workspaceSubtitle: {
    color: colors.muted,
    fontSize: 14,
    fontWeight: '800',
    marginTop: 3,
  },
  classTabBar: {
    minHeight: 48,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    flexDirection: 'row',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
    gap: 18,
  },
  classTabButton: {
    minHeight: 46,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingHorizontal: 2,
  },
  classTabButtonActive: {
    borderBottomColor: colors.secondary,
  },
  classTabText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
  classTabTextActive: {
    color: colors.text,
    fontWeight: '900',
  },
  classTabBadge: {
    minWidth: 20,
    height: 20,
    borderRadius: 999,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 6,
  },
  classTabBadgeText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '900',
  },
  studentTabStack: {
    gap: 12,
  },
  studentToolbar: {
    minHeight: 48,
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 10,
  },
  studentSearchInput: {
    height: 46,
    flexGrow: 1,
    flexBasis: 320,
    maxWidth: 520,
    minWidth: 0,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 14,
    paddingHorizontal: 14,
  },
  studentListPanel: {
    width: '100%',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flexGrow: 1,
    flexBasis: 180,
    flexShrink: 1,
    minWidth: 0,
    minHeight: 132,
    borderRadius: 12,
    gap: 8,
  },
  metricValue: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '900',
  },
  metricLabel: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
  contentGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  mainColumn: {
    flexGrow: 2,
    flexBasis: 620,
    flexShrink: 1,
    minWidth: 0,
    gap: 16,
  },
  sideColumn: {
    flexGrow: 1,
    flexBasis: 320,
    flexShrink: 1,
    minWidth: 0,
    gap: 16,
  },
  panel: {
    borderRadius: 12,
    gap: 14,
  },
  classSelectorPanel: {
    borderRadius: 12,
    gap: 12,
    paddingVertical: 16,
  },
  classHeaderCard: {
    borderRadius: 12,
    paddingVertical: 14,
  },
  classHeaderMain: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  classHeaderStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 8,
  },
  classTabsCard: {
    borderRadius: 12,
    paddingVertical: 12,
  },
  panelHeader: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 12,
  },
  cardTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  cardDescription: {
    color: colors.muted,
    fontSize: 14,
    lineHeight: 21,
  },
  mutedText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  classRosterGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
  },
  classRows: {
    flexGrow: 1,
    flexBasis: 260,
    gap: 10,
  },
  classSelectorList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  classChip: {
    minHeight: 72,
    flexGrow: 1,
    flexBasis: 180,
    maxWidth: 260,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  classChipActive: {
    borderColor: colors.secondary,
    backgroundColor: colors.softTeal,
  },
  classChipText: {
    flex: 1,
    minWidth: 0,
    gap: 2,
  },
  classChipName: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  classCard: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 14,
    gap: 5,
  },
  classCardActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  classSummaryRow: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 12,
    gap: 4,
  },
  classCode: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  className: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  rosterPanel: {
    flexGrow: 2,
    flexBasis: 480,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 14,
    gap: 12,
  },
  rosterTitle: {
    color: colors.text,
    fontSize: 19,
    fontWeight: '900',
  },
  studentRows: {
    gap: 10,
  },
  classScoreSummary: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  scoreSummaryItem: {
    flexGrow: 1,
    flexBasis: 150,
    minHeight: 92,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 12,
    justifyContent: 'center',
    gap: 4,
  },
  scoreTableHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    gap: 12,
  },
  scoreTableScrollContent: {
    flexGrow: 1,
  },
  scoreTableScrollable: {
    minWidth: 720,
  },
  scoreTableHeaderCell: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.muted,
  },
  scoreColName: {
    flex: 3,
    borderRightWidth: 1,
    borderRightColor: '#f1f5f9',
    paddingRight: 8,
  },
  scoreColScore: {
    flex: 1,
    alignItems: 'center',
    borderRightWidth: 1,
    borderRightColor: '#f1f5f9',
  },
  scoreColComment: {
    flex: 1,
    alignItems: 'center',
    borderRightWidth: 1,
    borderRightColor: '#f1f5f9',
  },
  scoreColBtn: {
    flex: 1,
    alignItems: 'flex-end',
  },
  studentDetailBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 12,
  },
  modalKeyboardRoot: {
    flex: 1,
  },
  studentDetailModal: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 600,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  modalCloseBtn: {
    width: 44,
    height: 44,
    borderRadius: 8,
    backgroundColor: '#F1F5F9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  scoreTableRow: {
    minHeight: 76,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 12,
  },
  scoreStudentIdentity: {
    flexGrow: 1,
    flexBasis: 180,
    minWidth: 160,
    gap: 2,
  },
  scoreCell: {
    flexBasis: 120,
    gap: 2,
  },
  scorePill: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scorePillNeutral: { backgroundColor: '#F1F5F9' },
  scorePillGreen: { backgroundColor: '#D1FAE5' },
  scorePillYellow: { backgroundColor: '#FEF3C7' },
  scorePillRed: { backgroundColor: '#FEE2E2' },
  scorePillText: {
    fontSize: 13,
    fontWeight: '800',
    color: '#065F46',
  },
  scorePillTextRed: { color: '#991B1B' },
  classHeaderStatLine: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '500',
  },
  scoreSummaryItemLarge: {
    flexBasis: 200,
    minHeight: 104,
  },
  scoreSummaryItemSmall: {
    flexBasis: 120,
    minHeight: 80,
  },
  scoreSummaryItemRed: {
    backgroundColor: '#FEE2E2',
    borderColor: '#FECACA',
  },
  scoreSummaryItemOrange: {
    backgroundColor: '#FEF3C7',
    borderColor: '#FDE68A',
  },
  metricValueRed: { color: '#DC2626' },
  metricLabelRed: { color: '#DC2626' },
  metricValueOrange: { color: '#D97706' },
  metricLabelOrange: { color: '#D97706' },
  studentRow: {
    minHeight: 68,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 12,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: colors.primary,
    fontSize: 16,
    fontWeight: '900',
  },
  studentInfo: {
    flex: 1,
    gap: 2,
  },
  studentName: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  segmentRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  segmentButton: {
    minHeight: 36,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    paddingHorizontal: 11,
  },
  segmentButtonActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  segmentButtonText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
  },
  segmentButtonTextActive: {
    color: '#FFFFFF',
  },
  bigNumber: {
    color: colors.text,
    fontSize: 42,
    fontWeight: '900',
  },
  barChart: {
    height: 230,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    gap: 12,
  },
  barColumn: {
    flex: 1,
    height: '100%',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 8,
  },
  barValue: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  barTrack: {
    width: '100%',
    maxWidth: 72,
    height: 150,
    borderRadius: 8,
    backgroundColor: '#F1F5F9',
    justifyContent: 'flex-end',
    overflow: 'hidden',
  },
  barFill: {
    width: '100%',
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
  },
  barLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
    textAlign: 'center',
  },
  aiReviewRows: {
    gap: 10,
  },
  aiReviewRow: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 12,
  },
  aiReviewMain: {
    flex: 1,
    gap: 4,
  },
  reviewReason: {
    color: colors.error,
    fontSize: 13,
    fontWeight: '900',
  },
  smallButton: {
    minHeight: 44,
    borderRadius: 8,
    backgroundColor: colors.softBlue,
    justifyContent: 'center',
    paddingHorizontal: 12,
  },
  smallButtonText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  disabledActionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  disabledAction: {
    opacity: 0.55,
  },
  pillList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  infoPill: {
    borderRadius: 999,
    backgroundColor: colors.softBlue,
    paddingHorizontal: 11,
    paddingVertical: 8,
  },
  infoPillText: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  errorRows: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  errorRow: {
    minHeight: 56,
    flexGrow: 1,
    flexBasis: 220,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  errorLabel: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  pronunciationIssueCopy: {
    flex: 1,
    minWidth: 0,
    gap: 3,
  },
  mistakeGroupGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  mistakeGroupCard: {
    flexGrow: 1,
    flexBasis: 240,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 12,
    gap: 8,
  },
  mistakeGroupCopy: {
    flex: 1,
    minWidth: 0,
    gap: 4,
  },
  statusRows: {
    gap: 8,
  },
  statusRow: {
    alignSelf: 'flex-start',
  },
  assignmentPanel: {
    gap: 16,
  },
  assignmentToolbar: {
    minHeight: 60,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 12,
  },
  assignmentToolbarActions: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
  },
  successBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#F0FDF4',
    borderRadius: 8,
    padding: 12,
    marginTop: 12,
    borderLeftWidth: 4,
    borderLeftColor: colors.success,
  },
  successBannerText: {
    color: '#15803D',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  assignmentList: {
    gap: 12,
  },
  assignmentWorkspaceGrid: {
    width: '100%',
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    gap: 18,
  },
  assignmentWorkspaceGridStacked: {
    flexDirection: 'column',
  },
  assignmentStudentColumn: {
    flexGrow: 0,
    flexShrink: 1,
    flexBasis: '36%',
    minWidth: 0,
  },
  assignmentDetailColumn: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: '62%',
    minWidth: 0,
  },
  assignmentSearchInput: {
    height: 42,
    width: '100%',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 14,
    paddingHorizontal: 12,
  },
  assignmentStudentList: {
    gap: 10,
  },
  assignmentStudentRow: {
    minHeight: 70,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 10,
  },
  assignmentStudentRowActive: {
    borderColor: colors.secondary,
    backgroundColor: colors.softTeal,
  },
  assignmentDetailSection: {
    gap: 10,
  },
  assignmentTopicList: {
    gap: 10,
  },
  gradebookRows: {
    gap: 8,
  },
  gradebookRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    padding: 10,
  },
  gradebookName: {
    flexGrow: 1,
    minWidth: 160,
    gap: 3,
  },
  gradebookDetail: {
    gap: 8,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  assignmentRow: {
    minHeight: 90,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  assignmentIcon: {
    width: 42,
    height: 42,
    borderRadius: 8,
    backgroundColor: '#F1F5F9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  assignmentMain: {
    flex: 1,
    minWidth: 0,
    gap: 6,
  },
  assignmentTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
  },
  assignmentTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '900',
  },
  assignmentProgress: {
    width: 190,
    gap: 8,
  },
  assignmentProgressHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  assignmentCount: {
    color: colors.secondary,
    fontSize: 13,
    fontWeight: '900',
  },
  assignmentProgressTrack: {
    height: 6,
    borderRadius: 999,
    backgroundColor: '#E2E8F0',
    overflow: 'hidden',
  },
  assignmentProgressFill: {
    height: '100%',
    borderRadius: 999,
    backgroundColor: colors.secondary,
  },
  reportKindGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  reportRows: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  reportRow: {
    minHeight: 58,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 14,
    gap: 12,
  },
  reportLabel: {
    flex: 1,
    color: colors.muted,
    fontSize: 13,
    fontWeight: '800',
  },
  reportValueWrap: {
    flex: 1,
    alignItems: 'flex-end',
    gap: 2,
  },
  reportValue: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
    textAlign: 'right',
  },
  reportNote: {
    color: colors.muted,
    fontSize: 11,
    textAlign: 'right',
  },
  emptyPanel: {
    minHeight: 112,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    padding: 16,
  },
  emptyCopy: {
    flex: 1,
    gap: 4,
  },
  emptyTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  emptyText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
  notice: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: colors.softBlue,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
  },
  noticeText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '800',
  },
  weekPickerWrap: {},
  weekPickerBackdrop: {
    flex: 1,
  },
  weekPickerDropdown: {
    position: 'absolute',
    width: 320,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    backgroundColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.12,
    shadowRadius: 12,
    elevation: 16,
    overflow: 'hidden',
  },
  weekPickerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: '#F8FAFC',
  },
  weekPickerNav: {
    width: 44,
    height: 44,
    borderRadius: 8,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  weekPickerMonthLabel: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  weekPickerDayHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
    backgroundColor: '#F8FAFC',
  },
  weekPickerWeekCell: {
    width: 52,
  },
  weekPickerDayLabel: {
    flex: 1,
    textAlign: 'center',
    fontSize: 11,
    fontWeight: '700',
    color: colors.muted,
  },
  weekPickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },
  weekPickerRowActive: {
    backgroundColor: '#EEF2FF',
  },
  weekPickerWeekLabel: {
    width: 52,
    fontSize: 12,
    fontWeight: '700',
    color: colors.muted,
  },
  weekPickerWeekLabelActive: {
    color: colors.primary,
  },
  weekPickerDayNum: {
    flex: 1,
    textAlign: 'center',
    fontSize: 13,
    fontWeight: '600',
    color: colors.text,
  },
  weekPickerDayNumOut: {
    color: '#cbd5e1',
  },
  weekPickerDayNumActive: {
    color: colors.primary,
  },
  createAssignmentModal: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 520,
    gap: 4,
    overflow: 'hidden',
  },
  createFormGroup: {
    gap: 8,
    marginTop: 16,
  },
  createFormLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '700',
  },
  createFormInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    fontSize: 14,
    color: colors.text,
    backgroundColor: '#FAFAFA',
  },
  setPickerList: {
    gap: 6,
  },
  setPickerItem: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#FAFAFA',
  },
  setPickerItemActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  setPickerText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '700',
  },
  setPickerTextActive: {
    color: colors.primary,
  },
  createTargetRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  createTargetRowCompact: {
    flexDirection: 'column',
  },
  createTargetBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    alignItems: 'center',
    backgroundColor: '#FAFAFA',
  },
  createTargetBtnActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  createTargetBtnText: {
    color: colors.muted,
    fontSize: 13,
    fontWeight: '700',
  },
  createTargetBtnTextActive: {
    color: colors.primary,
  },
  createErrorText: {
    color: '#DC2626',
    fontSize: 13,
    marginTop: 8,
  },
  createHintText: {
    color: '#F59E0B',
    fontSize: 12,
    marginTop: 6,
  },
  createModalActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 20,
  },
  createModalActionsCompact: {
    flexDirection: 'column-reverse',
  },
  createCancelBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  createCancelBtnText: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '700',
  },
  createSubmitBtn: {
    flex: 1,
    borderRadius: 10,
    backgroundColor: colors.primary,
    paddingVertical: 12,
    alignItems: 'center',
  },
  createSubmitBtnText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
  },
});
