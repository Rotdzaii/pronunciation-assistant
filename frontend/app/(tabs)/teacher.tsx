import { useEffect, useMemo, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import {
  AppCard,
  AppScreen,
  ErrorState,
  LoadingState,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import {
  fetchTeacherAnalytics,
  fetchTeacherClassDetail,
  fetchTeacherClassScores,
  fetchTeacherClassStudents,
  fetchTeacherClasses,
  fetchTeacherReviewRequestDetail,
  fetchTeacherReviewRequests,
  fetchTeacherStudentScores,
  addTeacherReviewNote,
  requestTeacherReviewReanalysis,
  resolveTeacherReviewRequest,
} from '../../lib/api';
import {
  WeekInMonth,
  formatDateRange,
  getCurrentWeekOfMonth,
} from '../../lib/dateRange';
import { exportCsv } from '../../lib/exportCsv';
import { useAuth } from '../../lib/auth';
import { WeekSelector } from '../../components/WeekSelector';
import {
  formatPracticeStatus,
  formatReviewReason,
  formatReviewSeverity,
  formatReviewSource,
  formatReviewStatus,
} from '../../lib/format';
import type {
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
  { key: 'learning', label: 'Kết quả học tập' },
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
        <WeekSelector selectedWeek={selectedWeek} onChange={onWeekChange} />
        <ActionButton icon="file-eye-outline" label="Xem báo cáo" primary onPress={onReports} />
      </View>
    );
  }

  if (section === 'reports') {
    return (
      <View style={styles.headerActions}>
        <WeekSelector selectedWeek={selectedWeek} onChange={onWeekChange} />
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
                <Text style={styles.mutedText}>Điểm TB: {averageScore === null ? 'Chưa có điểm' : `${Math.round(averageScore)}%`}</Text>
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
          <Text style={styles.mutedText}>{log.created_at ?? ''}</Text>
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
  onOpenReviewDetail: (requestId: string) => void;
  onCloseReviewDetail: () => void;
  onSaveReviewNote: (requestId: string, teacherNote: string) => void;
  onResolveReview: (requestId: string, status: 'resolved' | 'rejected', resolution: string, teacherNote?: string | null) => void;
  onRequestReviewReanalysis: (requestId: string) => void;
}) {
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
            <StatusBadge label={`${students.length} học sinh`} tone="primary" />
            <StatusBadge label={`Điểm TB: ${averageScore === null ? 'Chưa có điểm' : `${averageScore}%`}`} tone="processing" />
            <StatusBadge label={`${reviewData?.pending_count ?? 0} yêu cầu chờ`} tone={(reviewData?.pending_count ?? 0) > 0 ? 'warning' : 'success'} />
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
          {selectedScoreStudent ? (
            <StudentScoresPanel student={selectedScoreStudent} scores={studentScores} loading={loadingScores} error={scoreError} />
          ) : null}
        </View>
      ) : null}

      {selectedClassId && workspaceTab === 'scores' ? (
        <View style={styles.sectionStack}>
          <ClassWorkspaceOverview
            students={students}
            classScores={classScores}
            onSelectStudent={onSelectStudent}
          />
          {selectedScoreStudent ? (
            <StudentScoresPanel student={selectedScoreStudent} scores={studentScores} loading={loadingScores} error={scoreError} />
          ) : null}
        </View>
      ) : null}

      {selectedClassId && workspaceTab === 'assignments' ? (
        <AssignmentSection
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
    </View>
  );
}

function ClassWorkspaceOverview({
  students,
  classScores,
  onSelectStudent,
}: {
  students: ClassStudent[];
  classScores: TeacherClassScoresResponse | null;
  onSelectStudent: (student: ClassStudent) => void;
}) {
  const scoredStudents = classScores?.students.filter((student) => student.average_score !== null) ?? [];
  const averageScore = scoredStudents.length
    ? Math.round(scoredStudents.reduce((sum, student) => sum + (student.average_score ?? 0), 0) / scoredStudents.length)
    : null;
  const highestScore = scoredStudents.length
    ? Math.max(...scoredStudents.map((student) => Math.round(student.average_score ?? 0)))
    : null;
  const needsImprovementCount = scoredStudents.filter((student) => (student.average_score ?? 0) < 70).length;
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
        <View style={styles.scoreSummaryItem}>
          <Text style={styles.metricValue}>{averageScore === null ? '--' : `${averageScore}%`}</Text>
          <Text style={styles.metricLabel}>Điểm trung bình lớp</Text>
        </View>
        <View style={styles.scoreSummaryItem}>
          <Text style={styles.metricValue}>{scoredStudents.length}</Text>
          <Text style={styles.metricLabel}>Học sinh đã có điểm</Text>
        </View>
        <View style={styles.scoreSummaryItem}>
          <Text style={styles.metricValue}>{highestScore === null ? '--' : `${highestScore}%`}</Text>
          <Text style={styles.metricLabel}>Điểm cao nhất</Text>
        </View>
        <View style={styles.scoreSummaryItem}>
          <Text style={styles.metricValue}>{needsImprovementCount}</Text>
          <Text style={styles.metricLabel}>Cần cải thiện</Text>
        </View>
      </View>
      <Text style={styles.cardTitle}>Bảng điểm theo học sinh</Text>
      <View style={styles.studentRows}>
        {(classScores?.students ?? []).map((student) => (
          <View key={student.student_id} style={styles.scoreTableRow}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>{student.student_name.slice(0, 1).toUpperCase()}</Text>
            </View>
            <View style={styles.scoreStudentIdentity}>
              <Text style={styles.studentName}>{student.student_name}</Text>
              <Text style={styles.mutedText}>{student.student_email ?? 'Chưa có email'}</Text>
            </View>
            <View style={styles.scoreCell}>
              <Text style={styles.reportLabel}>Điểm trung bình</Text>
              <Text style={styles.reportValue}>{student.average_score === null ? 'Chưa có điểm' : `${Math.round(student.average_score)}%`}</Text>
            </View>
            <View style={styles.scoreCell}>
              <Text style={styles.reportLabel}>Điểm lần gần nhất</Text>
              <Text style={styles.reportValue}>Chưa có dữ liệu</Text>
            </View>
            <View style={styles.scoreCell}>
              <Text style={styles.reportLabel}>Bài đã có điểm</Text>
              <Text style={styles.reportValue}>{student.completed_attempts}</Text>
            </View>
            <View style={styles.scoreCell}>
              <Text style={styles.reportLabel}>Nhận xét</Text>
              <Text style={styles.reportValue}>{scoreComment(student.average_score)}</Text>
            </View>
            <Pressable
              accessibilityRole="button"
              onPress={() => {
                const matchedStudent = studentsById.get(student.student_id);
                if (matchedStudent) onSelectStudent(matchedStudent);
              }}
              disabled={!studentsById.has(student.student_id)}
              style={[styles.smallButton, !studentsById.has(student.student_id) ? styles.disabledAction : null]}
            >
              <Text style={styles.smallButtonText}>Xem chi tiết</Text>
            </Pressable>
          </View>
        ))}
        {!classScores || classScores.students.length === 0 ? (
          <EmptyPanel icon="chart-box-outline" title="Chưa có dữ liệu điểm của lớp" message="Khi học sinh hoàn thành bài luyện và Phoenix trả score, kết quả sẽ hiển thị tại đây." />
        ) : null}
      </View>
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
                <Text style={styles.mutedText}>{item.created_at ?? 'Chưa rõ thời gian'} - {formatReviewStatus(item.status)}</Text>
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
  students,
  selectedWeek,
  onWeekChange,
  onExport,
}: {
  students: ClassStudent[];
  selectedWeek: WeekInMonth;
  onWeekChange: (week: WeekInMonth) => void;
  onExport: () => void;
}) {
  const [assignmentStudentSearch, setAssignmentStudentSearch] = useState('');
  const [selectedAssignmentStudentId, setSelectedAssignmentStudentId] = useState<string | null>(null);
  const filteredAssignmentStudents = students.filter((student) => {
    const needle = assignmentStudentSearch.trim().toLowerCase();
    if (!needle) return true;
    return student.display_name.toLowerCase().includes(needle) || (student.email ?? '').toLowerCase().includes(needle);
  });
  const selectedStudent = students.find((student) => student.id === selectedAssignmentStudentId) ?? null;
  const assignableTopics = [
    {
      type: 'Phát âm',
      title: 'Luyện âm cuối /t/ và /d/',
      description: 'Tập trung phát âm rõ phụ âm cuối trong từ ngắn.',
      icon: 'book-open-page-variant-outline' as IconName,
    },
    {
      type: 'Câu',
      title: 'Luyện trọng âm từ trong câu ngắn',
      description: 'Luyện nhấn trọng âm khi đọc câu ngắn.',
      icon: 'comment-text-outline' as IconName,
    },
    {
      type: 'Lỗi phát âm',
      title: 'Cụm phụ âm /str/ và /spl/',
      description: 'Luyện đọc liền cụm phụ âm khó ở tốc độ chậm.',
      icon: 'microphone-off' as IconName,
    },
    {
      type: 'Quiz',
      title: 'Ôn tập lỗi phát âm tuần này',
      description: 'Kiểm tra nhanh các lỗi học sinh thường mắc trong tuần.',
      icon: 'clipboard-check-outline' as IconName,
    },
  ];

  return (
    <View style={styles.assignmentPanel}>
      <View style={styles.assignmentToolbar}>
        <View>
          <Text style={styles.cardTitle}>Bài giao</Text>
          <Text style={styles.cardDescription}>Chọn học sinh để xem bài đã giao và chuẩn bị bài luyện phù hợp.</Text>
        </View>
        <WeekSelector selectedWeek={selectedWeek} onChange={onWeekChange} />
        <View style={styles.assignmentToolbarActions}>
          <ActionButton icon="download-outline" label="Xuất danh sách" onPress={onExport} />
          <ActionButton icon="plus" label="Sắp có" primary disabled onPress={() => undefined} />
        </View>
      </View>

      <View style={styles.assignmentWorkspaceGrid}>
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
            {filteredAssignmentStudents.map((student) => (
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
                  <Text style={styles.mutedText}>0 bài đã giao</Text>
                </View>
                <StatusBadge label="Chưa giao" tone="idle" />
              </Pressable>
            ))}
            {filteredAssignmentStudents.length === 0 ? (
              <EmptyPanel icon="account-search-outline" title="Không tìm thấy học sinh" message="Thử nhập tên hoặc email khác." />
            ) : null}
          </View>
        </AppCard>

        <AppCard style={[styles.panel, styles.assignmentDetailColumn]}>
          {!selectedStudent ? (
            <EmptyPanel icon="account-arrow-right-outline" title="Chọn một học sinh để xem bài đã giao." message="Danh sách bài và chủ đề có thể giao sẽ hiển thị ở đây." />
          ) : (
            <>
              <View style={styles.panelHeader}>
                <View>
                  <Text style={styles.cardTitle}>{selectedStudent.display_name}</Text>
                  <Text style={styles.cardDescription}>{selectedStudent.email ?? 'Chưa có email'} - 0 bài đã giao trong tuần này</Text>
                </View>
                <StatusBadge label={selectedWeek.label} tone="primary" />
              </View>

              <View style={styles.assignmentDetailSection}>
                <Text style={styles.cardTitle}>Bài đã giao</Text>
                <EmptyPanel
                  icon="clipboard-text-outline"
                  title="Học sinh này chưa có bài được giao trong tuần này."
                  message="Backend assignment creation chưa hoàn thiện, nên không ghi dữ liệu giao bài giả."
                />
              </View>

              <View style={styles.assignmentDetailSection}>
                <Text style={styles.cardTitle}>Chủ đề có thể giao</Text>
                <View style={styles.assignmentTopicList}>
                  {assignableTopics.map((topic) => (
                    <AssignmentTopicRow key={topic.title} topic={topic} />
                  ))}
                </View>
              </View>
            </>
          )}
        </AppCard>
      </View>
    </View>
  );
}

function AssignmentTopicRow({
  topic,
}: {
  topic: {
    type: string;
    title: string;
    description: string;
    icon: IconName;
  };
}) {
  return (
    <View style={styles.assignmentRow}>
      <View style={styles.assignmentIcon}>
        <MaterialCommunityIcons name={topic.icon} size={22} color={colors.primary} />
      </View>
      <View style={styles.assignmentMain}>
        <View style={styles.assignmentTitleRow}>
          <StatusBadge label={topic.type} tone="primary" />
          <Text style={styles.assignmentTitle}>{topic.title}</Text>
        </View>
        <Text style={styles.mutedText}>{topic.description}</Text>
      </View>
      <Pressable accessibilityRole="button" disabled style={[styles.smallButton, styles.disabledAction]}>
        <Text style={styles.smallButtonText}>Sắp có</Text>
      </Pressable>
    </View>
  );
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
        <Text style={styles.mutedText}>{student.email ?? 'Chưa có email'}</Text>
        {scoreSummary ? (
          <Text style={styles.mutedText}>
            {scoreSummary.total_attempts} lượt luyện - {scoreSummary.average_score === null ? 'Chưa có điểm' : `${Math.round(scoreSummary.average_score)}%`} - {scoreSummary.last_practice_at ?? 'Chưa có lần luyện'}
          </Text>
        ) : null}
      </View>
      {onViewScores ? (
        <Pressable accessibilityRole="button" onPress={onViewScores} style={styles.smallButton}>
          <Text style={styles.smallButtonText}>Xem điểm</Text>
        </Pressable>
      ) : null}
      <StatusBadge
        label={scoreSummary?.needs_review_count ? 'Cần xem lại' : scoreSummary ? 'Đang luyện tập' : classStatusLabel(student.status)}
        tone={scoreSummary?.needs_review_count ? 'warning' : scoreSummary ? 'processing' : 'success'}
      />
    </View>
  );
}

function StudentScoresPanel({
  student,
  scores,
  loading,
  error,
}: {
  student: ClassStudent | null;
  scores: TeacherStudentScoresResponse | null;
  loading: boolean;
  error: string | null;
}) {
  if (!student) {
    return (
      <AppCard style={styles.panel}>
        <Text style={styles.cardTitle}>Điểm và kết quả luyện tập</Text>
        <Text style={styles.cardDescription}>Chọn một học sinh rồi bấm Xem điểm để xem kết quả luyện tập gần đây.</Text>
      </AppCard>
    );
  }

  const pronunciationIssues = buildStudentPronunciationIssues(scores);

  return (
    <AppCard style={styles.panel}>
      <Text style={styles.cardTitle}>{student.display_name}</Text>
      <Text style={styles.cardDescription}>{student.email ?? 'Chưa có email'}</Text>
      {loading ? <Text style={styles.mutedText}>Đang tải kết quả luyện tập...</Text> : null}
      {error ? <ErrorState title="Chưa thể tải điểm" message={error} /> : null}
      {scores ? (
        <View style={styles.reportRows}>
          <ReportRow label="Tổng lượt luyện" value={String(scores.summary.total_attempts)} />
          <ReportRow label="Đã có kết quả" value={String(scores.summary.completed_attempts)} />
          <ReportRow label="Điểm trung bình" value={scores.summary.average_score === null ? 'Chưa có điểm' : `${Math.round(scores.summary.average_score)}%`} />
          <ReportRow label="Lần luyện gần nhất" value={scores.summary.last_practice_at ?? 'Chưa có dữ liệu'} />
        </View>
      ) : null}
      <View style={styles.assignmentDetailSection}>
        <Text style={styles.cardTitle}>Lỗi phát âm cần chú ý</Text>
        {pronunciationIssues.length > 0 ? (
          <View style={styles.errorRows}>
            {pronunciationIssues.map((issue) => (
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
          <EmptyPanel
            icon="microphone-off"
            title="Chưa có đủ dữ liệu để tổng hợp lỗi phát âm của học sinh này."
            message="Phoenix sẽ hiển thị lỗi cần chú ý khi học sinh có kết quả luyện tập hoàn tất."
          />
        )}
      </View>
      <View style={styles.studentRows}>
        {scores?.items.map((item) => (
          <View key={item.practice_history_id} style={styles.classSummaryRow}>
            <Text style={styles.className}>{item.target_word ?? 'Bài luyện'}</Text>
            <Text style={styles.mutedText}>{formatPracticeStatus(item.status)}</Text>
            <Text style={styles.reportValue}>{item.score === null ? 'Chưa có điểm' : `${Math.round(item.score)}%`}</Text>
            {item.needs_review ? <StatusBadge label="Cần xem lại" tone="warning" /> : null}
          </View>
        ))}
        {!loading && scores && scores.items.length === 0 ? (
          <EmptyPanel
            icon="chart-box-outline"
            title="Học sinh chưa có kết quả luyện tập hoàn tất"
            message="Khi học sinh hoàn thành bài luyện và AI trả kết quả, danh sách sẽ hiển thị tại đây."
          />
        ) : null}
      </View>
    </AppCard>
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
    gap: 16,
  },
  sideColumn: {
    flexGrow: 1,
    flexBasis: 320,
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
    minHeight: 36,
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
  assignmentList: {
    gap: 12,
  },
  assignmentWorkspaceGrid: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 18,
  },
  assignmentStudentColumn: {
    flexGrow: 0,
    flexShrink: 1,
    flexBasis: '36%',
    minWidth: 300,
  },
  assignmentDetailColumn: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: '62%',
    minWidth: 420,
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
});
