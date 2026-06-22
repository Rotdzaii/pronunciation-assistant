import { useEffect, useMemo, useState } from 'react';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { usePathname } from 'expo-router';
import { Alert, Platform, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import {
  AppCard,
  AppScreen,
  ErrorState,
  LoadingState,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import {
  fetchAdminClassDetail,
  fetchAdminClasses,
  fetchAdminDemoReadiness,
  fetchAdminUsers,
  updateAdminClass,
  deleteAdminUser,
} from '../../lib/api';
import { formatMembershipStatus, formatReadinessStatus, formatTeacherRole, formatUserRole } from '../../lib/format';
import { useAuth } from '../../lib/auth';
import type {
  AdminClassUpdatePayload,
  AdminMenuKey,
  AdminUser,
  ClassDetail,
  ClassSummary,
  DemoReadinessResponse,
} from '../../types';

type IconName = keyof typeof MaterialCommunityIcons.glyphMap;
type RoleFilter = 'all' | 'student' | 'teacher' | 'admin';

const adminMenus: Array<{ key: AdminMenuKey; label: string; icon: IconName }> = [
  { key: 'overview', label: 'Tổng quan', icon: 'view-dashboard-outline' },
  { key: 'classes', label: 'Quản lý lớp', icon: 'school-outline' },
  { key: 'delete_users', label: 'Xóa người dùng', icon: 'account-remove-outline' },
  { key: 'exports', label: 'Xuất báo cáo', icon: 'download-outline' },
];

const protectedDemoEmails = new Set([
  'admin@gmail.com',
  'hocsinh07@phoenix.edu.vn',
  'giangvien03@phoenix.edu.vn',
]);

export default function AdminScreen() {
  const pathname = usePathname();
  const { appRole, currentUser } = useAuth();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [classes, setClasses] = useState<ClassSummary[]>([]);
  const [readiness, setReadiness] = useState<DemoReadinessResponse | null>(null);
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);
  const [selectedClass, setSelectedClass] = useState<ClassDetail | null>(null);
  const [activeMenu, setActiveMenu] = useState<AdminMenuKey>('overview');
  const [userQuery, setUserQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState<RoleFilter>('all');
  const [classForm, setClassForm] = useState<AdminClassUpdatePayload>({});
  const [savingClass, setSavingClass] = useState(false);
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  console.debug('[AdminScreen] render', {
    usersCount: users.length,
    classesCount: classes.length,
    activeMenu,
    selectedClassId,
  });

  const loadAdminData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [userList, classList, demoReadiness] = await Promise.all([
        fetchAdminUsers(),
        fetchAdminClasses(),
        fetchAdminDemoReadiness(),
      ]);
      setUsers(userList);
      setClasses(classList);
      setReadiness(demoReadiness);
      setSelectedClassId((current) => current ?? classList[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chưa thể tải dữ liệu quản trị.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.debug('[AdminScreen] load data effect');
    void loadAdminData();
  }, []);

  useEffect(() => {
    console.debug('[AdminScreen] class detail effect', { selectedClassId });
    if (!selectedClassId) {
      setSelectedClass(null);
      return;
    }

    const loadClassDetail = async () => {
      setDetailLoading(true);
      setError(null);
      try {
        const detail = await fetchAdminClassDetail(selectedClassId);
        setSelectedClass(detail);
        setClassForm({
          code: detail.code ?? '',
          name: detail.name,
          description: detail.description ?? '',
          status: detail.status ?? 'active',
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Chưa thể tải chi tiết lớp.');
      } finally {
        setDetailLoading(false);
      }
    };

    void loadClassDetail();
  }, [selectedClassId]);

  const roleCounts = useMemo(() => {
    return {
      students: users.filter((user) => user.app_role === 'student').length,
      teachers: users.filter((user) => user.app_role === 'teacher').length,
      admins: users.filter((user) => user.app_role === 'admin').length,
    };
  }, [users]);

  useEffect(() => {
    console.debug('[AdminScreen] runtime', {
      adminUsersLength: users.length,
      adminClassesLength: classes.length,
      roleCounts,
      currentRole: appRole,
      currentRoute: pathname,
    });
  }, [appRole, classes.length, pathname, roleCounts, users.length]);

  const filteredUsers = useMemo(() => {
    const needle = userQuery.trim().toLowerCase();
    return users.filter((user) => {
      const matchesRole = roleFilter === 'all' || user.app_role === roleFilter;
      const haystack = `${user.display_name ?? ''} ${user.email ?? ''}`.toLowerCase();
      return matchesRole && (!needle || haystack.includes(needle));
    });
  }, [roleFilter, userQuery, users]);

  const adminCount = roleCounts.admins;

  const saveClass = async () => {
    if (!selectedClassId) return;
    setSavingClass(true);
    setError(null);
    setNotice(null);
    try {
      await updateAdminClass(selectedClassId, classForm);
      const [classList, classDetail] = await Promise.all([
        fetchAdminClasses(),
        fetchAdminClassDetail(selectedClassId),
      ]);
      setClasses(classList);
      setSelectedClass(classDetail);
      setClassForm({
        code: classDetail.code ?? '',
        name: classDetail.name,
        description: classDetail.description ?? '',
        status: classDetail.status ?? 'active',
      });
      setNotice('Đã lưu thông tin lớp và tải lại dữ liệu mới nhất.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chưa thể lưu thông tin lớp.');
    } finally {
      setSavingClass(false);
    }
  };

  const requestDeleteUser = (user: AdminUser) => {
    if (user.id === currentUser?.id) {
      setError('Không thể xóa chính tài khoản đang đăng nhập.');
      return;
    }
    if (user.app_role === 'admin' && adminCount <= 1) {
      setError('Không thể xóa quản trị viên cuối cùng trong hệ thống.');
      return;
    }
    if (user.email && protectedDemoEmails.has(user.email.toLowerCase())) {
      setError('Không xóa tài khoản demo chính trong màn này.');
      return;
    }

    const message = `Bạn sắp xóa người dùng ${displayUserName(user)} (${user.email ?? 'chưa có email'}). Thao tác này nguy hiểm và sẽ xóa membership liên quan.`;
    if (Platform.OS === 'web') {
      if (typeof window !== 'undefined' && !window.confirm(message)) {
        return;
      }
      void performDeleteUser(user);
      return;
    }

    Alert.alert('Xác nhận xóa người dùng', message, [
      { text: 'Hủy', style: 'cancel' },
      { text: 'Xóa', style: 'destructive', onPress: () => void performDeleteUser(user) },
    ]);
  };

  const performDeleteUser = async (user: AdminUser) => {
    setDeletingUserId(user.id);
    setError(null);
    setNotice(null);
    try {
      await deleteAdminUser(user.id);
      setNotice(`Đã xóa người dùng ${user.email ?? user.id}.`);
      await loadAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chưa thể xóa người dùng.');
    } finally {
      setDeletingUserId(null);
    }
  };

  const exportCsv = (kind: 'users' | 'classes' | 'readiness') => {
    if (Platform.OS !== 'web') {
      setNotice('Chức năng xuất CSV hiện hỗ trợ tốt nhất trên phiên bản web.');
      return;
    }

    const report = buildCsvReport(kind, users, classes, readiness);
    downloadCsv(report.filename, report.csv);
    setNotice(`Đã tạo file ${report.filename}.`);
  };

  return (
    <AppScreen maxWidth={1200}>
      <View style={styles.header}>
        <View>
          <Text style={styles.kicker}>Phoenix</Text>
          <Text style={styles.title}>Quản trị Phoenix</Text>
          <Text style={styles.subtitle}>Theo dõi người dùng, lớp học và trạng thái dữ liệu demo.</Text>
        </View>
        <StatusBadge
          label={readiness?.passed ? 'Dữ liệu demo đạt' : 'Cần kiểm tra dữ liệu demo'}
          tone={readiness?.passed ? 'success' : 'warning'}
        />
      </View>

      {loading ? (
        <LoadingState title="Đang tải dữ liệu quản trị" message="Hệ thống đang lấy người dùng, lớp học và dữ liệu demo." />
      ) : null}
      {error ? <ErrorState title="Lỗi quản trị" message={error} /> : null}

      {notice ? <StatusNotice message={notice} /> : null}

      <View style={styles.menuRow}>
        {adminMenus.map((item) => (
          <Pressable
            key={item.key}
            accessibilityRole="button"
            onPress={() => setActiveMenu(item.key)}
            style={[styles.menuButton, activeMenu === item.key ? styles.menuButtonActive : null]}
          >
            <MaterialCommunityIcons name={item.icon} size={18} color={activeMenu === item.key ? '#FFFFFF' : colors.primary} />
            <Text style={[styles.menuButtonText, activeMenu === item.key ? styles.menuButtonTextActive : null]}>
              {item.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {activeMenu === 'overview' ? (
        <AdminOverviewSection
          users={users}
          classes={classes}
          readiness={readiness}
          roleCounts={roleCounts}
          selectedClass={selectedClass}
          selectedClassId={selectedClassId}
          detailLoading={detailLoading}
          onSelectClass={setSelectedClassId}
        />
      ) : null}

      {activeMenu === 'classes' ? (
        <ClassManagementSection
          classes={classes}
          selectedClass={selectedClass}
          selectedClassId={selectedClassId}
          detailLoading={detailLoading}
          classForm={classForm}
          saving={savingClass}
          onSelectClass={setSelectedClassId}
          onChangeForm={setClassForm}
          onSave={saveClass}
        />
      ) : null}

      {activeMenu === 'delete_users' ? (
        <DeleteUsersSection
          users={filteredUsers}
          userQuery={userQuery}
          roleFilter={roleFilter}
          totalUsers={users.length}
          currentUserId={currentUser?.id ?? null}
          adminCount={adminCount}
          deletingUserId={deletingUserId}
          onChangeQuery={setUserQuery}
          onChangeRoleFilter={setRoleFilter}
          onDelete={requestDeleteUser}
        />
      ) : null}

      {activeMenu === 'exports' ? (
        <ExportReportsSection onExport={exportCsv} />
      ) : null}
    </AppScreen>
  );
}

function AdminMetric({ icon, label, value }: { icon: IconName; label: string; value: number }) {
  return (
    <AppCard style={styles.metricCard}>
      <MaterialCommunityIcons name={icon} size={24} color={colors.primary} />
      <Text style={styles.metricValue}>{value}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </AppCard>
  );
}

function StatusNotice({ message }: { message: string }) {
  return (
    <View style={styles.notice}>
      <MaterialCommunityIcons name="check-circle-outline" size={18} color="#047857" />
      <Text style={styles.noticeText}>{message}</Text>
    </View>
  );
}

function AdminOverviewSection({
  users,
  classes,
  readiness,
  roleCounts,
  selectedClass,
  selectedClassId,
  detailLoading,
  onSelectClass,
}: {
  users: AdminUser[];
  classes: ClassSummary[];
  readiness: DemoReadinessResponse | null;
  roleCounts: { students: number; teachers: number; admins: number };
  selectedClass: ClassDetail | null;
  selectedClassId: string | null;
  detailLoading: boolean;
  onSelectClass: (classId: string) => void;
}) {
  return (
    <View style={styles.sectionStack}>
      <View style={styles.metricsGrid}>
        <AdminMetric icon="account-group-outline" label="Tổng người dùng" value={users.length} />
        <AdminMetric icon="school-outline" label="Lớp học" value={classes.length} />
        <AdminMetric icon="account-child-outline" label="Học sinh" value={roleCounts.students} />
        <AdminMetric icon="account-tie-outline" label="Giảng viên" value={roleCounts.teachers} />
        <AdminMetric icon="shield-account-outline" label="Quản trị viên" value={roleCounts.admins} />
      </View>

      <View style={styles.contentGrid}>
        <AppCard style={[styles.panel, styles.mainColumn]}>
          <View style={styles.panelHeader}>
            <View>
              <Text style={styles.panelTitle}>Lớp học</Text>
              <Text style={styles.mutedText}>Chọn một lớp để xem giáo viên, học sinh và trạng thái.</Text>
            </View>
          </View>
          <ClassList classes={classes} selectedClassId={selectedClassId} onSelectClass={onSelectClass} />
        </AppCard>

        <View style={styles.sideColumn}>
          <ClassDetailPanel selectedClass={selectedClass} detailLoading={detailLoading} />
          <ReadinessPanel readiness={readiness} />
        </View>
      </View>
    </View>
  );
}

function ClassManagementSection({
  classes,
  selectedClass,
  selectedClassId,
  detailLoading,
  classForm,
  saving,
  onSelectClass,
  onChangeForm,
  onSave,
}: {
  classes: ClassSummary[];
  selectedClass: ClassDetail | null;
  selectedClassId: string | null;
  detailLoading: boolean;
  classForm: AdminClassUpdatePayload;
  saving: boolean;
  onSelectClass: (classId: string) => void;
  onChangeForm: (form: AdminClassUpdatePayload) => void;
  onSave: () => void;
}) {
  const canSave = Boolean(selectedClassId && classForm.name?.trim());

  return (
    <View style={styles.contentGrid}>
      <AppCard style={[styles.panel, styles.mainColumn]}>
        <View style={styles.panelHeader}>
          <View>
            <Text style={styles.panelTitle}>Quản lý lớp</Text>
            <Text style={styles.mutedText}>Sửa mã lớp, tên lớp và mô tả qua API quản trị.</Text>
          </View>
        </View>
        <ClassList classes={classes} selectedClassId={selectedClassId} onSelectClass={onSelectClass} />
      </AppCard>

      <AppCard style={[styles.panel, styles.sideColumn]}>
        <Text style={styles.panelTitle}>Thông tin lớp</Text>
        {detailLoading ? <Text style={styles.mutedText}>Đang tải chi tiết lớp...</Text> : null}
        {selectedClass ? (
          <View style={styles.formStack}>
            <LabeledInput
              label="Mã lớp"
              value={classForm.code ?? ''}
              onChangeText={(code) => onChangeForm({ ...classForm, code })}
              placeholder="Ví dụ: PHOENIX-A"
            />
            <LabeledInput
              label="Tên lớp"
              value={classForm.name ?? ''}
              onChangeText={(name) => onChangeForm({ ...classForm, name })}
              placeholder="Tên lớp"
            />
            <LabeledInput
              label="Mô tả"
              value={classForm.description ?? ''}
              onChangeText={(description) => onChangeForm({ ...classForm, description })}
              placeholder="Mô tả ngắn"
              multiline
            />
            <Pressable
              accessibilityRole="button"
              disabled={!canSave || saving}
              onPress={onSave}
              style={[styles.primaryButton, !canSave || saving ? styles.disabledButton : null]}
            >
              <MaterialCommunityIcons name="content-save-outline" size={18} color="#FFFFFF" />
              <Text style={styles.primaryButtonText}>{saving ? 'Đang lưu...' : 'Lưu thay đổi'}</Text>
            </Pressable>
          </View>
        ) : (
          <Text style={styles.mutedText}>Chọn một lớp để chỉnh sửa.</Text>
        )}
      </AppCard>
    </View>
  );
}

function DeleteUsersSection({
  users,
  userQuery,
  roleFilter,
  totalUsers,
  currentUserId,
  adminCount,
  deletingUserId,
  onChangeQuery,
  onChangeRoleFilter,
  onDelete,
}: {
  users: AdminUser[];
  userQuery: string;
  roleFilter: RoleFilter;
  totalUsers: number;
  currentUserId: string | null;
  adminCount: number;
  deletingUserId: string | null;
  onChangeQuery: (query: string) => void;
  onChangeRoleFilter: (role: RoleFilter) => void;
  onDelete: (user: AdminUser) => void;
}) {
  const filters: Array<{ key: RoleFilter; label: string }> = [
    { key: 'all', label: 'Tất cả' },
    { key: 'student', label: 'Học sinh' },
    { key: 'teacher', label: 'Giảng viên' },
    { key: 'admin', label: 'Quản trị viên' },
  ];

  return (
    <AppCard style={styles.panel}>
      <View style={styles.panelHeader}>
        <View>
          <Text style={styles.panelTitle}>Xóa người dùng</Text>
          <Text style={styles.mutedText}>Xóa tài khoản qua endpoint admin, có guard ở backend.</Text>
          <Text style={styles.mutedText}>Hiển thị {users.length}/{totalUsers} người dùng.</Text>
        </View>
      </View>

      <TextInput
        value={userQuery}
        onChangeText={onChangeQuery}
        placeholder="Tìm theo tên hoặc email"
        placeholderTextColor="#94A3B8"
        style={styles.input}
      />

      <View style={styles.filterRow}>
        {filters.map((filter) => (
          <Pressable
            key={filter.key}
            accessibilityRole="button"
            onPress={() => onChangeRoleFilter(filter.key)}
            style={[styles.filterButton, roleFilter === filter.key ? styles.filterButtonActive : null]}
          >
            <Text style={[styles.filterButtonText, roleFilter === filter.key ? styles.filterButtonTextActive : null]}>
              {filter.label}
            </Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.userRows}>
        {users.map((user) => {
          const isDeleting = deletingUserId === user.id;
          return (
            <View key={user.id} style={styles.deleteUserRow}>
              <View style={styles.userAvatar}>
                <Text style={styles.userInitial}>{displayUserName(user).slice(0, 1).toUpperCase()}</Text>
              </View>
              <View style={styles.userInfo}>
                <Text style={styles.userName}>{displayUserName(user)}</Text>
                <Text style={styles.userEmail}>{user.email ?? 'Chưa có email'}</Text>
              </View>
              <StatusBadge label={roleLabel(user.app_role)} tone={roleTone(user.app_role)} style={styles.roleBadge} />
              <Pressable
                accessibilityRole="button"
                disabled={isDeleting}
                onPress={() => onDelete(user)}
                style={[styles.dangerButton, isDeleting ? styles.disabledButton : null]}
              >
                <MaterialCommunityIcons name="trash-can-outline" size={17} color="#FFFFFF" />
                <Text style={styles.dangerButtonText}>{isDeleting ? 'Đang xóa' : 'Xóa'}</Text>
              </Pressable>
            </View>
          );
        })}
        {users.length === 0 ? <Text style={styles.mutedText}>Không có người dùng phù hợp.</Text> : null}
      </View>
    </AppCard>
  );
}

function ExportReportsSection({ onExport }: { onExport: (kind: 'users' | 'classes' | 'readiness') => void }) {
  return (
    <AppCard style={styles.panel}>
      <Text style={styles.panelTitle}>Xuất báo cáo CSV</Text>
      <Text style={styles.mutedText}>Xuất dữ liệu người dùng, lớp học và demo readiness trên web.</Text>
      <View style={styles.exportGrid}>
        <ExportButton icon="account-group-outline" label="Báo cáo người dùng" onPress={() => onExport('users')} />
        <ExportButton icon="school-outline" label="Báo cáo lớp học" onPress={() => onExport('classes')} />
        <ExportButton icon="clipboard-check-outline" label="Báo cáo dữ liệu demo" onPress={() => onExport('readiness')} />
      </View>
    </AppCard>
  );
}

function ExportButton({ icon, label, onPress }: { icon: IconName; label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.exportButton}>
      <MaterialCommunityIcons name={icon} size={22} color={colors.primary} />
      <Text style={styles.exportButtonText}>{label}</Text>
      <MaterialCommunityIcons name="download-outline" size={18} color={colors.muted} />
    </Pressable>
  );
}

function ClassList({
  classes,
  selectedClassId,
  onSelectClass,
}: {
  classes: ClassSummary[];
  selectedClassId: string | null;
  onSelectClass: (classId: string) => void;
}) {
  return (
    <View style={styles.classRows}>
      <Text style={styles.mutedText}>Hiển thị {classes.length}/{classes.length} lớp học.</Text>
      {classes.map((classItem) => (
        <Pressable
          key={classItem.id}
          accessibilityRole="button"
          onPress={() => onSelectClass(classItem.id)}
          style={[styles.classRow, selectedClassId === classItem.id ? styles.classRowActive : null]}
        >
          <View style={styles.classMain}>
            <Text style={styles.classCode}>{classItem.code ?? 'CHƯA CÓ MÃ'}</Text>
            <Text style={styles.className}>{classItem.name}</Text>
            <Text style={styles.mutedText}>{formatMembershipStatus(classItem.status)}</Text>
          </View>
          <View style={styles.classCounts}>
            <Text style={styles.countText}>{classItem.student_count} học sinh</Text>
            <Text style={styles.countText}>{classItem.teacher_count} giảng viên</Text>
          </View>
        </Pressable>
      ))}
      {classes.length === 0 ? <Text style={styles.mutedText}>Chưa có lớp học.</Text> : null}
    </View>
  );
}

function ClassDetailPanel({
  selectedClass,
  detailLoading,
}: {
  selectedClass: ClassDetail | null;
  detailLoading: boolean;
}) {
  if (detailLoading) {
    return (
      <AppCard style={styles.panel}>
        <Text style={styles.panelTitle}>Chi tiết lớp</Text>
        <Text style={styles.mutedText}>Đang tải chi tiết lớp...</Text>
      </AppCard>
    );
  }

  if (!selectedClass) {
    return (
      <AppCard style={styles.panel}>
        <Text style={styles.panelTitle}>Chi tiết lớp</Text>
        <Text style={styles.mutedText}>Chọn một lớp để xem chi tiết.</Text>
      </AppCard>
    );
  }

  return (
    <AppCard style={styles.panel}>
      <View style={styles.detailSummary}>
        <Text style={styles.detailCode}>{selectedClass.code ?? 'CHƯA CÓ MÃ'}</Text>
        <Text style={styles.detailName}>{selectedClass.name}</Text>
        <Text style={styles.mutedText}>{selectedClass.description ?? 'Chưa có mô tả.'}</Text>
        <View style={styles.detailStats}>
          <SmallDetailStat label="Học sinh" value={selectedClass.student_count} />
          <SmallDetailStat label="Giảng viên" value={selectedClass.teacher_count} />
        </View>
      </View>
      <MemberList title="Giảng viên" items={selectedClass.teachers ?? []} teacher />
      <MemberList title="Học sinh" items={selectedClass.students ?? []} />
    </AppCard>
  );
}

function ReadinessPanel({ readiness }: { readiness: DemoReadinessResponse | null }) {
  return (
    <AppCard style={styles.panel}>
      <View style={styles.panelHeader}>
        <View>
          <Text style={styles.panelTitle}>Demo readiness</Text>
          <Text style={styles.mutedText}>{formatReadinessStatus(readiness?.passed)}</Text>
        </View>
        <StatusBadge label={formatReadinessStatus(readiness?.passed)} tone={readiness?.passed ? 'success' : 'warning'} />
      </View>
      <View style={styles.checkRows}>
        {(readiness?.checks ?? []).map((check) => (
          <View key={check.key} style={styles.checkRow}>
            <MaterialCommunityIcons
              name={check.passed ? 'check-circle-outline' : 'alert-circle-outline'}
              size={20}
              color={check.passed ? '#047857' : '#D97706'}
            />
            <View style={styles.checkCopy}>
              <Text style={styles.checkKey}>{check.key}</Text>
              <Text style={styles.mutedText}>{check.message}</Text>
            </View>
          </View>
        ))}
        {!readiness ? <Text style={styles.mutedText}>Chưa có dữ liệu readiness.</Text> : null}
      </View>
    </AppCard>
  );
}

function LabeledInput({
  label,
  value,
  onChangeText,
  placeholder,
  multiline = false,
}: {
  label: string;
  value: string;
  onChangeText: (value: string) => void;
  placeholder: string;
  multiline?: boolean;
}) {
  return (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#94A3B8"
        multiline={multiline}
        style={[styles.input, multiline ? styles.textArea : null]}
      />
    </View>
  );
}

function SmallDetailStat({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.smallDetailStat}>
      <Text style={styles.smallDetailValue}>{value}</Text>
      <Text style={styles.smallDetailLabel}>{label}</Text>
    </View>
  );
}

function MemberList({
  title,
  items,
  teacher = false,
}: {
  title: string;
  items: Array<{ id: string; display_name: string; email: string | null; status: string | null; teacher_role?: string | null }>;
  teacher?: boolean;
}) {
  return (
    <View style={styles.memberList}>
      <Text style={styles.memberTitle}>{title}</Text>
      {items.map((item) => (
        <View key={item.id} style={styles.memberRow}>
          <View style={styles.memberInfo}>
            <Text style={styles.memberName}>{item.display_name}</Text>
            <Text style={styles.memberEmail}>{item.email ?? 'Chưa có email'}</Text>
          </View>
          <Text style={styles.memberStatus}>
            {teacher ? formatTeacherRole(item.teacher_role) : formatMembershipStatus(item.status)}
          </Text>
        </View>
      ))}
      {items.length === 0 ? <Text style={styles.mutedText}>Chưa có dữ liệu.</Text> : null}
    </View>
  );
}

function displayUserName(user: AdminUser): string {
  if (user.display_name?.trim()) return user.display_name.trim();
  if (user.app_role === 'admin') return 'Quản trị viên';
  return user.email ?? 'Chưa cập nhật tên';
}

function roleLabel(role: string | null): string {
  return formatUserRole(role);
}

function roleTone(role: string | null): 'primary' | 'success' | 'warning' {
  if (role === 'admin') return 'warning';
  if (role === 'teacher') return 'primary';
  return 'success';
}

function buildCsvReport(
  kind: 'users' | 'classes' | 'readiness',
  users: AdminUser[],
  classes: ClassSummary[],
  readiness: DemoReadinessResponse | null,
): { filename: string; csv: string } {
  if (kind === 'users') {
    return {
      filename: 'phoenix_users_report.csv',
      csv: toCsv(
        ['Email', 'Tên hiển thị', 'Vai trò'],
        users.map((user) => [
          user.email ?? '',
          displayUserName(user),
          formatUserRole(user.app_role),
        ]),
      ),
    };
  }

  if (kind === 'classes') {
    return {
      filename: 'phoenix_classes_report.csv',
      csv: toCsv(
        ['Mã lớp', 'Tên lớp', 'Số học sinh', 'Số giảng viên'],
        classes.map((classItem) => [
          classItem.code ?? '',
          classItem.name,
          String(classItem.student_count),
          String(classItem.teacher_count),
        ]),
      ),
    };
  }

  return {
    filename: 'phoenix_demo_readiness_report.csv',
    csv: toCsv(
      ['Tiêu chí', 'Giá trị thực tế', 'Kỳ vọng', 'Trạng thái'],
      (readiness?.checks ?? []).map((check) => [
        check.key,
        stringifyCsvValue(check.actual),
        stringifyCsvValue(check.expected),
        formatReadinessStatus(check.passed),
      ]),
    ),
  };
}

function toCsv(headers: string[], rows: string[][]): string {
  return [headers, ...rows].map((row) => row.map(escapeCsvCell).join(',')).join('\r\n');
}

function stringifyCsvValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function escapeCsvCell(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function downloadCsv(filename: string, csv: string) {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8;' });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(url);
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
  },
  kicker: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
    textTransform: 'uppercase',
  },
  title: {
    color: colors.text,
    fontSize: 34,
    fontWeight: '900',
    lineHeight: 42,
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
  },
  sectionStack: {
    gap: 16,
  },
  notice: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#A7F3D0',
    backgroundColor: '#ECFDF5',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 12,
  },
  noticeText: {
    color: '#047857',
    fontSize: 13,
    fontWeight: '800',
  },
  menuRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  menuButton: {
    minHeight: 42,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  menuButtonActive: {
    borderColor: colors.primary,
    backgroundColor: colors.primary,
  },
  menuButtonText: {
    color: colors.primary,
    fontSize: 13,
    fontWeight: '900',
  },
  menuButtonTextActive: {
    color: '#FFFFFF',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  metricCard: {
    flexGrow: 1,
    flexBasis: 180,
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
    flexBasis: 360,
    gap: 16,
  },
  panel: {
    borderRadius: 12,
    gap: 14,
  },
  panelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 10,
  },
  panelTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  classRows: {
    gap: 10,
  },
  classRow: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 14,
    gap: 12,
  },
  classRowActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  classMain: {
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
    fontSize: 16,
    fontWeight: '900',
  },
  classCounts: {
    alignItems: 'flex-end',
    gap: 4,
  },
  countText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  detailGrid: {
    gap: 12,
  },
  detailSummary: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#BFDBFE',
    backgroundColor: '#F8FBFF',
    padding: 14,
    gap: 8,
  },
  detailCode: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  detailName: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  detailStats: {
    flexDirection: 'row',
    gap: 10,
  },
  smallDetailStat: {
    flex: 1,
    borderRadius: 10,
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: colors.border,
    padding: 10,
  },
  smallDetailValue: {
    color: colors.text,
    fontSize: 24,
    fontWeight: '900',
  },
  smallDetailLabel: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '800',
  },
  memberList: {
    gap: 8,
  },
  memberTitle: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  memberRow: {
    minHeight: 58,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    padding: 12,
  },
  memberInfo: {
    flex: 1,
    gap: 3,
  },
  memberName: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  memberEmail: {
    color: colors.muted,
    fontSize: 12,
  },
  memberStatus: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '900',
  },
  formStack: {
    gap: 12,
  },
  inputGroup: {
    gap: 6,
  },
  inputLabel: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  input: {
    minHeight: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    color: colors.text,
    fontSize: 14,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  textArea: {
    minHeight: 96,
    textAlignVertical: 'top',
  },
  filterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  filterButton: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  filterButtonActive: {
    borderColor: colors.primary,
    backgroundColor: colors.softBlue,
  },
  filterButtonText: {
    color: colors.muted,
    fontSize: 12,
    fontWeight: '900',
  },
  filterButtonTextActive: {
    color: colors.primary,
  },
  primaryButton: {
    minHeight: 44,
    borderRadius: 10,
    backgroundColor: colors.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 14,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '900',
  },
  dangerButton: {
    minHeight: 38,
    borderRadius: 10,
    backgroundColor: '#DC2626',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 12,
  },
  dangerButtonText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  disabledButton: {
    opacity: 0.5,
  },
  checkRows: {
    gap: 10,
  },
  checkRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  checkCopy: {
    flex: 1,
  },
  checkKey: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  userRows: {
    gap: 9,
  },
  deleteUserRow: {
    minHeight: 70,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 12,
  },
  userRow: {
    minHeight: 64,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  userAvatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  userInitial: {
    color: colors.primary,
    fontSize: 15,
    fontWeight: '900',
  },
  userInfo: {
    flex: 1,
    gap: 3,
  },
  userName: {
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  userEmail: {
    color: colors.muted,
    fontSize: 12,
  },
  roleBadge: {
    paddingHorizontal: 9,
    paddingVertical: 5,
  },
  exportGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  exportButton: {
    minHeight: 64,
    flexGrow: 1,
    flexBasis: 220,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    padding: 14,
  },
  exportButtonText: {
    flex: 1,
    color: colors.text,
    fontSize: 14,
    fontWeight: '900',
  },
  mutedText: {
    color: colors.muted,
    fontSize: 13,
    lineHeight: 19,
  },
});
