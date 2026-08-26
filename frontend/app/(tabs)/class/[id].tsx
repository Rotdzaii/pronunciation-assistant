import { useEffect, useState } from 'react';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { AppScreen, ErrorState, LoadingState, colors } from '../../../components/AppUI';
import { fetchStudentClassDetail } from '../../../lib/api';
import type { ClassDetail, ClassStudent } from '../../../types';

const AVATAR_PALETTE = [
  { bg: '#EFF6FF', fg: '#2563EB' },
  { bg: '#F0FDFA', fg: '#0D9488' },
  { bg: '#FFF7ED', fg: '#EA580C' },
  { bg: '#F5F3FF', fg: '#7C3AED' },
  { bg: '#FEF2F2', fg: '#DC2626' },
];

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function StudentAvatar({ student, index }: { student: ClassStudent; index: number }) {
  const pal = AVATAR_PALETTE[index % AVATAR_PALETTE.length];
  if (student.avatar_url) {
    return <Image source={{ uri: student.avatar_url }} style={styles.avatar} />;
  }
  return (
    <View style={[styles.avatar, { backgroundColor: pal.bg }]}>
      <Text style={[styles.avatarText, { color: pal.fg }]}>{initials(student.display_name)}</Text>
    </View>
  );
}

export default function ClassDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const [data, setData] = useState<ClassDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);
    fetchStudentClassDetail(id)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : 'Chưa thể tải thông tin lớp.'))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <AppScreen>
        <LoadingState title="Đang tải thông tin lớp" message="Vui lòng chờ trong giây lát." />
      </AppScreen>
    );
  }

  const students = data?.students ?? [];
  const teacher = data?.teachers?.[0] ?? null;

  return (
    <AppScreen maxWidth={760}>
      <Pressable accessibilityRole="button" onPress={() => router.back()} style={styles.backBtn}>
        <MaterialCommunityIcons name="arrow-left" size={20} color={colors.primary} />
        <Text style={styles.backText}>Quay lại</Text>
      </Pressable>

      {error || !data ? (
        <ErrorState
          title="Không thể tải thông tin lớp"
          message={error ?? 'Lớp học không tồn tại hoặc bạn không có quyền truy cập.'}
        />
      ) : (
        <>
          <View style={styles.header}>
            <Text style={styles.classTitle}>{data.name}</Text>
            {teacher ? (
              <Text style={styles.teacherLine}>GV: {teacher.display_name}</Text>
            ) : null}
          </View>

          <View style={styles.rosterCard}>
            <View style={styles.rosterCardHeader}>
              <Text style={styles.rosterHeading}>Danh sách lớp học</Text>
              <Text style={styles.rosterCount}>{students.length} học sinh</Text>
            </View>

            {students.length === 0 ? (
              <View style={styles.emptyState}>
                <MaterialCommunityIcons name="account-group-outline" size={44} color={colors.muted} />
                <Text style={styles.emptyTitle}>Lớp chưa có học sinh</Text>
                <Text style={styles.emptyText}>
                  Lớp này hiện chưa có học sinh nào được thêm vào.
                </Text>
              </View>
            ) : (
              students.map((student, idx) => (
                <View
                  key={student.id}
                  style={[styles.studentRow, idx > 0 ? styles.studentRowBorder : null]}
                >
                  <StudentAvatar student={student} index={idx} />
                  <Text style={styles.studentName}>{student.display_name}</Text>
                </View>
              ))
            )}
          </View>
        </>
      )}
    </AppScreen>
  );
}

const styles = StyleSheet.create({
  backBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    marginBottom: 8,
    alignSelf: 'flex-start',
  },
  backText: {
    color: colors.primary,
    fontSize: 14,
    fontWeight: '700',
  },
  header: {
    marginBottom: 24,
    gap: 6,
  },
  classTitle: {
    fontSize: 28,
    fontWeight: '900',
    color: colors.text,
    lineHeight: 34,
  },
  teacherLine: {
    fontSize: 15,
    fontWeight: '500',
    color: colors.muted,
  },
  rosterCard: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  rosterCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  rosterHeading: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.text,
  },
  rosterCount: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.muted,
  },
  studentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  studentRowBorder: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    fontSize: 14,
    fontWeight: '800',
  },
  studentName: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
    flex: 1,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 48,
    paddingHorizontal: 24,
    gap: 10,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: colors.text,
    marginTop: 8,
  },
  emptyText: {
    fontSize: 14,
    color: colors.muted,
    textAlign: 'center',
    lineHeight: 20,
  },
});
