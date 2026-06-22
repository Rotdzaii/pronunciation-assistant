import { useEffect, useMemo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import {
  AppCard,
  AppScreen,
  EmptyState,
  ErrorState,
  LoadingState,
  SectionHeader,
  StatusBadge,
  colors,
} from '../../components/AppUI';
import { fetchTeacherClasses, fetchTeacherClassStudents } from '../../lib/api';
import type { ClassStudent, ClassSummary } from '../../types';

type StudentWithClass = ClassStudent & {
  class_id: string;
  class_name: string;
  class_code: string | null;
};

export default function StudentsScreen() {
  const [classes, setClasses] = useState<ClassSummary[]>([]);
  const [students, setStudents] = useState<StudentWithClass[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<string | 'all'>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.debug('[StudentsScreen] load students effect');
    const loadStudents = async () => {
      setLoading(true);
      setError(null);
      try {
        const classList = await fetchTeacherClasses();
        const rosterByClass = await Promise.all(
          classList.map(async (classItem) => {
            const classStudents = await fetchTeacherClassStudents(classItem.id);
            return classStudents.map((student) => ({
              ...student,
              class_id: classItem.id,
              class_name: classItem.name,
              class_code: classItem.code,
            }));
          }),
        );
        setClasses(classList);
        setStudents(rosterByClass.flat());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Chưa thể tải danh sách học viên.');
      } finally {
        setLoading(false);
      }
    };

    void loadStudents();
  }, []);

  const visibleStudents = useMemo(() => {
    if (selectedClassId === 'all') {
      return students;
    }
    return students.filter((student) => student.class_id === selectedClassId);
  }, [selectedClassId, students]);

  return (
    <AppScreen maxWidth={1100}>
      <SectionHeader
        eyebrow="Giáo viên"
        title="Danh sách học viên"
        subtitle="Danh sách học sinh đang hoạt động trong các lớp bạn phụ trách."
      />

      {loading ? <LoadingState title="Đang tải học viên" message="Hệ thống đang lấy học sinh theo từng lớp." /> : null}
      {error ? <ErrorState title="Chưa thể tải học viên" message={error} /> : null}

      {!loading && students.length === 0 ? (
        <EmptyState
          title="Chưa có học viên trong lớp"
          message="Khi lớp có học sinh đang hoạt động, danh sách sẽ xuất hiện tại đây."
        />
      ) : null}

      {students.length > 0 ? (
        <>
          <AppCard style={styles.filterCard}>
            <Text style={styles.filterTitle}>Lọc theo lớp</Text>
            <View style={styles.filterRow}>
              <FilterPill
                label="Tất cả lớp"
                active={selectedClassId === 'all'}
                onPress={() => setSelectedClassId('all')}
              />
              {classes.map((classItem) => (
                <FilterPill
                  key={classItem.id}
                  label={classItem.code ?? classItem.name}
                  active={selectedClassId === classItem.id}
                  onPress={() => setSelectedClassId(classItem.id)}
                />
              ))}
            </View>
          </AppCard>

          <AppCard style={styles.studentListCard}>
            <View style={styles.listHeader}>
              <Text style={styles.listTitle}>Học viên</Text>
              <StatusBadge label={`${visibleStudents.length} học viên`} tone="primary" />
            </View>
            <View style={styles.studentRows}>
              {visibleStudents.map((student) => (
                <View key={`${student.class_id}-${student.id}`} style={styles.studentRow}>
                  <View style={styles.avatar}>
                    <Text style={styles.avatarText}>{student.display_name.slice(0, 1).toUpperCase()}</Text>
                  </View>
                  <View style={styles.studentInfo}>
                    <Text style={styles.studentName}>{student.display_name}</Text>
                    <Text style={styles.studentEmail}>{student.email ?? 'Chưa có email'}</Text>
                    <Text style={styles.studentClass}>{student.class_code ?? 'Lớp'} - {student.class_name}</Text>
                  </View>
                  <StatusBadge label={student.status === 'active' ? 'Đang hoạt động' : student.status ?? 'Đang hoạt động'} tone="success" />
                </View>
              ))}
            </View>
          </AppCard>
        </>
      ) : null}
    </AppScreen>
  );
}

function FilterPill({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={[styles.filterPill, active ? styles.filterPillActive : null]}
    >
      <Text style={[styles.filterPillText, active ? styles.filterPillTextActive : null]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  filterCard: {
    borderRadius: 12,
    gap: 12,
  },
  filterTitle: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '900',
  },
  filterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  filterPill: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  filterPillActive: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  filterPillText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '900',
  },
  filterPillTextActive: {
    color: '#FFFFFF',
  },
  studentListCard: {
    borderRadius: 12,
    gap: 14,
  },
  listHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 12,
  },
  listTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '900',
  },
  studentRows: {
    gap: 10,
  },
  studentRow: {
    minHeight: 72,
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
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: colors.softBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: colors.primary,
    fontSize: 17,
    fontWeight: '900',
  },
  studentInfo: {
    flex: 1,
    gap: 3,
  },
  studentName: {
    color: colors.text,
    fontSize: 15,
    fontWeight: '900',
  },
  studentEmail: {
    color: colors.muted,
    fontSize: 12,
  },
  studentClass: {
    color: colors.primary,
    fontSize: 12,
    fontWeight: '800',
  },
});
