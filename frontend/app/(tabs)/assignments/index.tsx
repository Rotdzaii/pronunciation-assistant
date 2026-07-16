import { useCallback, useMemo, useState } from 'react';
import { useFocusEffect, useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, EmptyState, ErrorState, LoadingState, StatusBadge, colors } from '../../../components/AppUI';
import { fetchStudentAssignments } from '../../../lib/api';
import type { StudentAssignment } from '../../../types';

type Filter = 'all' | 'todo' | 'doing' | 'done';

const filters: Array<{ key: Filter; label: string }> = [
  { key: 'all', label: 'Tất cả' }, { key: 'todo', label: 'Chưa làm' },
  { key: 'doing', label: 'Đang làm' }, { key: 'done', label: 'Đã hoàn thành' },
];

export default function StudentAssignmentsScreen() {
  const router = useRouter();
  const [items, setItems] = useState<StudentAssignment[]>([]);
  const [filter, setFilter] = useState<Filter>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try { setItems(await fetchStudentAssignments()); }
    catch (err) { setError(err instanceof Error ? err.message : 'Không thể tải bài tập.'); }
    finally { setLoading(false); }
  }, []);
  // Tabs stay mounted after the student navigates away. Reload when focused so
  // assignments created during the same session are never hidden by old state.
  useFocusEffect(useCallback(() => {
    void load();
  }, [load]));

  const visible = useMemo(() => items.filter((item) => {
    if (filter === 'todo') return item.work_status === 'not_started';
    if (filter === 'doing') return item.work_status === 'in_progress';
    if (filter === 'done') return ['completed', 'submitted', 'locked'].includes(item.work_status);
    return true;
  }).sort((a, b) => rank(a) - rank(b) || dueAt(a) - dueAt(b)), [filter, items]);

  return <AppScreen maxWidth={900}>
    <View style={styles.header}><View><Text style={styles.title}>Bài tập</Text><Text style={styles.subtitle}>Bài giáo viên giao và tiến độ chấm điểm của bạn.</Text></View><Pressable onPress={() => void load()} style={styles.reload}><Text style={styles.reloadText}>Tải lại</Text></Pressable></View>
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filters}>{filters.map((item) => <Pressable key={item.key} onPress={() => setFilter(item.key)} style={[styles.filter, filter === item.key && styles.filterActive]}><Text style={[styles.filterText, filter === item.key && styles.filterTextActive]}>{item.label}</Text></Pressable>)}</ScrollView>
    {loading ? <LoadingState title="Đang tải bài tập" message="Vui lòng chờ trong giây lát." /> : null}
    {error ? <><ErrorState title="Chưa thể tải bài tập" message={error} /><Pressable onPress={() => void load()} style={styles.retry}><Text style={styles.retryText}>Thử lại</Text></Pressable></> : null}
    {!loading && !error && visible.length === 0 ? <EmptyState title="Chưa có bài tập" message="Các bài giáo viên giao sẽ xuất hiện tại đây." /> : null}
    <View style={styles.list}>{visible.map((item) => <AssignmentCard key={item.assignment_id} item={item} onPress={() => router.push(item.is_assessment ? { pathname: '/(tabs)/assessment', params: { assignment_id: item.assignment_id } } : { pathname: '/(tabs)/assignments/[id]', params: { id: item.assignment_id } } as never)} />)}</View>
  </AppScreen>;
}

function AssignmentCard({ item, onPress }: { item: StudentAssignment; onPress: () => void }) {
  const action = item.is_overdue ? 'Xem kết quả' : item.is_assessment ? (item.can_continue ? 'Bắt đầu / tiếp tục kiểm tra' : 'Xem kết quả') : item.work_status === 'completed' ? 'Luyện lại' : item.work_status === 'in_progress' ? 'Tiếp tục luyện' : 'Bắt đầu luyện';
  return <AppCard style={styles.card}><View style={styles.cardTop}><View style={styles.cardCopy}><Text style={styles.cardTitle}>{item.title}</Text><Text style={styles.meta}>{item.class_name ?? 'Bài tập cá nhân'} · {item.is_assessment ? 'Kiểm tra' : 'Luyện tập'}</Text></View><StatusBadge label={workLabel(item.work_status)} tone={item.is_overdue ? 'error' : item.work_status === 'completed' ? 'success' : 'primary'} /></View>
    {item.description ? <Text style={styles.description}>{item.description}</Text> : null}
    <Text style={styles.meta}>Tiến độ: {item.completed_items}/{item.total_items} từ{item.deadline ? ` · Hạn: ${formatDate(item.deadline)}` : ''}</Text>
    <View style={styles.cardBottom}><StatusBadge label={gradeLabel(item.grading_status, item.final_score)} tone={item.grading_status === 'needs_review' ? 'warning' : item.grading_status === 'graded' ? 'success' : 'processing'} /><Pressable onPress={onPress} style={[styles.action, item.is_overdue && !item.final_score ? styles.actionMuted : null]}><Text style={styles.actionText}>{action}</Text></Pressable></View>
  </AppCard>;
}

function rank(item: StudentAssignment) { return item.work_status === 'in_progress' ? 0 : item.work_status === 'not_started' ? 1 : item.is_overdue ? 3 : 2; }
function dueAt(item: StudentAssignment) { return item.deadline ? new Date(item.deadline).getTime() : Number.MAX_SAFE_INTEGER; }
function formatDate(value: string) { return new Intl.DateTimeFormat('vi-VN', { dateStyle: 'short', timeStyle: 'short' }).format(new Date(value)); }
function workLabel(status: StudentAssignment['work_status']) { return ({ not_started: 'Chưa làm', in_progress: 'Đang làm', completed: 'Hoàn thành', submitted: 'Đã nộp', overdue: 'Quá hạn', locked: 'Đã khóa' } as const)[status]; }
function gradeLabel(status: StudentAssignment['grading_status'], score: number | null) { return score != null ? `Điểm: ${Math.round(score)}/100` : ({ pending: 'Chờ làm bài', provisional: 'Điểm tạm tính', processing: 'Đang chấm', graded: 'Đã chấm', needs_review: 'Cần giáo viên xem' } as const)[status]; }

const styles = StyleSheet.create({ header: { flexDirection: 'row', justifyContent: 'space-between', gap: 16, alignItems: 'center', marginBottom: 16 }, title: { fontSize: 28, fontWeight: '900', color: colors.text }, subtitle: { color: colors.muted, marginTop: 4 }, reload: { paddingHorizontal: 14, minHeight: 40, justifyContent: 'center', borderRadius: 8, backgroundColor: colors.softBlue }, reloadText: { color: colors.primary, fontWeight: '800' }, filters: { gap: 8, paddingBottom: 16 }, filter: { paddingHorizontal: 14, minHeight: 38, justifyContent: 'center', borderRadius: 20, backgroundColor: '#F1F5F9' }, filterActive: { backgroundColor: colors.primary }, filterText: { color: colors.muted, fontWeight: '700' }, filterTextActive: { color: '#fff' }, list: { gap: 12 }, card: { gap: 10 }, cardTop: { flexDirection: 'row', justifyContent: 'space-between', gap: 10 }, cardCopy: { flex: 1 }, cardTitle: { fontSize: 17, fontWeight: '900', color: colors.text }, meta: { color: colors.muted, fontSize: 13 }, description: { color: colors.text, lineHeight: 20 }, cardBottom: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 10 }, action: { minHeight: 40, paddingHorizontal: 14, borderRadius: 8, justifyContent: 'center', backgroundColor: colors.primary }, actionMuted: { backgroundColor: colors.muted }, actionText: { color: '#fff', fontWeight: '800' }, retry: { marginTop: 12, alignSelf: 'flex-start', padding: 12, backgroundColor: colors.primary, borderRadius: 8 }, retryText: { color: '#fff', fontWeight: '800' } });
