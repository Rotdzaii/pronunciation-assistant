import { useCallback, useEffect, useState } from 'react';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { AppCard, AppScreen, ErrorState, LoadingState, StatusBadge, colors } from '../../../components/AppUI';
import { fetchAssignmentStatus, fetchAssignmentWords, fetchStudentAssignments } from '../../../lib/api';
import type { AssignmentStatus, AssignmentWords, StudentAssignment } from '../../../types';

export default function RegularAssignmentDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const [assignment, setAssignment] = useState<StudentAssignment | null>(null);
  const [words, setWords] = useState<AssignmentWords | null>(null);
  const [status, setStatus] = useState<AssignmentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    if (!id) return;
    setError(null);
    try {
      const [all, wordData, statusData] = await Promise.all([fetchStudentAssignments(), fetchAssignmentWords(id), fetchAssignmentStatus(id)]);
      const found = all.find((item) => item.assignment_id === id);
      if (!found || found.is_assessment) throw new Error('Không tìm thấy bài luyện được giao.');
      setAssignment(found); setWords(wordData); setStatus(statusData);
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể tải bài tập.'); }
  }, [id]);
  useEffect(() => { void load(); }, [load]);
  if (error) return <AppScreen><ErrorState title="Không thể mở bài tập" message={error} /><Pressable onPress={() => void load()} style={styles.action}><Text style={styles.actionText}>Thử lại</Text></Pressable></AppScreen>;
  if (!assignment || !words || !status) return <LoadingState title="Đang tải bài tập" message="Đang kiểm tra quyền truy cập và tiến độ." />;
  const canPractice = !assignment.is_overdue && status.can_start;
  return <AppScreen maxWidth={820}><ScrollView contentContainerStyle={styles.root}>
    <Pressable onPress={() => router.back()}><Text style={styles.back}>‹ Quay lại bài tập</Text></Pressable>
    <AppCard style={styles.hero}><Text style={styles.title}>{assignment.title}</Text><Text style={styles.meta}>{assignment.class_name ?? 'Bài tập cá nhân'} · {assignment.completed_items}/{assignment.total_items} từ đã có kết quả AI</Text>{assignment.description ? <Text style={styles.description}>{assignment.description}</Text> : null}<View style={styles.statuses}><StatusBadge label={assignment.is_overdue ? 'Quá hạn' : 'Có thể luyện'} tone={assignment.is_overdue ? 'error' : 'success'} /><StatusBadge label={assignment.final_score != null ? `Điểm ${Math.round(assignment.final_score)}/100` : assignment.grading_status === 'provisional' ? 'Điểm tạm tính' : 'Chưa có điểm'} tone="processing" /></View>{canPractice ? <Pressable onPress={() => router.push({ pathname: '/(tabs)/practice', params: { assignment_id: id } })} style={styles.action}><Text style={styles.actionText}>{assignment.work_status === 'in_progress' ? 'Tiếp tục luyện' : assignment.work_status === 'completed' ? 'Luyện lại' : 'Bắt đầu luyện'}</Text></Pressable> : null}</AppCard>
    <Text style={styles.section}>Từ được giao</Text><View style={styles.words}>{words.items.map((word, index) => <AppCard key={word.id} style={styles.word}><Text style={styles.number}>{index + 1}</Text><View><Text style={styles.wordText}>{word.word}</Text>{word.phonetic ? <Text style={styles.meta}>{word.phonetic}</Text> : null}{word.meaning_vi ? <Text style={styles.meta}>{word.meaning_vi}</Text> : null}</View></AppCard>)}</View>
  </ScrollView></AppScreen>;
}

const styles = StyleSheet.create({ root: { gap: 16, paddingBottom: 32 }, back: { color: colors.primary, fontWeight: '800' }, hero: { gap: 12 }, title: { color: colors.text, fontSize: 25, fontWeight: '900' }, meta: { color: colors.muted }, description: { color: colors.text, lineHeight: 20 }, statuses: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, action: { alignSelf: 'flex-start', backgroundColor: colors.primary, minHeight: 44, justifyContent: 'center', paddingHorizontal: 16, borderRadius: 8 }, actionText: { color: '#fff', fontWeight: '900' }, section: { color: colors.text, fontSize: 18, fontWeight: '900' }, words: { gap: 8 }, word: { flexDirection: 'row', gap: 12, alignItems: 'center' }, number: { color: colors.primary, fontWeight: '900', width: 24 }, wordText: { color: colors.text, fontSize: 16, fontWeight: '800' } });
