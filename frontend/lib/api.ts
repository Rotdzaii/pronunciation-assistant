import type {
  Assignment,
  AssignmentDetail,
  AssignmentGradebookItem,
  AssignmentsListResponse,
  AssignmentStatus,
  AssignmentWords,
  AssessmentSubmission,
  AudioUploadResponse,
  AdminClassUpdatePayload,
  AdminDeleteUserResponse,
  AdminUser,
  ClassDetail,
  ClassStudent,
  ClassSummary,
  ClassTeacher,
  CreatePracticeJobResponse,
  CurrentUser,
  DemoReadinessResponse,
  PracticeHistoryQuery,
  PracticeHistoryResponse,
  PracticeHistoryAudioResponse,
  PracticeJob,
  ProfileAvatarUploadResponse,
  ReportPracticeResultPayload,
  StudentAssignment,
  StudentReviewRequest,
  StudentReviewRequestDetail,
  TeacherAnalyticsResponse,
  TeacherClassScoresResponse,
  TeacherReviewRequestDetail,
  TeacherReviewRequestsResponse,
  TeacherStudentScoresResponse,
  UpdateProfilePayload,
  UserProfile,
  VocabularyItem,
  VocabularySet,
  VocabularySetDetail,
} from '../types';
import { Platform } from 'react-native';
import { supabase } from './supabase';

const BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
const SESSION_EXPIRED_MESSAGE = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

type CreatePracticeJobPayload = {
  target_word: string;
  audio_url: string;
  assignment_id?: string | null;
  item_id?: string | null;
  audio_storage_path?: string | null;
};

type TeacherReviewRequestQuery = {
  class_id?: string | null;
  status?: string | null;
  source?: string | null;
  q?: string | null;
  limit?: number;
  offset?: number;
};

type TeacherScoresQuery = {
  class_id?: string | null;
  limit?: number;
  offset?: number;
  date_from?: string | null;
  date_to?: string | null;
};

async function parseError(response: Response): Promise<string> {
  const fallback = `Request failed with status ${response.status}`;

  try {
    const data = await response.json();
    if (typeof data?.detail === 'string') {
      return data.detail;
    }
    if (Array.isArray(data?.detail)) {
      return data.detail
        .map((item: { msg?: string; message?: string }) => item.msg || item.message || JSON.stringify(item))
        .join(', ');
    }
    if (typeof data?.message === 'string') {
      return data.message;
    }
    return JSON.stringify(data);
  } catch {
    const text = await response.text();
    return text || fallback;
  }
}

async function apiFetch<T>(
  path: string,
  accessToken: string | null,
  options: RequestInit = {},
): Promise<T> {
  // A just-created Supabase session can arrive before persisted storage and the
  // auth-state listener have caught up. Prefer the explicitly supplied token
  // so `/auth/me` never authenticates the signup request as a previous user.
  let currentAccessToken = accessToken;
  if (!currentAccessToken) {
    const { data } = await supabase.auth.getSession();
    currentAccessToken = data.session?.access_token ?? null;
  }

  if (!currentAccessToken) {
    throw new ApiError(SESSION_EXPIRED_MESSAGE, 401);
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options.headers || {}),
      Authorization: `Bearer ${currentAccessToken}`,
    },
  });

  if (!response.ok) {
    throw new ApiError(await parseError(response), response.status);
  }

  return response.json() as Promise<T>;
}

export async function getMe(accessToken: string | null): Promise<CurrentUser> {
  return apiFetch<CurrentUser>('/auth/me', accessToken);
}

export async function fetchMyProfile(): Promise<UserProfile> {
  return apiFetch<UserProfile>('/profile/me', null);
}

export async function updateMyProfile(payload: UpdateProfilePayload): Promise<UserProfile> {
  return apiFetch<UserProfile>('/profile/me', null, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

export async function uploadMyAvatar(file: File): Promise<ProfileAvatarUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  return apiFetch<ProfileAvatarUploadResponse>('/profile/avatar', null, {
    method: 'POST',
    body: formData,
  });
}

export async function uploadPracticeAudio(
  audioUri: string,
  accessToken: string | null,
): Promise<AudioUploadResponse> {
  const formData = new FormData();
  const endpoint = '/practice/upload-audio';

  console.debug('Uploading practice audio to', `${BASE_URL}${endpoint}`);
  console.debug('Audio upload platform', Platform.OS);

  if (Platform.OS === 'web') {
    const response = await fetch(audioUri);
    const blob = await response.blob();
    const file = new File([blob], 'recording.webm', {
      type: blob.type || 'audio/webm',
    });

    console.debug('Audio upload blob', {
      type: blob.type || 'audio/webm',
      size: blob.size,
    });

    formData.append('file', file);
  } else {
    formData.append('file', {
      uri: audioUri,
      name: 'recording.m4a',
      type: 'audio/m4a',
    } as any);
  }

  return apiFetch<AudioUploadResponse>(endpoint, accessToken, {
    method: 'POST',
    body: formData,
  });
}

export async function createPracticeJob(
  payload: CreatePracticeJobPayload,
  accessToken: string | null,
): Promise<CreatePracticeJobResponse> {
  return apiFetch<CreatePracticeJobResponse>('/practice/create-job', accessToken, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

export async function fetchPracticeStatus(
  jobId: string,
  accessToken: string | null,
): Promise<PracticeJob> {
  return apiFetch<PracticeJob>(`/practice/${jobId}`, accessToken);
}

export async function fetchPracticeHistoryAudio(
  practiceId: string,
  accessToken: string | null,
): Promise<PracticeHistoryAudioResponse> {
  return apiFetch<PracticeHistoryAudioResponse>(
    `/practice/history/${encodeURIComponent(practiceId)}/audio`,
    accessToken,
  );
}

export async function fetchPracticeHistory(
  accessToken: string | null,
  query: PracticeHistoryQuery = {},
): Promise<PracticeJob[]> {
  const params = new URLSearchParams();
  params.set('limit', String(query.limit ?? 20));
  params.set('offset', String(query.offset ?? 0));
  if (query.status) {
    params.set('status', query.status);
  }

  const response = await apiFetch<PracticeHistoryResponse>(
    `/practice/history?${params.toString()}`,
    accessToken,
  );
  return response.items;
}

export async function reportStudentPracticeResult(
  practiceHistoryId: string,
  payload: ReportPracticeResultPayload,
): Promise<StudentReviewRequest> {
  return apiFetch<StudentReviewRequest>(`/student/practice-history/${practiceHistoryId}/report`, null, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function fetchStudentReviewRequests(): Promise<StudentReviewRequest[]> {
  return apiFetch<StudentReviewRequest[]>('/student/review-requests', null);
}

export async function fetchStudentReviewRequestDetail(
  id: string,
): Promise<StudentReviewRequestDetail> {
  return apiFetch<StudentReviewRequestDetail>(`/student/review-requests/${id}`, null);
}

export async function fetchTeacherAnalytics(
  accessToken: string | null,
): Promise<TeacherAnalyticsResponse> {
  return apiFetch<TeacherAnalyticsResponse>('/teacher/analytics', accessToken);
}

function buildQuery(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      search.set(key, String(value));
    }
  });
  const query = search.toString();
  return query ? `?${query}` : '';
}

export async function fetchTeacherReviewRequests(
  params: TeacherReviewRequestQuery = {},
): Promise<TeacherReviewRequestsResponse> {
  return apiFetch<TeacherReviewRequestsResponse>(
    `/teacher/review-requests${buildQuery(params)}`,
    null,
  );
}

export async function fetchTeacherReviewRequestDetail(
  id: string,
): Promise<TeacherReviewRequestDetail> {
  return apiFetch<TeacherReviewRequestDetail>(`/teacher/review-requests/${id}`, null);
}

export async function addTeacherReviewNote(
  id: string,
  payload: { teacher_note: string },
): Promise<TeacherReviewRequestDetail> {
  return apiFetch<TeacherReviewRequestDetail>(`/teacher/review-requests/${id}/note`, null, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function resolveTeacherReviewRequest(
  id: string,
  payload: { status: 'resolved' | 'rejected'; teacher_resolution: string; teacher_note?: string | null },
): Promise<TeacherReviewRequestDetail> {
  return apiFetch<TeacherReviewRequestDetail>(`/teacher/review-requests/${id}/resolve`, null, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function requestTeacherReviewReanalysis(
  id: string,
): Promise<TeacherReviewRequestDetail> {
  return apiFetch<TeacherReviewRequestDetail>(`/teacher/review-requests/${id}/request-reanalysis`, null, {
    method: 'POST',
  });
}

export async function fetchTeacherStudentScores(
  studentId: string,
  params: TeacherScoresQuery = {},
): Promise<TeacherStudentScoresResponse> {
  return apiFetch<TeacherStudentScoresResponse>(
    `/teacher/students/${studentId}/scores${buildQuery(params)}`,
    null,
  );
}

export async function fetchTeacherClassScores(
  classId: string,
): Promise<TeacherClassScoresResponse> {
  return apiFetch<TeacherClassScoresResponse>(`/teacher/classes/${classId}/scores`, null);
}

export async function fetchStudentClasses(): Promise<ClassSummary[]> {
  return apiFetch<ClassSummary[]>('/student/classes', null);
}

export async function fetchStudentClassDetail(classId: string): Promise<ClassDetail> {
  return apiFetch<ClassDetail>(`/student/classes/${classId}`, null);
}

export async function fetchTeacherClasses(): Promise<ClassSummary[]> {
  return apiFetch<ClassSummary[]>('/teacher/classes', null);
}

export async function fetchTeacherClassDetail(classId: string): Promise<ClassDetail> {
  return apiFetch<ClassDetail>(`/teacher/classes/${classId}`, null);
}

export async function fetchTeacherClassStudents(classId: string): Promise<ClassStudent[]> {
  return apiFetch<ClassStudent[]>(`/teacher/classes/${classId}/students`, null);
}

export async function fetchTeacherClassTeachers(classId: string): Promise<ClassTeacher[]> {
  return apiFetch<ClassTeacher[]>(`/teacher/classes/${classId}/teachers`, null);
}

export async function fetchAdminUsers(): Promise<AdminUser[]> {
  return expectArray(await apiFetch<unknown>('/admin/users', null), '/admin/users');
}

export async function fetchAdminClasses(): Promise<ClassSummary[]> {
  return expectArray(await apiFetch<unknown>('/admin/classes', null), '/admin/classes');
}

export async function fetchAdminClassDetail(classId: string): Promise<ClassDetail> {
  return apiFetch<ClassDetail>(`/admin/classes/${classId}`, null);
}

export async function fetchAdminDemoReadiness(): Promise<DemoReadinessResponse> {
  return apiFetch<DemoReadinessResponse>('/admin/demo/readiness', null);
}

export async function updateAdminClass(
  classId: string,
  payload: AdminClassUpdatePayload,
): Promise<ClassDetail> {
  return apiFetch<ClassDetail>(`/admin/classes/${classId}`, null, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
}

export async function deleteAdminUser(userId: string): Promise<AdminDeleteUserResponse> {
  return apiFetch<AdminDeleteUserResponse>(`/admin/users/${userId}`, null, {
    method: 'DELETE',
  });
}

function expectArray<T>(payload: unknown, endpoint: string): T[] {
  if (!Array.isArray(payload)) {
    throw new ApiError(`Phản hồi ${endpoint} không phải danh sách dữ liệu.`, 500);
  }
  return payload as T[];
}

export async function fetchRandomVocabularyItem(): Promise<VocabularyItem | null> {
  const data = await apiFetch<{ items: VocabularyItem[] }>('/vocabulary/items?limit=1', null);
  return data.items[0] ?? null;
}

export async function fetchVocabularyItems(
  limit = 20,
  offset = 0,
  topic?: string,
): Promise<{ items: VocabularyItem[]; limit: number; offset: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (topic) params.set('topic', topic);
  return apiFetch(`/vocabulary/items?${params.toString()}`, null);
}

export async function fetchVocabularySets(
  limit = 50,
  offset = 0,
): Promise<{ items: VocabularySet[]; limit: number; offset: number }> {
  return apiFetch(`/vocabulary/sets?limit=${limit}&offset=${offset}`, null);
}

export async function fetchVocabularySetDetail(id: string): Promise<VocabularySetDetail> {
  return apiFetch(`/vocabulary/sets/${id}`, null);
}

export type AssignmentCreatePayload = {
  title: string;
  description?: string | null;
  content_type: 'vocabulary_set' | 'sentence_set';
  content_id: string;
  class_id?: string | null;
  student_id?: string | null;
  due_date?: string | null;
  is_assessment?: boolean;
  deadline?: string | null;
  timer_per_word_seconds?: number;
};

export async function fetchTeacherAssignments(params: {
  class_id?: string | null;
  student_id?: string | null;
  limit?: number;
  offset?: number;
} = {}): Promise<AssignmentsListResponse> {
  const q = new URLSearchParams();
  if (params.class_id) q.set('class_id', params.class_id);
  if (params.student_id) q.set('student_id', params.student_id);
  if (params.limit != null) q.set('limit', String(params.limit));
  if (params.offset != null) q.set('offset', String(params.offset));
  return apiFetch<AssignmentsListResponse>(`/assignments?${q.toString()}`, null);
}

export async function createAssignment(payload: AssignmentCreatePayload): Promise<AssignmentDetail> {
  return apiFetch<AssignmentDetail>('/assignments', null, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function deleteAssignment(id: string): Promise<void> {
  await apiFetch<void>(`/assignments/${id}`, null, { method: 'DELETE' });
}

export async function fetchStudentAssignments(): Promise<StudentAssignment[]> {
  return apiFetch<StudentAssignment[]>('/assignments/student', null);
}

export async function fetchAssignmentWords(assignmentId: string): Promise<AssignmentWords> {
  return apiFetch<AssignmentWords>(`/assignments/${assignmentId}/words`, null);
}

export async function fetchAssignmentStatus(assignmentId: string): Promise<AssignmentStatus> {
  return apiFetch<AssignmentStatus>(`/assignments/${assignmentId}/status`, null);
}

export async function startAssessment(assignmentId: string): Promise<{ submission_id: string; started_at: string }> {
  return apiFetch(`/assignments/${assignmentId}/start`, null, { method: 'POST' });
}

export async function submitAssessment(assignmentId: string): Promise<AssessmentSubmission> {
  return apiFetch<AssessmentSubmission>(`/assignments/${assignmentId}/submit`, null, { method: 'POST' });
}

export async function fetchAssignmentGradebook(assignmentId: string): Promise<AssignmentGradebookItem[]> {
  return apiFetch<AssignmentGradebookItem[]>(`/assignments/${assignmentId}/submissions`, null);
}

export async function overrideAssignmentGrade(assignmentId: string, studentId: string, score: number, overrideReason: string): Promise<AssignmentGradebookItem> {
  return apiFetch<AssignmentGradebookItem>(`/assignments/${assignmentId}/grades/${studentId}`, null, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ score, override_reason: overrideReason }),
  });
}

export async function fetchWordPronunciation(word: string, token: string | null): Promise<{ audio_url: string | null }> {
  return apiFetch<{ audio_url: string | null }>(`/vocabulary/pronunciation/${encodeURIComponent(word)}`, token);
}
