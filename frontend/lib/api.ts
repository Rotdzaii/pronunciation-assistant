import type {
  PracticeHistoryItem,
  PracticeJobResponse,
  PracticeStatusResponse,
  TeacherAnalyticsResponse,
} from '../types';

const BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

async function apiFetch<T>(
  path: string,
  token: string | null,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      Accept: 'application/json',
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Request failed');
  }

  return response.json() as Promise<T>;
}

export async function createPracticeJob(
  audioUri: string,
  token: string | null,
): Promise<PracticeJobResponse> {
  const formData = new FormData();
  formData.append('file', {
    uri: audioUri,
    name: 'practice-audio.m4a',
    type: 'audio/m4a',
  } as any);

  return apiFetch<PracticeJobResponse>('/practice', token, {
    method: 'POST',
    body: formData,
  });
}

export async function fetchPracticeStatus(
  jobId: string,
  token: string | null,
): Promise<PracticeStatusResponse> {
  return apiFetch<PracticeStatusResponse>(`/practice/${jobId}`, token);
}

export async function fetchPracticeHistory(
  token: string | null,
): Promise<PracticeHistoryItem[]> {
  return apiFetch<PracticeHistoryItem[]>('/practice/history', token);
}

export async function fetchTeacherAnalytics(
  token: string | null,
): Promise<TeacherAnalyticsResponse> {
  return apiFetch<TeacherAnalyticsResponse>('/teacher/analytics', token);
}
