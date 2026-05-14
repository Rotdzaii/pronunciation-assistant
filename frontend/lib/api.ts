import type {
  AudioUploadResponse,
  CreatePracticeJobResponse,
  CurrentUser,
  PracticeHistoryQuery,
  PracticeHistoryResponse,
  PracticeJob,
  TeacherAnalyticsResponse,
} from '../types';
import { Platform } from 'react-native';
import { supabase } from './supabase';

const BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
const SESSION_EXPIRED_MESSAGE = 'Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.';

type CreatePracticeJobPayload = {
  target_word: string;
  audio_url: string;
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
  _accessToken: string | null,
  options: RequestInit = {},
): Promise<T> {
  const { data } = await supabase.auth.getSession();
  const currentAccessToken = data.session?.access_token ?? null;

  if (!currentAccessToken) {
    throw new Error(SESSION_EXPIRED_MESSAGE);
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
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<T>;
}

export async function getMe(accessToken: string | null): Promise<CurrentUser> {
  return apiFetch<CurrentUser>('/auth/me', accessToken);
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

export async function fetchTeacherAnalytics(
  accessToken: string | null,
): Promise<TeacherAnalyticsResponse> {
  return apiFetch<TeacherAnalyticsResponse>('/teacher/analytics', accessToken);
}
