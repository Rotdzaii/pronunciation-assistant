import type { AnalyzeResponse } from '../types';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export function getApiBaseUrl() {
  return process.env.EXPO_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL;
}

export async function uploadAudio(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${getApiBaseUrl()}/upload-audio`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'Upload failed');
  }

  return response.json() as Promise<AnalyzeResponse>;
}
