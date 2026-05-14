export type AppRole = 'student' | 'teacher' | string;

export type CurrentUser = {
  id: string;
  email: string | null;
  app_role: AppRole | null;
};

export type FastApiPracticeJobStatus = 'processing' | 'completed' | 'failed';
export type PracticeJobStatus = FastApiPracticeJobStatus | 'queued';

export type AudioUploadResponse = {
  message: string;
  storage_path: string;
  audio_url: string;
  mime_type: string;
  size: number;
};

export type CreatePracticeJobResponse = {
  job_id: string;
  status: FastApiPracticeJobStatus;
  message: string;
};

export type ProblemPhoneme = string | {
  phoneme?: unknown;
  type?: unknown;
  severity?: unknown;
  tip?: unknown;
  word?: unknown;
  message?: unknown;
  description?: unknown;
};

export type PracticeFeedback = {
  summary?: unknown;
  tips?: unknown;
  message?: unknown;
  text?: unknown;
  [key: string]: unknown;
};

export type PracticeJob = {
  id: string;
  student_id: string;
  target_word: string;
  audio_url: string;
  status: FastApiPracticeJobStatus;
  score: number | null;
  problem_phonemes: ProblemPhoneme[] | null;
  feedback: PracticeFeedback | null;
  created_at: string | null;
  updated_at: string | null;
};

export type PracticeHistoryResponse = {
  items: PracticeJob[];
  limit: number;
  offset: number;
};

export type PracticeHistoryQuery = {
  limit?: number;
  offset?: number;
  status?: FastApiPracticeJobStatus;
};

export type PracticeHistoryItem = PracticeJob;
export type PracticeStatusResponse = PracticeJob;

export type TeacherAnalyticsResponse = {
  total_students: number;
  avg_score: number;
  active_jobs: number;
};
