export type PracticeJobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export type PracticeResult = {
  score: number;
  problem_phonemes: string[];
};

export type PracticeJobResponse = {
  job_id: string;
};

export type PracticeStatusResponse = {
  status: PracticeJobStatus;
  result?: PracticeResult;
  error?: string;
};

export type PracticeHistoryItem = {
  id: string;
  created_at: string;
  score: number;
  problem_phonemes: string[];
};

export type TeacherAnalyticsResponse = {
  total_students: number;
  avg_score: number;
  active_jobs: number;
};
