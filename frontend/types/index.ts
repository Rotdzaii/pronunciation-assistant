export type AppRole = 'student' | 'teacher' | 'admin' | string;

export type CurrentUser = {
  id: string;
  email: string | null;
  app_role: AppRole | null;
};

export type UserProfile = {
  id: string;
  email: string | null;
  display_name: string;
  app_role: AppRole | null;
  avatar_url: string | null;
  avatar_path: string | null;
  updated_at?: string | null;
};

export type UpdateProfilePayload = {
  display_name: string;
};

export type ProfileAvatarUploadResponse = {
  profile: UserProfile;
  avatar_url: string;
  avatar_path: string;
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

export type TeacherCommonError = {
  label: string;
  count: number;
};

export type TeacherStudentSummary = {
  id: string;
  email: string | null;
  display_name: string;
  practice_count: number;
  average_score: number;
  latest_practice_at: string | null;
  status: 'need_support' | 'stable' | 'good_progress' | string;
  common_errors: TeacherCommonError[];
};

export type TeacherProgressDistribution = {
  need_support: number;
  improving: number;
  good_progress: number;
};

export type TeacherAnalyticsResponse = {
  total_students: number;
  total_practice_sessions?: number;
  average_score?: number;
  avg_score: number;
  active_jobs: number;
  need_support_count?: number;
  good_progress_count?: number;
  improving_count?: number;
  common_errors?: TeacherCommonError[];
  progress_distribution?: TeacherProgressDistribution;
  students?: TeacherStudentSummary[];
};

export type ReviewSource = 'student_report' | 'system_flag' | string;
export type ReviewReason = 'ai_scored_wrong' | 'audio_issue' | 'result_mismatch' | 'low_confidence' | 'other' | string;
export type ReviewStatus = 'pending' | 'in_review' | 'resolved' | 'rejected' | 'reanalyzing' | string;
export type ReviewSeverity = 'low' | 'medium' | 'high' | string;

export type ReviewAuditLog = {
  id: string;
  actor_id: string;
  actor_role: string;
  action_type: string;
  before_value: Record<string, unknown> | null;
  after_value: Record<string, unknown> | null;
  reason: string | null;
  created_at: string | null;
};

export type TeacherReviewRequest = {
  id: string;
  practice_history_id: string;
  student_id: string;
  student_name: string;
  student_email: string | null;
  class_id: string | null;
  class_name: string | null;
  class_code: string | null;
  target_word: string | null;
  audio_url: string | null;
  ai_score: number | null;
  practice_status: string | null;
  submitted_at: string | null;
  source: ReviewSource;
  reason: ReviewReason;
  severity: ReviewSeverity;
  status: ReviewStatus;
  student_note: string | null;
  teacher_note: string | null;
  teacher_resolution: string | null;
  created_at: string | null;
  updated_at: string | null;
  resolved_at: string | null;
};

export type TeacherReviewRequestsResponse = {
  items: TeacherReviewRequest[];
  total: number;
  pending_count: number;
  student_report_count: number;
  system_flag_count: number;
  resolved_this_week_count: number;
  limit: number;
  offset: number;
};

export type TeacherReviewRequestDetail = TeacherReviewRequest & {
  practice_history: Record<string, unknown> | null;
  ai_feedback: Record<string, unknown> | null;
  problem_phonemes: unknown[] | null;
  audit_logs: ReviewAuditLog[];
};

export type TeacherStudentScoreSummary = {
  total_attempts: number;
  completed_attempts: number;
  average_score: number | null;
  last_practice_at: string | null;
  common_mistakes: TeacherCommonError[];
};

export type TeacherPracticeResult = {
  practice_history_id: string;
  target_word: string | null;
  audio_url: string | null;
  score: number | null;
  text_similarity: number | null;
  model_confidence: number | null;
  status: string;
  mistake_categories: string[];
  feedback: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
  needs_review: boolean;
};

export type TeacherStudentScoresResponse = {
  student: TeacherStudentSummary;
  class_id: string | null;
  summary: TeacherStudentScoreSummary;
  items: TeacherPracticeResult[];
  limit: number;
  offset: number;
};

export type TeacherClassScoreStudent = {
  student_id: string;
  student_name: string;
  student_email: string | null;
  total_attempts: number;
  completed_attempts: number;
  average_score: number | null;
  last_practice_at: string | null;
  needs_review_count: number;
};

export type TeacherClassScoresResponse = {
  class_id: string;
  class_name: string;
  class_code: string | null;
  completed_count: number;
  pending_count: number;
  failed_count: number;
  common_mistakes: TeacherCommonError[];
  students: TeacherClassScoreStudent[];
};

export type ReportPracticeResultPayload = {
  reason: 'ai_scored_wrong' | 'audio_issue' | 'result_mismatch' | 'other';
  student_note?: string | null;
};

export type StudentReviewRequest = {
  id: string;
  practice_history_id: string;
  reason: string;
  student_note: string | null;
  status: string;
  teacher_note: string | null;
  teacher_resolution: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type StudentReviewRequestDetail = StudentReviewRequest & {
  source: string;
  severity: string;
  audit_logs: Array<Record<string, unknown>>;
};

export type ClassTeacher = {
  id: string;
  email: string | null;
  display_name: string;
  teacher_role: string | null;
  status: string | null;
};

export type ClassStudent = {
  id: string;
  email: string | null;
  display_name: string;
  joined_at: string | null;
  status: string | null;
};

export type ClassSummary = {
  id: string;
  code: string | null;
  name: string;
  description: string | null;
  status?: string | null;
  teacher_role?: string | null;
  student_count: number;
  teacher_count: number;
  teachers?: ClassTeacher[];
};

export type ClassDetail = ClassSummary & {
  students?: ClassStudent[];
  teachers: ClassTeacher[];
};

export type AdminUser = {
  id: string;
  email: string | null;
  app_role: string | null;
  display_name: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminClassUpdatePayload = {
  code?: string;
  name?: string;
  description?: string | null;
  status?: string;
};

export type AdminDeleteUserResponse = {
  deleted: boolean;
  user_id: string;
  email: string | null;
  auth_user_deleted: boolean;
};

export type AdminMenuKey = 'overview' | 'classes' | 'delete_users' | 'exports';

export type DemoReadinessCheck = {
  key: string;
  passed: boolean;
  expected: unknown;
  actual: unknown;
  message: string;
};

export type DemoReadinessResponse = {
  passed: boolean;
  checks: DemoReadinessCheck[];
};
