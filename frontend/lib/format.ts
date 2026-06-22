export function formatUserRole(role: string | null | undefined): string {
  if (role === 'admin') return 'Quản trị viên';
  if (role === 'teacher') return 'Giảng viên';
  if (role === 'student') return 'Học sinh';
  return 'Chưa xác định';
}

export function formatMembershipStatus(status: string | null | undefined): string {
  if (!status || status === 'active') return 'Đang hoạt động';
  if (status === 'pending') return 'Đang chờ';
  if (status === 'inactive') return 'Không hoạt động';
  return status;
}

export function formatTeacherRole(role: string | null | undefined): string {
  if (role === 'owner') return 'Chủ nhiệm';
  if (role === 'co_teacher') return 'Đồng giảng dạy';
  if (role === 'substitute') return 'Dạy thay';
  return 'Giảng viên';
}

export function formatReadinessStatus(passed: boolean | null | undefined): string {
  return passed ? 'Đạt' : 'Chưa đạt';
}

export function formatReviewSource(source: string | null | undefined): string {
  if (source === 'student_report') return 'Học sinh báo cáo';
  if (source === 'system_flag') return 'Hệ thống đánh dấu';
  return source || 'Không rõ';
}

export function formatReviewReason(reason: string | null | undefined): string {
  if (reason === 'ai_scored_wrong') return 'AI chấm sai';
  if (reason === 'audio_issue') return 'Âm thanh bị lỗi';
  if (reason === 'result_mismatch') return 'Kết quả không khớp';
  if (reason === 'low_confidence') return 'Độ tin cậy thấp';
  if (reason === 'other') return 'Khác';
  return reason || 'Không rõ';
}

export function formatReviewStatus(status: string | null | undefined): string {
  if (status === 'pending') return 'Chờ xử lý';
  if (status === 'in_review') return 'Đang xem xét';
  if (status === 'resolved') return 'Đã xử lý';
  if (status === 'rejected') return 'Từ chối';
  if (status === 'reanalyzing') return 'Đã yêu cầu chấm lại';
  return status || 'Không rõ';
}

export function formatReviewSeverity(severity: string | null | undefined): string {
  if (severity === 'low') return 'Thấp';
  if (severity === 'medium') return 'Trung bình';
  if (severity === 'high') return 'Cao';
  return severity || 'Không rõ';
}

export function formatPracticeStatus(status: string | null | undefined): string {
  if (status === 'pending' || status === 'queued') return 'Chờ xử lý';
  if (status === 'processing') return 'Đang chẩn đoán';
  if (status === 'completed') return 'Đã có kết quả';
  if (status === 'failed') return 'Xử lý lỗi';
  return status || 'Không rõ';
}
