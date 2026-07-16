import type { Session } from '@supabase/supabase-js';
import { isPasswordStrong } from './register';

export type PasswordRecoveryFieldErrors = {
  email?: string;
  password?: string;
  confirmPassword?: string;
};

export type PasswordRecoveryFeedback = {
  tone: 'success' | 'error';
  message: string;
};

type ResetEmailResult = { error?: unknown | null };
type ResetEmailRequest = (email: string, redirectTo: string) => Promise<ResetEmailResult>;
type UpdatePasswordResult = { error?: unknown | null };
type UpdatePasswordRequest = (password: string) => Promise<UpdatePasswordResult>;

let recoverySessionUserId: string | null = null;

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const FORGOT_PASSWORD_ROUTE = '/(auth)/forgot-password';
export const RESET_PASSWORD_ROUTE = '/(auth)/reset-password';

export function normalizeRecoveryEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function validateRecoveryEmail(email: string): PasswordRecoveryFieldErrors {
  const normalizedEmail = normalizeRecoveryEmail(email);
  if (!normalizedEmail) return { email: 'Vui lòng nhập địa chỉ email.' };
  if (!EMAIL_PATTERN.test(normalizedEmail)) return { email: 'Địa chỉ email không hợp lệ.' };
  return {};
}

export function validatePasswordReset(password: string, confirmPassword: string): PasswordRecoveryFieldErrors {
  const errors: PasswordRecoveryFieldErrors = {};
  if (!password) {
    errors.password = 'Vui lòng tạo mật khẩu mới.';
  } else if (!isPasswordStrong(password)) {
    errors.password = 'Mật khẩu chưa đáp ứng đầy đủ yêu cầu bảo mật.';
  }

  if (!confirmPassword) {
    errors.confirmPassword = 'Vui lòng nhập lại mật khẩu mới.';
  } else if (confirmPassword !== password) {
    errors.confirmPassword = 'Xác nhận mật khẩu không trùng khớp.';
  }
  return errors;
}

export async function requestPasswordRecovery(
  email: string,
  redirectTo: string,
  request: ResetEmailRequest,
): Promise<PasswordRecoveryFeedback> {
  try {
    const { error } = await request(normalizeRecoveryEmail(email), redirectTo);
    if (error) return mapPasswordRecoveryError(error);
    return {
      tone: 'success',
      message: 'Nếu email này tồn tại trong hệ thống, chúng tôi đã gửi liên kết đặt lại mật khẩu. Vui lòng kiểm tra hộp thư và thư rác.',
    };
  } catch (error) {
    return mapPasswordRecoveryError(error);
  }
}

export async function updateRecoveredPassword(
  password: string,
  confirmPassword: string,
  update: UpdatePasswordRequest,
): Promise<{ fieldErrors?: PasswordRecoveryFieldErrors; feedback?: PasswordRecoveryFeedback; success: boolean }> {
  const fieldErrors = validatePasswordReset(password, confirmPassword);
  if (Object.keys(fieldErrors).length > 0) return { fieldErrors, success: false };

  try {
    const { error } = await update(password);
    if (error) return { feedback: mapUpdatePasswordError(error), success: false };
    return { success: true };
  } catch (error) {
    return { feedback: mapUpdatePasswordError(error), success: false };
  }
}

export function markPasswordRecoverySession(session: Session | null): void {
  recoverySessionUserId = session?.user.id ?? null;
}

export function hasPasswordRecoverySession(session: Session | null): boolean {
  return Boolean(session?.user.id && recoverySessionUserId === session.user.id);
}

export function clearPasswordRecoverySession(): void {
  recoverySessionUserId = null;
}

export function isPasswordRecoveryEvent(event: string): boolean {
  return event === 'PASSWORD_RECOVERY';
}

export function isRecoveryFlow(flow: string | string[] | undefined): boolean {
  return flow === 'recovery';
}

function mapPasswordRecoveryError(error: unknown): PasswordRecoveryFeedback {
  const details = getErrorDetails(error);
  const combined = `${details.code} ${details.message}`.toLowerCase();
  if (/rate.?limit|too many|over.?request/.test(combined)) {
    return { tone: 'error', message: 'Bạn đã yêu cầu quá nhiều lần. Vui lòng chờ một lát rồi thử lại.' };
  }
  if (/redirect|redirect_to|url.*allow|not allowed/.test(combined)) {
    return { tone: 'error', message: 'Liên kết đặt lại mật khẩu chưa được cấu hình đúng. Vui lòng thử lại sau.' };
  }
  if (details.status === 0 || /network|fetch|offline|timeout|econn|failed to fetch|network request failed/.test(combined)) {
    return { tone: 'error', message: 'Không thể kết nối để gửi email. Vui lòng kiểm tra mạng và thử lại.' };
  }
  return { tone: 'error', message: 'Chưa thể gửi liên kết đặt lại mật khẩu. Vui lòng thử lại sau.' };
}

function mapUpdatePasswordError(error: unknown): PasswordRecoveryFeedback {
  const details = getErrorDetails(error);
  const combined = `${details.code} ${details.message}`.toLowerCase();
  if (/expired|invalid.*(token|link)|session.*(expired|missing)|same password/.test(combined)) {
    return { tone: 'error', message: 'Liên kết đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.' };
  }
  if (details.status === 0 || /network|fetch|offline|timeout|econn|failed to fetch|network request failed/.test(combined)) {
    return { tone: 'error', message: 'Không thể kết nối đến máy chủ. Vui lòng thử lại.' };
  }
  return { tone: 'error', message: 'Chưa thể cập nhật mật khẩu. Vui lòng thử lại sau.' };
}

function getErrorDetails(error: unknown) {
  if (!error || typeof error !== 'object') return { message: '', code: '', status: undefined as number | undefined };
  const candidate = error as Record<string, unknown>;
  return {
    message: typeof candidate.message === 'string' ? candidate.message : '',
    code: typeof candidate.code === 'string' ? candidate.code : '',
    status: typeof candidate.status === 'number' ? candidate.status : undefined,
  };
}
