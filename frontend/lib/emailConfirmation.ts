export type ConfirmationFeedback = {
  tone: 'success' | 'error';
  message: string;
};

type ResendResult = { error?: unknown | null };
type ResendSignupConfirmation = (email: string, redirectTo: string) => Promise<ResendResult>;

export async function resendSignupConfirmation(
  email: string,
  redirectTo: string,
  resend: ResendSignupConfirmation,
): Promise<ConfirmationFeedback> {
  try {
    const { error } = await resend(email.trim().toLowerCase(), redirectTo);
    if (error) return mapConfirmationError(error);
    return {
      tone: 'success',
      message: 'Đã gửi lại email xác nhận. Vui lòng kiểm tra hộp thư của bạn.',
    };
  } catch (error) {
    return mapConfirmationError(error);
  }
}

export function mapConfirmationError(error: unknown): ConfirmationFeedback {
  const details = getErrorDetails(error);
  const message = details.message.toLowerCase();
  const code = details.code.toLowerCase();

  if (/rate.?limit|too many|over.?request/.test(`${code} ${message}`)) {
    return {
      tone: 'error',
      message: 'Bạn đã yêu cầu quá nhiều lần. Vui lòng chờ một lát rồi thử lại.',
    };
  }
  if (
    details.status === 0 ||
    /network|fetch|offline|timeout|econn|failed to fetch|network request failed/.test(message)
  ) {
    return {
      tone: 'error',
      message: 'Không thể kết nối để gửi email xác nhận. Vui lòng kiểm tra mạng và thử lại.',
    };
  }
  return {
    tone: 'error',
    message: 'Chưa thể gửi lại email xác nhận. Vui lòng thử lại sau.',
  };
}

function getErrorDetails(error: unknown) {
  if (!error || typeof error !== 'object') {
    return { message: '', code: '', status: undefined as number | undefined };
  }
  const candidate = error as Record<string, unknown>;
  return {
    message: typeof candidate.message === 'string' ? candidate.message : '',
    code: typeof candidate.code === 'string' ? candidate.code : '',
    status: typeof candidate.status === 'number' ? candidate.status : undefined,
  };
}
