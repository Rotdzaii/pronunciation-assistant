export type LoginFieldErrors = {
  email?: string;
  password?: string;
};

export type LoginFeedback = {
  title: string;
  message: string;
};

type LoginCredentials = {
  email: string;
  password: string;
};

type SignInResult = {
  data?: {
    session?: { access_token?: string | null } | null;
    user?: unknown | null;
  } | null;
  error?: unknown | null;
};

type SignInWithPassword = (credentials: LoginCredentials) => Promise<SignInResult>;
type SignInResultObserver = (result: SignInResult) => void;

export type LoginAttemptResult =
  | { status: 'validation_error'; fieldErrors: LoginFieldErrors }
  | { status: 'error'; feedback: LoginFeedback }
  | { status: 'success' };

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const INVALID_CREDENTIALS_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Email hoặc mật khẩu không chính xác. Vui lòng kiểm tra và thử lại.',
};

const CONNECTION_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng và thử lại.',
};

const EMAIL_NOT_CONFIRMED_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Tài khoản đã được tạo nhưng email chưa được xác nhận. Vui lòng kiểm tra hộp thư hoặc gửi lại email xác nhận.',
};

const ACCOUNT_DISABLED_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Tài khoản này hiện không thể đăng nhập. Vui lòng liên hệ quản trị viên.',
};

const RATE_LIMIT_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Bạn đã thử quá nhiều lần. Vui lòng chờ một lát rồi thử lại.',
};

const ACCOUNT_NOT_FOUND_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Tài khoản không tồn tại trong dự án hiện tại. Vui lòng kiểm tra email hoặc đăng ký lại.',
};

const UNKNOWN_ERROR_FEEDBACK: LoginFeedback = {
  title: 'Không thể đăng nhập',
  message: 'Đã xảy ra lỗi. Vui lòng thử lại sau.',
};

export function validateLoginCredentials({ email, password }: LoginCredentials): LoginFieldErrors {
  const fieldErrors: LoginFieldErrors = {};
  const normalizedEmail = email.trim();

  if (!normalizedEmail) {
    fieldErrors.email = 'Vui lòng nhập địa chỉ email.';
  } else if (!EMAIL_PATTERN.test(normalizedEmail)) {
    fieldErrors.email = 'Địa chỉ email không hợp lệ.';
  }

  if (!password) {
    fieldErrors.password = 'Vui lòng nhập mật khẩu.';
  }

  return fieldErrors;
}

export function mapLoginError(error: unknown): LoginFeedback {
  const details = getErrorDetails(error);
  const message = details.message.toLowerCase();
  const code = details.code.toLowerCase();
  const name = details.name.toLowerCase();

  if (code === 'invalid_credentials' || message.includes('invalid login credentials')) {
    return INVALID_CREDENTIALS_FEEDBACK;
  }

  // GoTrue normally uses `invalid_credentials` for both an unknown email and
  // a wrong password to prevent account enumeration. Only show this message
  // when a provider explicitly returns the separate code.
  if (code === 'user_not_found') {
    return ACCOUNT_NOT_FOUND_FEEDBACK;
  }

  if (isEmailNotConfirmedError(error)) {
    return EMAIL_NOT_CONFIRMED_FEEDBACK;
  }

  if (/banned|disabled|user_banned/.test(`${code} ${message}`)) {
    return ACCOUNT_DISABLED_FEEDBACK;
  }

  if (/rate.?limit|too many|over.?request/.test(`${code} ${message}`)) {
    return RATE_LIMIT_FEEDBACK;
  }

  if (
    details.status === 0 ||
    name.includes('retryablefetch') ||
    /network|fetch|offline|timeout|econn|failed to fetch|network request failed/.test(message)
  ) {
    return CONNECTION_FEEDBACK;
  }

  return UNKNOWN_ERROR_FEEDBACK;
}

export function isEmailNotConfirmedError(error: unknown): boolean {
  const details = getErrorDetails(error);
  return /email.*not.*confirm|email_not_confirmed/.test(`${details.code.toLowerCase()} ${details.message.toLowerCase()}`);
}

export async function attemptLogin(
  credentials: LoginCredentials,
  signInWithPassword: SignInWithPassword,
  onResult?: SignInResultObserver,
): Promise<LoginAttemptResult> {
  const fieldErrors = validateLoginCredentials(credentials);
  if (Object.keys(fieldErrors).length > 0) {
    return { status: 'validation_error', fieldErrors };
  }

  try {
    const result = await signInWithPassword({
      email: credentials.email.trim().toLowerCase(),
      password: credentials.password,
    });
    onResult?.(result);

    return result.error ? { status: 'error', feedback: mapLoginError(result.error) } : { status: 'success' };
  } catch (error) {
    return { status: 'error', feedback: mapLoginError(error) };
  }
}

export function createLoginRequestGate() {
  let requestInProgress = false;

  return {
    start() {
      if (requestInProgress) {
        return false;
      }

      requestInProgress = true;
      return true;
    },
    finish() {
      requestInProgress = false;
    },
  };
}

function getErrorDetails(error: unknown) {
  if (!error || typeof error !== 'object') {
    return { message: '', code: '', name: '', status: undefined as number | undefined };
  }

  const candidate = error as Record<string, unknown>;
  return {
    message: typeof candidate.message === 'string' ? candidate.message : '',
    code: typeof candidate.code === 'string' ? candidate.code : '',
    name: typeof candidate.name === 'string' ? candidate.name : '',
    status: typeof candidate.status === 'number' ? candidate.status : undefined,
  };
}
