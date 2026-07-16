export type SignupRole = 'student' | 'teacher';

export type SignupFieldErrors = {
  email?: string;
  password?: string;
  confirmPassword?: string;
};

export type SignupFeedback = {
  title: string;
  message: string;
  tone: 'error' | 'success';
};

export type PasswordRequirements = {
  minLength: boolean;
  uppercase: boolean;
  lowercase: boolean;
  number: boolean;
};

type SignupValues = {
  email: string;
  password: string;
  confirmPassword: string;
  role: SignupRole;
};

type SignupResponse = {
  data?: { session?: { access_token?: string | null } | null } | null;
  error?: unknown | null;
};

type SignUp = (values: { email: string; password: string; role: SignupRole }) => Promise<SignupResponse>;

export type SignupAttemptResult =
  | { status: 'validation_error'; fieldErrors: SignupFieldErrors }
  | { status: 'error'; feedback: SignupFeedback }
  | { status: 'confirmation_required'; feedback: SignupFeedback }
  | { status: 'success'; accessToken: string };

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function getPasswordRequirements(password: string): PasswordRequirements {
  return {
    minLength: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /\d/.test(password),
  };
}

export function isPasswordStrong(password: string): boolean {
  return Object.values(getPasswordRequirements(password)).every(Boolean);
}

export function validateSignup(values: SignupValues): SignupFieldErrors {
  const errors: SignupFieldErrors = {};
  const email = values.email.trim().toLowerCase();

  if (!email) {
    errors.email = 'Vui lòng nhập địa chỉ email.';
  } else if (!EMAIL_PATTERN.test(email)) {
    errors.email = 'Địa chỉ email không hợp lệ.';
  }

  if (!values.password) {
    errors.password = 'Vui lòng tạo mật khẩu.';
  } else if (!isPasswordStrong(values.password)) {
    errors.password = 'Mật khẩu chưa đáp ứng đầy đủ yêu cầu bảo mật.';
  }

  if (!values.confirmPassword) {
    errors.confirmPassword = 'Vui lòng nhập lại mật khẩu.';
  } else if (values.confirmPassword !== values.password) {
    errors.confirmPassword = 'Xác nhận mật khẩu không trùng khớp.';
  }

  return errors;
}

export function mapSignupError(error: unknown): SignupFeedback {
  const details = getErrorDetails(error);
  const message = details.message.toLowerCase();
  const code = details.code.toLowerCase();

  if (/already registered|already exists|email.*exist|user.*exist/.test(`${code} ${message}`)) {
    return { title: 'Không thể đăng ký', message: 'Email này đã được đăng ký. Vui lòng đăng nhập hoặc dùng email khác.', tone: 'error' };
  }
  if (/weak password|password.*(short|least|strong)|password.*8/.test(`${code} ${message}`)) {
    return { title: 'Không thể đăng ký', message: 'Mật khẩu chưa đủ mạnh. Vui lòng kiểm tra lại các yêu cầu.', tone: 'error' };
  }
  if (/too many|rate limit|over.*request/.test(`${code} ${message}`)) {
    return { title: 'Không thể đăng ký', message: 'Bạn đã thử quá nhiều lần. Vui lòng chờ một lát rồi thử lại.', tone: 'error' };
  }
  if (
    details.status === 0 ||
    /network|fetch|offline|timeout|econn|failed to fetch|network request failed/.test(message)
  ) {
    return { title: 'Không thể đăng ký', message: 'Không có kết nối mạng. Vui lòng kiểm tra kết nối và thử lại.', tone: 'error' };
  }
  return { title: 'Không thể đăng ký', message: 'Máy chủ đang gặp sự cố. Vui lòng thử lại sau.', tone: 'error' };
}

export async function attemptSignup(values: SignupValues, signUp: SignUp): Promise<SignupAttemptResult> {
  const fieldErrors = validateSignup(values);
  if (Object.keys(fieldErrors).length > 0) {
    return { status: 'validation_error', fieldErrors };
  }

  try {
    const result = await signUp({
      email: values.email.trim().toLowerCase(),
      password: values.password,
      role: values.role,
    });
    if (result.error) return { status: 'error', feedback: mapSignupError(result.error) };

    const accessToken = result.data?.session?.access_token;
    if (!accessToken) {
      return {
        status: 'confirmation_required',
        feedback: {
          title: 'Kiểm tra email của bạn',
          message: 'Chúng tôi đã gửi email xác nhận. Hãy xác nhận email rồi đăng nhập để tiếp tục.',
          tone: 'success',
        },
      };
    }
    return { status: 'success', accessToken };
  } catch (error) {
    return { status: 'error', feedback: mapSignupError(error) };
  }
}

export function getSignupRoute(
  role: string | null | undefined,
): '/(tabs)' | '/(tabs)/teacher' | null {
  if (role === 'student') return '/(tabs)';
  if (role === 'teacher') return '/(tabs)/teacher';
  return null;
}

export function createSignupRequestGate() {
  let requestInProgress = false;
  return {
    start() {
      if (requestInProgress) return false;
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
    return { message: '', code: '', status: undefined as number | undefined };
  }
  const candidate = error as Record<string, unknown>;
  return {
    message: typeof candidate.message === 'string' ? candidate.message : '',
    code: typeof candidate.code === 'string' ? candidate.code : '',
    status: typeof candidate.status === 'number' ? candidate.status : undefined,
  };
}
