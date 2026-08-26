import assert from 'node:assert/strict';
import test from 'node:test';
import {
  attemptLogin,
  createLoginRequestGate,
  validateLoginCredentials,
} from '../lib/login';

test('rejects an empty email without calling Supabase', async () => {
  let requests = 0;
  const result = await attemptLogin(
    { email: '', password: 'secret' },
    async () => {
      requests += 1;
      return { error: null };
    },
  );

  assert.deepEqual(result, {
    status: 'validation_error',
    fieldErrors: { email: 'Vui lòng nhập địa chỉ email.' },
  });
  assert.equal(requests, 0);
});

test('rejects an invalid email without calling Supabase', async () => {
  let requests = 0;
  const result = await attemptLogin(
    { email: 'abcxyz', password: 'secret' },
    async () => {
      requests += 1;
      return { error: null };
    },
  );

  assert.deepEqual(result, {
    status: 'validation_error',
    fieldErrors: { email: 'Địa chỉ email không hợp lệ.' },
  });
  assert.equal(requests, 0);
});

test('rejects an empty password without calling Supabase', async () => {
  let requests = 0;
  const result = await attemptLogin(
    { email: 'name@example.com', password: '' },
    async () => {
      requests += 1;
      return { error: null };
    },
  );

  assert.deepEqual(result, {
    status: 'validation_error',
    fieldErrors: { password: 'Vui lòng nhập mật khẩu.' },
  });
  assert.equal(requests, 0);
});

test('maps invalid credentials to a Vietnamese user-safe message', async () => {
  const result = await attemptLogin(
    { email: 'name@example.com', password: 'wrong-password' },
    async () => ({ error: { message: 'Invalid login credentials', code: 'invalid_credentials' } }),
  );

  assert.deepEqual(result, {
    status: 'error',
    feedback: {
      title: 'Không thể đăng nhập',
      message: 'Email hoặc mật khẩu không chính xác. Vui lòng kiểm tra và thử lại.',
    },
  });
});

test('maps an unconfirmed email to an actionable Vietnamese message', async () => {
  const result = await attemptLogin(
    { email: 'name@example.com', password: 'secret' },
    async () => ({ error: { message: 'Email not confirmed', code: 'email_not_confirmed' } }),
  );

  assert.deepEqual(result, {
    status: 'error',
    feedback: {
      title: 'Không thể đăng nhập',
      message: 'Tài khoản đã được tạo nhưng email chưa được xác nhận. Vui lòng kiểm tra hộp thư hoặc gửi lại email xác nhận.',
    },
  });
});

test('maps connection failures to a Vietnamese user-safe message', async () => {
  const result = await attemptLogin(
    { email: 'name@example.com', password: 'secret' },
    async () => {
      throw new TypeError('Failed to fetch');
    },
  );

  assert.deepEqual(result, {
    status: 'error',
    feedback: {
      title: 'Không thể đăng nhập',
      message: 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng và thử lại.',
    },
  });
});

test('maps unknown server errors without exposing the raw response', async () => {
  const result = await attemptLogin(
    { email: 'name@example.com', password: 'secret' },
    async () => ({ error: { message: 'Database connection pool exhausted: trace-123' } }),
  );

  assert.deepEqual(result, {
    status: 'error',
    feedback: {
      title: 'Không thể đăng nhập',
      message: 'Đã xảy ra lỗi. Vui lòng thử lại sau.',
    },
  });
});

test('accepts a successful login and normalizes the email sent to Supabase', async () => {
  let submittedEmail = '';
  const result = await attemptLogin(
    { email: '  name@example.com  ', password: 'secret' },
    async ({ email }) => {
      submittedEmail = email;
      return { error: null };
    },
  );

  assert.deepEqual(result, { status: 'success' });
  assert.equal(submittedEmail, 'name@example.com');
});

test('prevents overlapping login requests', () => {
  const gate = createLoginRequestGate();

  assert.equal(gate.start(), true);
  assert.equal(gate.start(), false);
  gate.finish();
  assert.equal(gate.start(), true);
});

test('reports both missing fields at once', () => {
  assert.deepEqual(validateLoginCredentials({ email: '', password: '' }), {
    email: 'Vui lòng nhập địa chỉ email.',
    password: 'Vui lòng nhập mật khẩu.',
  });
});
