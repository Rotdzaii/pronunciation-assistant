import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FORGOT_PASSWORD_ROUTE,
  isPasswordRecoveryEvent,
  requestPasswordRecovery,
  RESET_PASSWORD_ROUTE,
  updateRecoveredPassword,
  validateRecoveryEmail,
} from '../lib/passwordRecovery';

test('login uses the forgot-password route', () => {
  assert.equal(FORGOT_PASSWORD_ROUTE, '/(auth)/forgot-password');
});

test('rejects empty and malformed recovery emails without making a request', () => {
  assert.deepEqual(validateRecoveryEmail(''), { email: 'Vui lòng nhập địa chỉ email.' });
  assert.deepEqual(validateRecoveryEmail('invalid-email'), { email: 'Địa chỉ email không hợp lệ.' });
});

test('sends one normalized password-recovery request and keeps success neutral', async () => {
  let calls = 0;
  let submittedEmail = '';
  let submittedRedirect = '';
  const result = await requestPasswordRecovery(' Learner@Example.com ', 'http://localhost:8081/callback?flow=recovery', async (email, redirectTo) => {
    calls += 1;
    submittedEmail = email;
    submittedRedirect = redirectTo;
    return { error: null };
  });

  assert.equal(calls, 1);
  assert.equal(submittedEmail, 'learner@example.com');
  assert.equal(submittedRedirect, 'http://localhost:8081/callback?flow=recovery');
  assert.deepEqual(result, {
    tone: 'success',
    message: 'Nếu email này tồn tại trong hệ thống, chúng tôi đã gửi liên kết đặt lại mật khẩu. Vui lòng kiểm tra hộp thư và thư rác.',
  });
});

test('maps recovery rate-limit and network failures to user-safe messages', async () => {
  const rateLimit = await requestPasswordRecovery('learner@example.com', 'http://localhost/callback', async () => ({ error: { code: 'over_request_rate_limit' } }));
  const network = await requestPasswordRecovery('learner@example.com', 'http://localhost/callback', async () => { throw new TypeError('Failed to fetch'); });

  assert.equal(rateLimit.tone, 'error');
  assert.match(rateLimit.message, /quá nhiều/i);
  assert.equal(network.tone, 'error');
  assert.match(network.message, /kết nối/i);
});

test('PASSWORD_RECOVERY is the only callback event that opens reset-password', () => {
  assert.equal(isPasswordRecoveryEvent('PASSWORD_RECOVERY'), true);
  assert.equal(isPasswordRecoveryEvent('SIGNED_IN'), false);
  assert.equal(RESET_PASSWORD_ROUTE, '/(auth)/reset-password');
});

test('password mismatch never calls updateUser', async () => {
  let calls = 0;
  const result = await updateRecoveredPassword('Password1', 'Password2', async () => {
    calls += 1;
    return { error: null };
  });

  assert.equal(calls, 0);
  assert.equal(result.success, false);
  assert.deepEqual(result.fieldErrors, { confirmPassword: 'Xác nhận mật khẩu không trùng khớp.' });
});

test('a valid recovery password calls updateUser with the entered password', async () => {
  let submittedPassword = '';
  const result = await updateRecoveredPassword('Password1', 'Password1', async (password) => {
    submittedPassword = password;
    return { error: null };
  });

  assert.equal(submittedPassword, 'Password1');
  assert.equal(result.success, true);
});
