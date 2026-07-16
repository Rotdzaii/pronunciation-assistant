import assert from 'node:assert/strict';
import test from 'node:test';
import { resendSignupConfirmation } from '../lib/emailConfirmation';

test('resends confirmation with a normalized email and callback URL', async () => {
  let submittedEmail = '';
  let submittedRedirect = '';
  const result = await resendSignupConfirmation(
    ' Learner@Example.com ',
    'http://localhost:8081/callback',
    async (email, redirectTo) => {
      submittedEmail = email;
      submittedRedirect = redirectTo;
      return { error: null };
    },
  );

  assert.deepEqual(result, {
    tone: 'success',
    message: 'Đã gửi lại email xác nhận. Vui lòng kiểm tra hộp thư của bạn.',
  });
  assert.equal(submittedEmail, 'learner@example.com');
  assert.equal(submittedRedirect, 'http://localhost:8081/callback');
});

test('maps resend rate limits without exposing provider details', async () => {
  const result = await resendSignupConfirmation(
    'learner@example.com',
    'http://localhost:8081/callback',
    async () => ({ error: { code: 'over_request_rate_limit', message: 'Too many requests' } }),
  );

  assert.deepEqual(result, {
    tone: 'error',
    message: 'Bạn đã yêu cầu quá nhiều lần. Vui lòng chờ một lát rồi thử lại.',
  });
});
