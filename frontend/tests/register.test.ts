import assert from 'node:assert/strict';
import test from 'node:test';
import {
  attemptSignup,
  createSignupRequestGate,
  getSignupRoute,
  validateSignup,
} from '../lib/register';

const validSignup = {
  email: '  learner@example.com ',
  password: 'StrongPass1',
  confirmPassword: 'StrongPass1',
  role: 'student' as const,
};

test('does not call signup when passwords do not match', async () => {
  let requests = 0;
  const result = await attemptSignup(
    { ...validSignup, confirmPassword: 'OtherPass1' },
    async () => {
      requests += 1;
      return { data: { session: { access_token: 'unused' } } };
    },
  );
  assert.equal(result.status, 'validation_error');
  assert.equal(requests, 0);
  assert.deepEqual(validateSignup({ ...validSignup, confirmPassword: 'OtherPass1' }), {
    confirmPassword: 'Xác nhận mật khẩu không trùng khớp.',
  });
});

test('does not produce a route when signup has no session', async () => {
  const result = await attemptSignup(validSignup, async () => ({ data: { session: null } }));
  assert.equal(result.status, 'confirmation_required');
  assert.equal(getSignupRoute(null), null);
});

test('routes student and teacher only after a backend role is available', () => {
  assert.equal(getSignupRoute('student'), '/(tabs)');
  assert.equal(getSignupRoute('teacher'), '/(tabs)/teacher');
  assert.equal(getSignupRoute('admin'), null);
});

test('normalizes email and prevents overlapping signup requests', async () => {
  let submittedEmail = '';
  const result = await attemptSignup(validSignup, async ({ email }) => {
    submittedEmail = email;
    return { data: { session: { access_token: 'new-token' } } };
  });
  assert.deepEqual(result, { status: 'success', accessToken: 'new-token' });
  assert.equal(submittedEmail, 'learner@example.com');

  const gate = createSignupRequestGate();
  assert.equal(gate.start(), true);
  assert.equal(gate.start(), false);
  gate.finish();
  assert.equal(gate.start(), true);
});
