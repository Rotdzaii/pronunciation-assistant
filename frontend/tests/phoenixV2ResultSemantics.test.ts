import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

const resultScreen = readFileSync(
  resolve(process.cwd(), 'app/(tabs)/practice/result.tsx'),
  'utf8',
);

test('Phoenix v2 result UI uses suspected classifier wording, not confirmed-error wording', () => {
  assert.match(resultScreen, /Loại lỗi mô hình nghi ngờ/);
  assert.match(resultScreen, /Loại lỗi dự đoán:/);
  assert.match(resultScreen, /Độ tin cậy phân loại lỗi:/);
  assert.doesNotMatch(resultScreen, /3 lỗi phát âm/);
  assert.doesNotMatch(resultScreen, /Phát âm chính xác/);
});

test('Phoenix v2 result UI defaults old payloads to classifier-only capability', () => {
  assert.match(resultScreen, /\?\? 'error_type_classifier_only'/);
  assert.match(
    resultScreen,
    /Mô hình hiện chỉ phân loại loại lỗi khi có lỗi và chưa xác định được bạn phát âm đúng hay sai\./,
  );
});

test('invalid reliability does not render the diagnosis branch', () => {
  assert.match(resultScreen, /\{isInvalid \? \(/);
  assert.match(resultScreen, /hasSuspectedDiagnosis = !isInvalid/);
});
