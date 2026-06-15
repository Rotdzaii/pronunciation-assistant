# Phoenix v2 Runtime Validation Result

## Validation Summary

Phoenix v2 Stable runtime validation was completed with one successful scoring job and one controlled failure job.

The validation confirms:

- Queue message processing runs the worker once per message.
- The `cnn_attention_context` scorer path executes with MFA alignment.
- Successful jobs send a valid webhook payload and archive only after a 2xx webhook response.
- Missing checkpoint failures do not crash the worker.
- Failed scoring results are archived only after the failed-status webhook succeeds.

Final validation status: **Phoenix v2 Stable Runtime Validation PASSED**.

## Environment Profile

- Runtime profile: Phoenix v2 stable runtime validation
- Scorer mode: `cnn_attention_context`
- Alignment mode: `mfa`
- Model version: `phoenix_v2_stable`
- Webhook behavior under test: payload contract validation and 2xx delivery handling
- Archive behavior under test: message archive after successful webhook delivery
- Manual validation date: 2026-06-15

## Success Case Result

- Message ID: `33`
- Job ID: `7069f29a-aae1-40a0-9296-9f7301a30e8c`
- Target word: `Architecture`
- Scorer mode: `cnn_attention_context`
- Alignment mode: `mfa`
- Model version: `phoenix_v2_stable`
- Model confidence: `0.6485503315925598`
- Result status: `completed`
- Webhook payload valid: `true`
- Webhook status code: `200`
- Archive success: `true`
- Final result: processed job and archived message `33`

This case confirms the queue-to-worker path, one-time worker processing, `cnn_attention_context` scoring, MFA alignment, successful webhook delivery, and archive behavior.

## Failure Case Result

- Message ID: `34`
- Job ID: `9749e4b0-9dae-42f4-8897-ab95d3d2ce6a`
- Target word: `Architecture`
- Scorer mode: `cnn_attention_context`
- Alignment mode: `mfa`
- Model version: `phoenix_v2_stable`
- Failure type: `checkpoint_missing`
- Checkpoint configured: `True`
- Checkpoint exists: `False`
- Model confidence: unavailable
- Result status: `failed`
- Error type: `checkpoint_missing`
- Webhook payload valid: `true`
- Webhook status code: `200`
- Archive success: `true`
- Final result: processed failed job and archived message `34` after webhook success

This case confirms that a missing checkpoint produces a failed scoring result without crashing the worker.

## Webhook Behavior

- Completed scoring jobs emit a valid webhook payload.
- Failed scoring jobs emit a valid failed-status webhook payload.
- Both validation cases received webhook status code `200`.
- Archive processing occurred only after webhook success.

## Archive Behavior

- Success case message `33` was archived after the completed-status webhook succeeded.
- Failure case message `34` was archived after the failed-status webhook succeeded.
- Failed scoring is not treated as a worker crash path when the failure is represented in the output contract and delivered successfully.

## Output Contract Status

- Success result status: `completed`
- Failure result status: `failed`
- Failure error type: `checkpoint_missing`
- Webhook payload validation: passed for both success and failure cases
- `model_confidence` is emitted when available, but it is not pronunciation correctness.

## Warnings and Limitations

- PySoundFile and audioread warnings appeared during the success case.
- The warnings did not block runtime completion, webhook delivery, or archive behavior.
- The success confidence value is model confidence only and must not be interpreted as pronunciation correctness.
- The failure case intentionally validates missing checkpoint handling, not scoring quality.

## Final Conclusion

Phoenix v2 Stable Runtime Validation **PASSED**.
