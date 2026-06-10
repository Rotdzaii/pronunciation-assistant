# Phoenix v2 Stable Deep Learning Roadmap

## A. Goal

Phoenix v2 Stable is a Deep Learning-based pronunciation scoring pipeline for deploy/demo. The model is responsible for the main pronunciation score and the main pronunciation error detection result.

The stable deploy goal is not to train a new model in this phase. The goal is to select the strongest available stable Deep Learning scorer, expose its output through a clear contract, and protect the deployment path with validation, timeouts, and safe failure behavior.

Phoenix v2 Stable should be described as a deployable baseline for automatic English pronunciation error diagnosis based on Deep Learning, not as a mainly rule-based or heuristic-based scorer.

## B. Core Principle

Deep Learning-first, production-safe.

Deep Learning is the scoring core. The selected model must be the primary source of the pronunciation score and the primary source of pronunciation error diagnosis.

The production safety layer protects deployment stability. It can validate inputs and outputs, enforce runtime limits, catch exceptions, preserve webhook compatibility, and return safe failed results when inference cannot complete.

The safety layer does not replace model scoring. It must not fabricate a fake model score, convert classifier confidence into pronunciation correctness, or silently replace the Deep Learning scorer with rule-based scoring as the main output.

## C. Scope In

- Audio preprocessing: normalize deploy audio into the format expected by the selected scorer, including sample rate, mono conversion, duration limits, and safe temporary file handling where needed.
- Stable Deep Learning scorer selection: choose the scorer that has the strongest current evidence and can run reliably in the AI Worker.
- CNN Attention Context scorer: use `cnn_attention_context` as the preferred candidate if checkpoint compatibility and worker runtime are stable.
- Model checkpoint selection: identify the local checkpoint family, required metadata, loader compatibility, and configuration variables without committing checkpoint files.
- Model output contract: define stable fields for status, score, problem phonemes, scorer metadata, confidence metadata, alignment metadata, summaries, details, and failure information.
- `problem_phonemes` generation: derive affected phoneme candidates from model diagnosis and available alignment or segment metadata, while preserving source and reliability notes.
- AI Worker integration: keep scorer mode, environment configuration, validation, and final result building explicit.
- Webhook compatibility: preserve backend-compatible fields while retaining rich model output under feedback metadata.
- Timeout handling: define maximum processing time and safe timeout results for deploy/demo stability.
- Exception handling: catch model loading, audio preprocessing, alignment, inference, validation, and webhook payload errors.
- Output validation: reject malformed scorer output before sending it to the backend.
- Runtime validation: verify scorer mode, checkpoint presence, dependency availability, expected output shape, and failed-result behavior in local deploy/demo runs.

## D. Scope Out For Current Stable Phase

- Vietnamese regional accent modeling.
- Dialect-specific diagnosis.
- Age-group pronunciation modeling.
- Large new dataset expansion.
- Rule-based scoring as the main scoring method.
- Heuristic scoring replacing model scoring.
- Human-level pronunciation diagnosis claims.

These topics can be future research directions, but they should not define Phoenix v2 Stable.

## E. Current Candidate Model/Pipeline

Repository inventory for these candidates is recorded in
[PHOENIX_V2_MODEL_INVENTORY.md](PHOENIX_V2_MODEL_INVENTORY.md).

Current possible candidates:

- `wav2vec2` baseline: useful as an ASR baseline and pipeline validation path, but not the final pronunciation correctness scorer. ASR transcript confidence and text similarity do not reliably measure phoneme-level pronunciation quality.
- `cnn_attention` / `cnn_attention_context`: preferred Deep Learning candidate family for Phoenix v2 Stable if the checkpoint and AI Worker runtime are stable. Existing docs identify `cnn_attention_context` with `context_0_10` as the leading integrated research candidate, with the important caveat that classifier confidence is diagnosis confidence, not pronunciation correctness.
- MFA: alignment support, not the scoring core. MFA can provide phone timing to support segment selection and output localization, but timing quality is not pronunciation correctness and must not replace model scoring.

Recommended stable direction:

```text
audio preprocessing
-> optional alignment support
-> Deep Learning scorer inference
-> model-derived score and diagnosis
-> output validation and production safety layer
-> backend-compatible webhook payload
```

## F. Output Contract

Expected Phoenix v2 output:

```json
{
  "status": "completed",
  "score": 82.4,
  "problem_phonemes": ["EH", "K"],
  "feedback": {
    "model_version": "phoenix-v2-cnn-attention-context",
    "scorer_mode": "cnn_attention_context",
    "model_confidence": 0.81,
    "alignment_method": "mfa",
    "summary": "Model detected likely pronunciation issues in selected phonemes.",
    "details": [
      {
        "phoneme": "EH",
        "error_type": "substitution",
        "confidence": 0.81,
        "source": "deep_learning_scorer"
      }
    ],
    "error_type": null
  }
}
```

Required top-level fields:

- `status`
- `score`
- `problem_phonemes`

Required feedback fields:

- `feedback.model_version`
- `feedback.scorer_mode`
- `feedback.model_confidence` if available
- `feedback.alignment_method` if available
- `feedback.summary`
- `feedback.details`
- `feedback.error_type` if scoring fails

Failure behavior:

```json
{
  "status": "failed",
  "score": null,
  "problem_phonemes": [],
  "feedback": {
    "model_version": "phoenix-v2-cnn-attention-context",
    "scorer_mode": "cnn_attention_context",
    "model_confidence": null,
    "alignment_method": null,
    "summary": "Phoenix v2 could not produce a model score for this attempt.",
    "details": [],
    "error_type": "model_inference_failed"
  }
}
```

If the model fails, fallback should return a safe failed status, not a fake model score.

## G. Fastest Implementation Plan

1. `phoenix-v2-deep-learning-roadmap`
   - Define the Deep Learning-first direction and deployment boundaries.
   - Confirm that safety layers are guardrails, not the scoring core.

2. `phoenix-v2-model-inventory`
   - Inventory available scorer modes, checkpoints, checkpoint metadata, training scripts, evaluation reports, and worker runtime requirements.
   - Record which checkpoints are local-only artifacts and must not be committed.

3. `phoenix-v2-model-selection`
   - Select the stable deploy scorer.
   - Prefer `cnn_attention_context` if checkpoint compatibility, dependency availability, and runtime validation pass.
   - Keep `wav2vec2` documented as a baseline, not the final correctness scorer.

4. `phoenix-v2-output-contract`
   - Implement or update the Phoenix v2 output contract.
   - Ensure model version, scorer mode, confidence metadata, alignment metadata, details, and safe failure fields are represented.

5. `phoenix-v2-worker-hardening`
   - Add timeout handling, exception handling, safe failed results, payload sanitization, and webhook compatibility checks.
   - Ensure failures do not emit fake scores.

6. `phoenix-v2-runtime-validation`
   - Validate the selected scorer with local deploy/demo inputs.
   - Confirm output contract validation, failed-result behavior, and backend webhook compatibility.

7. `phoenix-v2-deploy-config`
   - Document stable environment variables for deploy/demo.
   - Confirm scorer mode, checkpoint path, alignment mode, timeout values, and safety validation settings.

## H. Risks and Limitations

- Model confidence is not pronunciation correctness.
- Forced alignment timing is not pronunciation correctness.
- Dataset size is limited.
- Phoenix v2 Stable is a deployable baseline, not the final research-optimized model.
- Vietnamese-specific accent modeling is deferred.
- MFA or fallback alignment can help locate segments, but alignment quality must be reported separately from model scoring.
- The current `cnn_attention_context` direction should not be overclaimed as human-level diagnosis.
- Heuristic or rule-based logic may support safety and formatting, but it must not become the main scoring method.

## I. Recommended Wording For Thesis/Report

Phoenix v2 sử dụng mô hình Deep Learning làm bộ chấm điểm và chẩn đoán lỗi phát âm cốt lõi. Các cơ chế an toàn như kiểm tra đầu vào, kiểm tra đầu ra, xử lý timeout và xử lý ngoại lệ chỉ được dùng để đảm bảo hệ thống triển khai ổn định; chúng không thay thế vai trò chấm điểm của mô hình.
