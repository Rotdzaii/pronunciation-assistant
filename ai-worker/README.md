# AI Worker

Lightweight worker for the demo pronunciation pipeline. It reads one practice job from Supabase PGMQ, scores it, posts the result to the FastAPI webhook, then archives the queue message after a successful webhook response.

Supported scorer modes:

- `SCORER_MODE=mock`: deterministic mock scoring for fast local demos.
- `SCORER_MODE=wav2vec2`: pretrained Wav2Vec2 speech-recognition baseline for demo/testing.

The Wav2Vec2 mode is not the final pronunciation diagnosis model. It uses pretrained ASR output plus heuristic text matching against the target word or phrase. The team can replace it later with a custom trained pronunciation diagnosis model.

## Setup

```powershell
cd ai-worker
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Wav2Vec2 mode uses `torch`, `transformers`, `imageio-ffmpeg`, `soundfile`, and `librosa`. The first run may download the configured Hugging Face model. Browser/mobile formats such as `webm`, `m4a`, and `mp3` are converted to temporary mono 16 kHz WAV before waveform loading, which avoids the noisy `librosa` audioread fallback path.

`imageio-ffmpeg` provides an FFmpeg binary for local development. If you still need a system FFmpeg for manual debugging on Windows, install it with one of these options, then open a new terminal:

```powershell
winget install Gyan.FFmpeg
# or
choco install ffmpeg
```

Fill `ai-worker/.env`:

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_SECRET=replace-with-ai-webhook-secret
QUEUE_NAME=practice_jobs
SCORER_MODE=mock
MODEL_CONFIDENCE_THRESHOLD=0.65
WAV2VEC2_MODEL_NAME=facebook/wav2vec2-base-960h
WAV2VEC2_BASELINE_MAX_SCORE=92
AUDIO_DOWNLOAD_TIMEOUT_SECONDS=30
AUDIO_TARGET_SAMPLE_RATE=16000
AUDIO_MIN_DURATION_SECONDS=0.5
AUDIO_MAX_DURATION_SECONDS=30
AUDIO_TRIM_SILENCE=true
AUDIO_NORMALIZE=true
AUDIO_DENOISE=false
AUDIO_DENOISE_STRENGTH=light
AUDIO_NOISE_PROFILE_SECONDS=0.3
WORKER_MODE=loop
WORKER_POLL_INTERVAL_SECONDS=1
WORKER_IDLE_BACKOFF_MAX_SECONDS=10
WORKER_BATCH_SIZE=1
WORKER_MAX_JOBS_PER_RUN=0
```

Never commit `.env` or service-role secrets.

## Run

Process one job and exit:

```powershell
set WORKER_MODE=once
python worker.py
```

Run continuously:

```powershell
set WORKER_MODE=loop
python worker.py
```

If the queue is empty, the worker prints:

```text
No job found in practice_jobs queue.
```

In loop mode, the worker polls quickly while jobs are available. When the queue is idle, it sleeps for `WORKER_POLL_INTERVAL_SECONDS` and gradually backs off up to `WORKER_IDLE_BACKOFF_MAX_SECONDS`. After any job is processed, it resets back to the base poll interval. `WORKER_BATCH_SIZE` controls how many consecutive jobs it tries before checking idle state. `WORKER_MAX_JOBS_PER_RUN=0` means unlimited. Press Ctrl+C to stop gracefully.

## Demo Flow

1. The frontend uploads audio.
2. The frontend calls FastAPI `POST /practice/create-job`.
3. FastAPI inserts `public.practice_history` with `status = processing`.
4. FastAPI enqueues a PGMQ message in `practice_jobs` with `job_id`, `student_id`, `target_word`, and `audio_url`.
5. Run `python worker.py`, or use `run_server.bat` from the repo root to start the backend, frontend, and AI worker together.
6. The worker reads one queue message, scores it with the selected scorer, and calls `POST /practice/webhook/ai-result`.
7. If the webhook succeeds, the worker archives the PGMQ message.
8. `public.practice_history` becomes `completed` with `score`, `problem_phonemes`, and `feedback`.

## PGMQ RPCs

The backend already assumes `public.enqueue_practice_job(...)` exists. This worker expects an exposed read RPC such as `read_practice_job` or `pgmq_read`, and an archive RPC such as `archive_practice_job(p_msg_id bigint)` or `pgmq_archive`.

The worker tries project-specific names first:

- `read_practice_job`
- `archive_practice_job`

Then it tries common exposed PGMQ wrapper names:

- `pgmq_read`
- `pgmq_archive`

## Confidence Threshold

`MODEL_CONFIDENCE_THRESHOLD` is a float from `0` to `1`. Both scorer modes populate `feedback.model_confidence` and compare the confidence value with the threshold:

- `feedback.model_confidence.threshold`
- `feedback.model_confidence.level`
- `feedback.model_confidence.is_reliable`

In mock mode, confidence is deterministic from the job id. In Wav2Vec2 mode, confidence is a rough average of model token probabilities from ASR inference. This confidence means the ASR model is confident in its transcript; it is not pronunciation correctness by itself. Low confidence still returns `status = completed` when inference succeeds, but the feedback includes:

```text
Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn.
```

If Wav2Vec2 audio download or model inference fails, the worker posts `status = failed` to the webhook when possible and leaves a failure summary in `feedback.summary`.

## Audio Preprocessing

Wav2Vec2 mode preprocesses downloaded audio before scoring:

1. Convert the uploaded audio file to temporary WAV with `imageio-ffmpeg`.
2. Decode the converted WAV with `soundfile`.
3. Convert multi-channel audio to mono.
4. Resample to `AUDIO_TARGET_SAMPLE_RATE`, default `16000`, if needed.
5. Trim leading/trailing silence when `AUDIO_TRIM_SILENCE=true`.
6. Truncate audio longer than `AUDIO_MAX_DURATION_SECONDS`.
7. Normalize peak volume safely when `AUDIO_NORMALIZE=true`.
8. Validate duration and loudness.

Config:

```dotenv
AUDIO_TARGET_SAMPLE_RATE=16000
AUDIO_MIN_DURATION_SECONDS=0.5
AUDIO_MAX_DURATION_SECONDS=30
AUDIO_TRIM_SILENCE=true
AUDIO_NORMALIZE=true
AUDIO_DENOISE=false
AUDIO_DENOISE_STRENGTH=light
AUDIO_NOISE_PROFILE_SECONDS=0.3
```

If audio is empty, unreadable, or shorter than `AUDIO_MIN_DURATION_SECONDS`, Wav2Vec2 returns `status = failed` with:

```text
Bản ghi quá ngắn hoặc không đủ rõ để AI phân tích.
```

Quiet audio can still be scored, but `feedback.audio.warnings` and `feedback.tips` include a warning. Wav2Vec2 feedback includes audio metadata:

```json
{
  "audio": {
    "duration_seconds": 1.24,
    "sample_rate": 16000,
    "preprocessing": {
      "target_sample_rate": 16000,
      "mono": true,
      "normalized": true,
      "trimmed_silence": true,
      "ffmpeg_converted": true,
      "converted_format": "wav"
    },
    "warnings": []
  }
}
```

Additional preprocessing metadata fields include `original_path`, `target_sample_rate`, `rms_energy`, `peak_amplitude`, `is_too_short`, `is_too_long`, and `is_too_quiet`. Empty, unreadable, too-short, or too-quiet audio returns `status = failed` for Wav2Vec2 with a Vietnamese summary explaining that the recording is too short, too quiet, not clear enough, or cannot be decoded.

If FFmpeg cannot decode the submitted audio, Wav2Vec2 returns `status = failed` with:

```text
Không thể đọc định dạng âm thanh của bản ghi. Vui lòng ghi âm lại và gửi lại.
```

Optional denoising is experimental and disabled by default:

```dotenv
AUDIO_DENOISE=false
AUDIO_DENOISE_STRENGTH=light
AUDIO_NOISE_PROFILE_SECONDS=0.3
```

When `AUDIO_DENOISE=true`, the worker applies only light conservative noise reduction. This is intentionally weak because overly strong denoising can remove final consonants, fricatives such as /s/, and other pronunciation cues needed for scoring. If denoising fails, the worker continues with the original preprocessed audio and adds a warning instead of crashing. If audio appears noisy while denoising is disabled, the worker does not fail automatically; it adds:

```text
Bản ghi có thể có nhiễu nền, kết quả AI có thể kém chính xác.
```

`feedback.audio` includes `denoise_enabled`, `denoise_strength`, and `noise_warning`.

## Wav2Vec2 Baseline

Set:

```dotenv
SCORER_MODE=wav2vec2
WAV2VEC2_MODEL_NAME=facebook/wav2vec2-base-960h
WAV2VEC2_BASELINE_MAX_SCORE=92
AUDIO_DOWNLOAD_TIMEOUT_SECONDS=30
```

The baseline flow is:

1. Download `audio_url`.
2. Preprocess audio to mono 16 kHz, trim silence, normalize safely, and validate duration/loudness.
3. Run pretrained Wav2Vec2 CTC inference.
4. Decode recognized text.
5. Compare recognized text with `target_word` to produce `feedback.text_similarity`, `feedback.target_match`, and `feedback.result_label`.
6. Produce a calibrated rough score from transcript similarity plus ASR confidence, then return heuristic feedback, optional problem item, audio metadata, target-match metadata, and model confidence metadata.

This is useful for pipeline testing and demos only. It is not research-grade phoneme-level pronunciation scoring. A high `feedback.model_confidence.value` only means Wav2Vec2 was confident in the recognized transcript; feedback summaries and the final score are driven primarily by how closely that transcript matches the target word.

The Wav2Vec2 baseline score is capped by `WAV2VEC2_BASELINE_MAX_SCORE`, default `92`, because exact ASR text match is not the same as full pronunciation correctness. In manual reference testing, Azure Pronunciation Assessment returned about `88%` while uncapped Wav2Vec2 text-match scoring could produce `100`; the cap prevents the baseline from overclaiming. To allow 100 for experiments only, explicitly set:

```dotenv
WAV2VEC2_BASELINE_MAX_SCORE=100
```

The score formula is:

```text
raw_score_before_cap = text_match_component + confidence_component
final_score = min(raw_score_before_cap, WAV2VEC2_BASELINE_MAX_SCORE)
```

`feedback.score_breakdown` includes `text_similarity`, `text_match_component`, `confidence_component`, `audio_quality_component` when available, `raw_score_before_cap`, `baseline_max_score`, and `final_score`.

Completed Wav2Vec2 results use these labels and student-facing summaries:

| `feedback.result_label` | Rule | Summary |
| --- | --- | --- |
| `correct` | `text_similarity >= 0.90` | `Bạn đọc khá đúng từ mục tiêu.` |
| `near_correct` | `0.80 <= text_similarity < 0.90` | `Bản ghi gần chính xác với từ mục tiêu, nhưng vẫn còn một vài điểm cần cải thiện.` |
| `partial_match` | `0.50 <= text_similarity < 0.80` | `Bản ghi chỉ khớp một phần với từ mục tiêu. Bạn nên đọc chậm hơn và rõ từng âm tiết hơn.` |
| `mismatch` | `text_similarity < 0.50` | `Wav2Vec2 nhận dạng bản ghi chưa khớp với từ mục tiêu. Có thể bạn đã đọc sai từ hoặc âm thanh chưa đủ rõ.` |

Failed results use `feedback.result_label = "failed"`. The technical `feedback.baseline_note` remains available for secondary/debug UI, but it should not be shown as a prominent student-facing note.

The first Wav2Vec2 run can be slow because the model is downloaded and cached. CPU mode is acceptable for this baseline, but inference will be slower than GPU.

During model loading, the worker suppresses the known Transformers missing-weight warning for `wav2vec2.masked_spec_embed` from the pretrained baseline checkpoint and logs a concise message instead:

```text
Loaded Wav2Vec2 baseline model: facebook/wav2vec2-base-960h
```

Real download, preprocessing, and inference exceptions are still returned as failed scoring results.

## Logs

The worker logs:

- `msg_id`
- `job_id`
- `target_word`
- `scorer_mode`
- Wav2Vec2 model name when applicable
- model confidence value
- downloaded audio byte count and content type when applicable
- webhook status code

It does not log secrets.

## Manual Tests

Mock mode:

```powershell
cd ai-worker
.\.venv\Scripts\activate
# set SCORER_MODE=mock and WORKER_MODE=once or loop in .env
python worker.py
```

Wav2Vec2 mode:

```powershell
cd ai-worker
.\.venv\Scripts\activate
# set SCORER_MODE=wav2vec2 and WORKER_MODE=once or loop in .env
python worker.py
```

On first Wav2Vec2 run, expect a model download before inference starts.

## Wav2Vec2 Baseline Evaluation Checklist

Use this checklist with `SCORER_MODE=wav2vec2`. These checks validate baseline output quality and UI visibility only; they do not prove final pronunciation diagnosis quality.

| Test case | Expected behavior |
| --- | --- |
| Clear correct recording | Worker returns `status = completed`; `recognized_text` should match or be close to `target_word`; confidence is usually medium/high; score is derived from text match plus confidence. |
| Noisy recording | Worker may still return `completed`; confidence should drop when recognition is uncertain; feedback should include the Vietnamese low-confidence warning if below `MODEL_CONFIDENCE_THRESHOLD`. |
| Wrong word recording | Worker returns `completed` if inference succeeds; `recognized_text` should differ from `target_word`; `text_similarity` and score should be lower; feedback should mention target mismatch. |
| Short/empty recording | Worker returns `failed` if audio cannot be decoded or has no samples, otherwise `completed` with low confidence/poor match; UI should render the status without `[object Object]`. |
| Quiet recording | Worker returns `completed` if decoding and inference succeed; confidence may be low; UI should show the confidence warning and baseline note. |
