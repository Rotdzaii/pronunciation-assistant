# AI Worker

Lightweight worker for the demo pronunciation pipeline. In demo loop mode, it keeps polling Supabase PGMQ for practice jobs, scores them with the selected scorer, posts results to the FastAPI webhook, then archives queue messages after successful webhook responses.

Supported scorer modes are `mock`, `wav2vec2`, `cnn_attention`, and `cnn_attention_context`. The Wav2Vec2 and CNN scorers import their heavier dependencies only when the corresponding scorer mode is used.

Wav2Vec2 audio is preprocessed through `audio/preprocessing.py`: uploaded WebM, M4A, MP3, and related browser audio formats are converted with FFmpeg to mono 16 kHz PCM WAV, then loaded with `soundfile`. The scorer does not directly decode WebM with `librosa`.

## Setup

```powershell
cd ai-worker
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `ai-worker/.env`:

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
NODE_WEBHOOK_URL=http://localhost:8000/practice/webhook/ai-result
AI_WEBHOOK_SECRET=replace-with-ai-webhook-secret
QUEUE_NAME=practice_jobs
WORKER_MODE=loop
WORKER_POLL_INTERVAL_SECONDS=1
WORKER_IDLE_BACKOFF_MAX_SECONDS=10
SCORER_MODE=mock
MODEL_CONFIDENCE_THRESHOLD=0.65
```

Never commit `.env` or service-role secrets.

## Run

```powershell
python worker.py
```

By default, `WORKER_MODE=loop` keeps the worker alive for the local demo. If the queue is empty, the worker prints:

```text
No job found in practice_jobs queue.
```

For one-shot debugging, run:

```powershell
$env:WORKER_MODE="once"; python worker.py
```

## Demo Flow

1. The frontend uploads audio.
2. The frontend calls FastAPI `POST /practice/create-job`.
3. FastAPI inserts `public.practice_history` with `status = processing`.
4. FastAPI enqueues a PGMQ message in `practice_jobs` with `job_id`, `student_id`, `target_word`, and `audio_url`.
5. Run `python worker.py`.
6. The worker reads one queue message, scores it with the configured `SCORER_MODE`, and calls `POST /practice/webhook/ai-result`.
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

`MODEL_CONFIDENCE_THRESHOLD` is a float from `0` to `1`. Scorers set `feedback.model_confidence.value`, compare it with the threshold, and return:

- `feedback.model_confidence.threshold`
- `feedback.model_confidence.level`
- `feedback.model_confidence.is_reliable`

In mock mode, low confidence still returns `status = completed`, but the feedback includes:

```text
Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn.
```

In `wav2vec2` mode, the scorer also returns `feedback.scorer = "wav2vec2"`, `feedback.target_match`, `feedback.score_breakdown`, and `feedback.audio` metadata when audio decoding succeeds.

The `feedback.audio.preprocessing` metadata includes whether FFmpeg conversion ran, the converted format, target sample rate, and mono/normalization settings.

## Logs

The worker logs:

- `msg_id`
- `job_id`
- `target_word`
- `scorer_mode`
- model confidence value
- webhook status code

It does not log secrets.

## Documentation

- [AI Worker Pipeline Summary](docs/AI_WORKER_PIPELINE_SUMMARY.md)
- [AI Worker Integration Status](docs/AI_WORKER_INTEGRATION_STATUS.md)
- [Final AI Output Contract](docs/FINAL_AI_OUTPUT_CONTRACT.md)
- [Backend Webhook Contract](docs/BACKEND_WEBHOOK_CONTRACT.md)
- [Context CNN Attention Integration Plan](docs/CONTEXT_CNN_ATTENTION_INTEGRATION_PLAN.md)
- [CNN Attention Context Scorer](docs/CNN_ATTENTION_CONTEXT_SCORER.md)
- [Context Worker Loop Validation](docs/CONTEXT_WORKER_LOOP_VALIDATION.md)
- [Context Runtime Benchmark](docs/CONTEXT_RUNTIME_BENCHMARK.md)
- [Context Runtime Real Audio Benchmark](docs/CONTEXT_RUNTIME_REAL_AUDIO_BENCHMARK.md)
- [End-to-End Worker Demo](docs/END_TO_END_WORKER_DEMO.md)
- [Backend Integration Test](docs/BACKEND_INTEGRATION_TEST.md)
