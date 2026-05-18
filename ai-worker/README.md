# AI Worker

Lightweight worker for the demo pronunciation pipeline. It reads one practice job from Supabase PGMQ, produces deterministic mock scoring, posts the result to the FastAPI webhook, then archives the queue message after a successful webhook response.

The current scorer is mock-only. `SCORER_MODE=wav2vec2` is reserved for a later Wav2Vec2 baseline and this worker intentionally does not install or import `torch` or `transformers`.

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
SCORER_MODE=mock
MODEL_CONFIDENCE_THRESHOLD=0.65
```

Never commit `.env` or service-role secrets.

## Run

```powershell
python worker.py
```

If the queue is empty, the worker prints:

```text
No job found in practice_jobs queue.
```

## Demo Flow

1. The frontend uploads audio.
2. The frontend calls FastAPI `POST /practice/create-job`.
3. FastAPI inserts `public.practice_history` with `status = processing`.
4. FastAPI enqueues a PGMQ message in `practice_jobs` with `job_id`, `student_id`, `target_word`, and `audio_url`.
5. Run `python worker.py`.
6. The worker reads one queue message, scores it with the mock scorer, and calls `POST /practice/webhook/ai-result`.
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

`MODEL_CONFIDENCE_THRESHOLD` is a float from `0` to `1`. The mock scorer produces a deterministic `feedback.model_confidence.value` from the job id, compares it with the threshold, and sets:

- `feedback.model_confidence.threshold`
- `feedback.model_confidence.level`
- `feedback.model_confidence.is_reliable`

In mock mode, low confidence still returns `status = completed`, but the feedback includes:

```text
Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn.
```

When Wav2Vec2 support is added later, the same threshold will gate real model confidence from inference.

## Logs

The worker logs:

- `msg_id`
- `job_id`
- `target_word`
- `scorer_mode`
- model confidence value
- webhook status code

It does not log secrets.
