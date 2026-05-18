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

Wav2Vec2 mode uses `torch`, `transformers`, and `librosa`. The first run may download the configured Hugging Face model. Some audio formats, especially browser `webm`, may require FFmpeg to be installed so `librosa` can decode them.

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
AUDIO_DOWNLOAD_TIMEOUT_SECONDS=30
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

In mock mode, confidence is deterministic from the job id. In Wav2Vec2 mode, confidence is a rough average of model token probabilities from ASR inference. Low confidence still returns `status = completed` when inference succeeds, but the feedback includes:

```text
Độ tin cậy của kết quả chưa cao, bạn nên ghi âm lại trong môi trường yên tĩnh hơn.
```

If Wav2Vec2 audio download or model inference fails, the worker posts `status = failed` to the webhook when possible and leaves a failure summary in `feedback.summary`.

## Wav2Vec2 Baseline

Set:

```dotenv
SCORER_MODE=wav2vec2
WAV2VEC2_MODEL_NAME=facebook/wav2vec2-base-960h
AUDIO_DOWNLOAD_TIMEOUT_SECONDS=30
```

The baseline flow is:

1. Download `audio_url`.
2. Load audio as 16 kHz mono.
3. Run pretrained Wav2Vec2 CTC inference.
4. Decode recognized text.
5. Compare recognized text with `target_word`.
6. Produce a rough score, heuristic feedback, optional problem item, and model confidence metadata.

This is useful for pipeline testing and demos only. It is not research-grade phoneme-level pronunciation scoring.

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
# set SCORER_MODE=mock in .env
python worker.py
```

Wav2Vec2 mode:

```powershell
cd ai-worker
.\.venv\Scripts\activate
# set SCORER_MODE=wav2vec2 in .env
python worker.py
```

On first Wav2Vec2 run, expect a model download before inference starts.
