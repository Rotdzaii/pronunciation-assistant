FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# fastapi-backend runtime source (app/ only; scripts/, db/, docs/ not needed at runtime)
COPY fastapi-backend/app/ ./fastapi-backend/app/

# ai-worker runtime source
COPY ai-worker/worker.py ./ai-worker/worker.py
COPY ai-worker/app/ ./ai-worker/app/
COPY ai-worker/audio/ ./ai-worker/audio/
COPY ai-worker/scorers/ ./ai-worker/scorers/
COPY ai-worker/checkpoints/ ./ai-worker/checkpoints/
