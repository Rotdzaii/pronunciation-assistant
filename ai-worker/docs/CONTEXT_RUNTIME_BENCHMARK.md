# Context Runtime Benchmark

## Purpose

This benchmark measures local runtime for the AI Worker `cnn_attention_context` scorer using the Phase 3 `context_0_10` path.

It times the major worker-side stages:

- setup
- context model loading
- generated or provided audio preparation
- alignment
- segment inference
- AI result validation
- webhook payload build and validation
- optional webhook POST, only when `--post` is passed

## Why Benchmark Before Optimization

The context scorer is functionally validated, but runtime can be dominated by different components on different machines. Measuring first prevents optimizing the wrong layer.

Interpretation guide:

- If `model_load_time_seconds` dominates, prioritize model caching in the worker process.
- If `inference_time_seconds` dominates, evaluate GPU torch, device placement, and crop/batch efficiency.
- If `audio_prepare_time_seconds` dominates, optimize decoding, resampling, and log-mel preprocessing.
- If `post_time_seconds` dominates, inspect backend or network latency.

## Default Benchmark

The default command does not POST. If no audio path is supplied, the script generates a temporary WAV and deletes it after the run.

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py --runs 3 --warmup-runs 1
```

Optional local audio:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py --audio-path path\to\audio.wav --runs 3 --warmup-runs 1
```

Optional checkpoint override:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py --checkpoint-path path\to\context-checkpoint.pt --runs 3 --warmup-runs 1
```

The small JSON report is written by default to:

```text
ai-worker/docs/context_runtime_benchmark_latest.json
```

## Optional POST Benchmark

Do not run POST benchmarks unless intentionally testing backend or network timing. Use an existing backend-compatible job id when needed.

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="<local-ai-webhook-secret>"
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py --job-id <existing-practice-history-job-id> --post --runs 3 --warmup-runs 1
```

The script never prints the webhook secret.

## Result Summary

Latest saved result:

- JSON: `ai-worker/docs/context_runtime_benchmark_latest.json`
- Mode: `SCORER_MODE=cnn_attention_context`
- Context: `context_0_10`
- Default POST behavior: disabled
- Environment: torch `2.12.0+cpu`, CUDA unavailable, CPU mode
- Runs: 3 measured, 1 warmup
- Aggregate total runtime mean: `0.049943` seconds
- Aggregate model load mean: `0.008724` seconds
- Aggregate audio prepare mean: `0.029953` seconds
- Aggregate inference mean: `0.008013` seconds
- Aggregate payload build mean: `0.001093` seconds
- Aggregate payload validation mean: `0.000964` seconds
- Current bottleneck recommendation: audio decoding/log-mel preprocessing

Use the aggregate `mean` values in the JSON to choose the next optimization target. The script also prints `bottleneck_recommendation`.

## Limitations

- Results are local-machine measurements only.
- Generated audio is not equivalent to real frontend audio. Pass `--audio-path` for representative measurements.
- CPU torch may be much slower than GPU torch.
- The checkpoint is local only and must not be committed.
- Fallback alignment is approximate.
- Heuristic score is not real GOP.
- Classifier confidence is diagnosis confidence, not pronunciation correctness.
