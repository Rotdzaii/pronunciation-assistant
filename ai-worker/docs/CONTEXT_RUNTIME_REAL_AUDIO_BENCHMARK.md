# Context Runtime Real Audio Benchmark

## Purpose

This benchmark measures the AI Worker `cnn_attention_context` runtime with real frontend or Supabase Storage audio.

The generated WAV benchmark is useful for checking the scorer path, but it does not represent browser uploads such as WebM or M4A. Real audio can change runtime because URL download, container decoding, resampling, and log-mel preprocessing may dominate more than model inference.

## Difference From Generated WAV Benchmark

Generated WAV benchmark:

- Creates a short local PCM WAV file.
- Does not measure Supabase Storage download.
- Avoids browser audio containers.
- Is best for regression checking the benchmark and scorer path.

Real audio benchmark:

- Uses `--audio-path` for a local frontend-recorded file, or `--audio-url` for a signed Supabase Storage URL.
- Downloads URL audio to a temporary file only.
- Deletes downloaded audio by default.
- Redacts signed URLs in printed output and JSON.
- Records input type as `local_audio_path` or `audio_url`.

## Commands

Benchmark with a local frontend audio file:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py `
  --audio-path C:\path\to\frontend-audio.webm `
  --target-word Architecture `
  --runs 3 `
  --warmup-runs 1
```

Benchmark with a signed Supabase Storage URL:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\benchmark_context_runtime.py `
  --audio-url "<signed-supabase-audio-url>" `
  --target-word Architecture `
  --runs 3 `
  --warmup-runs 1
```

The default real-audio JSON output path is:

```text
ai-worker/docs/context_runtime_real_audio_benchmark_latest.json
```

## Safety

- Do not commit downloaded audio.
- Do not commit signed URLs.
- Signed URLs may expire.
- The benchmark redacts signed URLs in output.
- Downloaded `--audio-url` files are temporary and deleted by default.
- Do not pass `--keep-downloaded-audio` unless you need local debugging, and do not commit the kept file.
- Default benchmark behavior does not POST.

## Result Summary

No real frontend/Supabase audio source was provided with this change, so the real-audio benchmark was not run here and no signed URL was written to docs or JSON.

Generated-audio regression benchmark still verifies that the benchmark script and context scorer path run without POST:

- Environment: torch `2.12.0+cpu`, CUDA unavailable, CPU mode
- Input type: `generated_wav`
- Runs: 3 measured, 1 warmup
- Aggregate total runtime mean: `0.048446` seconds
- Aggregate audio prepare mean: `0.024832` seconds
- Aggregate inference mean: `0.007652` seconds
- Bottleneck recommendation: audio preparation

Run one of the real-audio commands above with a representative frontend recording or signed Supabase URL to create:

```text
ai-worker/docs/context_runtime_real_audio_benchmark_latest.json
```

## Bottleneck Interpretation

- High `audio_download_time_seconds`: inspect network and Supabase Storage latency.
- High `audio_decode_time_seconds`: optimize duration/decode handling, especially for browser containers.
- High `audio_prepare_time_seconds`: optimize audio conversion, decoding, resampling, and log-mel preprocessing.
- High `inference_time_seconds`: evaluate GPU torch, model device placement, and crop/batch efficiency.
- Low `total_runtime_seconds`: no urgent runtime optimization is needed before higher-priority model or integration work.

## Next Recommended Optimization Decision

Run the real-audio benchmark with at least one representative frontend recording before changing the worker. If audio preparation remains dominant, prioritize audio decoding/preprocessing improvements. If inference dominates on real audio, evaluate GPU torch. If total runtime is already low enough for the product flow, defer optimization and focus on scoring quality, calibration, and real alignment.
