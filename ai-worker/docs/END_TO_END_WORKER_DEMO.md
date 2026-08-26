# End-to-End Worker Demo

## Purpose

`ai-worker/scripts/demo_worker_end_to_end.py` simulates one AI Worker job without reading Supabase PGMQ. It runs the configured scorer, builds the normalized AI result, validates it, builds the backend webhook payload, validates that payload, and optionally POSTs it to the backend.

The default mode is dry-run. No network request is sent unless `--post` is supplied.

## Prerequisites

Run from the repository root:

```powershell
python ai-worker/scripts/demo_worker_end_to_end.py --dry-run
```

If no `--audio-path` is supplied, the script creates a temporary silent WAV in the OS temp directory and deletes it automatically. Generated audio is not committed.

By default, `SCORER_MODE` falls back to `mock`, matching the worker default.

## CLI Examples

Dry run with temporary audio:

```powershell
python ai-worker/scripts/demo_worker_end_to_end.py --dry-run
```

Dry run with local audio:

```powershell
python ai-worker/scripts/demo_worker_end_to_end.py --audio-path path\to\audio.wav --target-text "example text"
```

Use CNN Attention:

```powershell
$env:SCORER_MODE="cnn_attention"
$env:CNN_ATTENTION_CHECKPOINT_PATH="C:\path\to\l2_arctic_error_type_cnn_attention.pt"
python ai-worker/scripts/demo_worker_end_to_end.py --target-text "example text"
```

CNN Attention requires local inference dependencies such as `torch`, plus the local checkpoint. If dependencies or checkpoint are missing, the demo builds a failed AI result and failed webhook payload instead of crashing.

## Optional Backend POST

To send the demo payload:

```powershell
$env:NODE_WEBHOOK_URL="http://localhost:8000/practice/webhook/ai-result"
$env:AI_WEBHOOK_SECRET="replace-with-ai-webhook-secret"
python ai-worker/scripts/demo_worker_end_to_end.py --post
```

The script also accepts `AI_WEBHOOK_URL` as an alternative webhook URL env var. If URL or secret is missing, it prints a clear skip reason.

## Output To Inspect

The demo prints:

- simulated job payload
- normalized AI result JSON
- AI result validation status
- backend webhook payload JSON
- webhook payload validation status
- optional POST status

## Safety Notes

Classifier confidence is not pronunciation score.

Fallback alignment is approximate and not real forced alignment.

`heuristic_gop` is internal diagnostic scaffolding, not a public score.

GOP/CaGOP is not the Phoenix v2 roadmap. Public score remains unavailable
until a learned quality scorer is trained and validated.

MFA execution is scaffolded only and requires local installation/configuration.

For real CNN Attention inference, local `torch` dependencies and a local checkpoint are required.
