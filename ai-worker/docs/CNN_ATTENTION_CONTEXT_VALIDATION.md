# CNN Attention Context Runtime Validation

## Purpose

This document records runtime validation for the AI Worker `cnn_attention_context` scorer. The goal is to confirm that the worker virtual environment can load the context CNN Attention scorer, use a local checkpoint, run context-expanded segment inference, and preserve the safety distinction between classifier confidence and pronunciation scoring.

No model training was performed.

## Dependency Checks

Python executable:

```text
ai-worker/.venv/Scripts/python.exe
```

Torch check:

```powershell
.\ai-worker\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Result:

```text
2.12.0+cpu
False
```

Audio dependency check:

```powershell
.\ai-worker\.venv\Scripts\python.exe -c "import librosa, soundfile, numpy; print('audio deps ok')"
```

Result:

```text
audio deps ok
```

No dependency install was needed.

## Checkpoint Setup

Validated with this local checkpoint:

```text
ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt
```

Checkpoint files are local artifacts and must not be committed.

## Environment Variables

```powershell
$env:SCORER_MODE="cnn_attention_context"
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\models\l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_42_HQTV.pt"
$env:CNN_ATTENTION_CONTEXT_MODE="context_0_10"
$env:CNN_ATTENTION_CONTEXT_LEFT_SECONDS="0.10"
$env:CNN_ATTENTION_CONTEXT_RIGHT_SECONDS="0.10"
```

## Demo Command

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\demo_cnn_attention_context_scorer.py
```

The demo generated a temporary WAV file, ran fallback alignment for the prompt `example`, ran context-expanded CNN Attention inference, printed normalized metadata, and deleted the temporary audio automatically.

## Result Summary

Inference ran successfully.

Top-level result:

| Field | Value |
|---|---|
| `predicted_error_type` | `deletion` |
| `diagnosis_confidence` | `0.45438268780708313` |
| `addition` probability | `0.2529054582118988` |
| `deletion` probability | `0.45438268780708313` |
| `substitution` probability | `0.2927118241786957` |

Top segment:

| Field | Value |
|---|---|
| Phone | `EH` |
| Word | `example` |
| Segment start | `0.0` |
| Segment end | `0.15` |
| Crop start | `0.0` |
| Crop end | `0.25` |
| Context mode | `context_0_10` |
| Context left/right | `0.10 / 0.10` |

The output included:

- `context_mode=context_0_10`
- `context_used=true`
- `context_left_seconds=0.1`
- `context_right_seconds=0.1`
- `segment_start_time`
- `segment_end_time`
- `crop_start_time`
- `crop_end_time`
- `model_output_is_scoring=false`
- `scoring_is_heuristic=true`

## Lightweight Compile Checks

Commands:

```powershell
.\ai-worker\.venv\Scripts\python.exe -m py_compile ai-worker\scripts\demo_cnn_attention_context_scorer.py
.\ai-worker\.venv\Scripts\python.exe -m py_compile ai-worker\app\scorers\cnn_attention_scorer.py
.\ai-worker\.venv\Scripts\python.exe -m py_compile ai-worker\worker.py
```

Result:

All compile checks passed.

## Limitations

- Torch is installed as CPU-only in the worker venv.
- The demo uses generated audio, so the predicted label is only a runtime validation signal, not a model-quality result.
- Fallback alignment is approximate and not real forced alignment.
- Heuristic scoring is not real GOP/CaGOP.
- Classifier confidence is not pronunciation correctness.
- The checkpoint is local only and must not be committed.
