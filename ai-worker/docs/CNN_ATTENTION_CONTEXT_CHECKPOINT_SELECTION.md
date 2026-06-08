# CNN Attention Context Checkpoint Selection

## Purpose

This document explains how to select a local checkpoint compatible with the current `cnn_attention_context` worker loader.

Checkpoint selection matters because `SCORER_MODE=cnn_attention_context` expects the current `SmallPronunciationCNNAttention` architecture and a checkpoint format that matches that loader.

Checkpoints are local artifacts and must not be committed.

## Why Selection Matters

The current worker context scorer expects:

- env var `CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH`
- a checkpoint dictionary loaded with `torch.load(..., map_location="cpu")`
- top-level key `model_state_dict`
- label metadata through `index_to_label`, `label_to_index`, or `error_to_label`
- parameter names compatible with `SmallPronunciationCNNAttention`
- attention keys such as `attention.score.weight` and `attention.score.bias`
- classifier keys such as `classifier.1.weight` and `classifier.1.bias`

If a checkpoint belongs to a different CNN architecture, the worker can fail even when the `.pt` file exists.

## Symptoms Of A Wrong Checkpoint

Common failure patterns include:

- missing keys
- unexpected keys
- size mismatch

Examples:

- missing `attention.score.*`
- missing `classifier.1.*`
- unexpected `classifier.weight` and `classifier.bias`

That usually means the checkpoint belongs to an older/simple CNN architecture instead of the current attention-based context model.

## Inspection Script

Script:

```text
ai-worker/scripts/inspect_context_checkpoints.py
```

Run it with:

```powershell
.\ai-worker\.venv\Scripts\python.exe ai-worker\scripts\inspect_context_checkpoints.py --models-dir ai-training\models
```

Optional flags:

- `--pattern *.pt`
- `--top 50`
- `--json-output path\to\checkpoint-inspection.json`

The script:

- lists local `.pt` files under the selected models directory
- loads each checkpoint on CPU
- reports top-level keys
- detects state-dict-like keys
- counts parameter keys
- checks attention and classifier key patterns
- tries the same `SmallPronunciationCNNAttention` strict load used by the current context scorer
- reports `compatible=true/false`
- prints a suggested PowerShell env command for compatible checkpoints

## How To Set The Checkpoint

When the script reports a compatible checkpoint, set:

```powershell
$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="C:\full\path\to\compatible-checkpoint.pt"
```

Then rerun your worker or validation script.

## Reminders

- Do not commit `.pt` files.
- Do not rename, copy, or edit checkpoint files just to make the worker pass.
- Checkpoints are local artifacts.
