# CNN Attention Scorer

## Purpose

The CNN Attention scorer integrates the selected phone error classifier into the AI Worker inference path. It loads a local checkpoint, runs audio inference, and maps the classifier output into the normalized AI result contract.

This feature does not train a model.

## Selected Model

Model: CNN Attention phone error classifier

Stability metrics:

- mean test macro F1: 0.5124 +/- 0.0214
- mean test addition F1: 0.1938 +/- 0.0415

Label order:

- `addition`
- `deletion`
- `substitution`

## Configuration

Enable the scorer:

```dotenv
SCORER_MODE=cnn_attention
```

Default checkpoint path:

```text
ai-training/models/l2_arctic_error_type_cnn_attention.pt
```

Override with:

```dotenv
CNN_ATTENTION_CHECKPOINT_PATH=C:\path\to\l2_arctic_error_type_cnn_attention.pt
```

Checkpoint files are local artifacts and must not be committed.

## Input And Output Behavior

The scorer accepts a worker job with `audio_url` or `audio_path`. HTTP(S) audio URLs are downloaded to a temporary local file for inference.

Preprocessing mirrors the training setup:

- mono audio
- 16 kHz sample rate
- max length 1.0 second
- pad or truncate to training length
- 64-bin log-mel spectrogram

The classifier outputs:

- `predicted_error_type`
- `class_probabilities`
- `diagnosis_confidence`

The worker maps those fields through `ai-worker/app/contracts/ai_result_contract.py`.

When prompt text is available, the scorer can run aligned inference over fallback alignment segments. See `ai-worker/docs/CNN_ATTENTION_ALIGNED_INFERENCE.md` for the segment-level contract and limitations.

Scorer metadata:

```json
{
  "name": "cnn_attention",
  "type": "phone_error_classifier",
  "version": "cnn_attention_selected_baseline"
}
```

Metadata flags:

```json
{
  "model_output_is_scoring": false,
  "alignment_used": false,
  "gop_used": false,
  "hybrid_used": false
}
```

## Current Limitation

Without prompt text, this scorer performs clip-level demo inference over the first/whole submitted audio segment.

With prompt text, it can perform segment-level inference using fallback alignment. Fallback alignment is approximate scaffolding only and is not real forced alignment, so segment boundaries must not be described as precise.

Classifier confidence is not pronunciation correctness. `diagnosis_confidence` and class probabilities must not be displayed as a pronunciation score.

Any score returned before GOP/CaGOP or hybrid scoring is marked as heuristic/demo metadata.

## Future Extension

Next integration steps:

- forced alignment for phone boundaries and localized `problem_phonemes`
- audited correct samples and a learned correctness head
- a learned quality/scoring head after suitable supervised labels exist
