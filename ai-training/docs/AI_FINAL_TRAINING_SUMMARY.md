# AI Final Training Summary

The current AI training phase is complete at research baseline level.

## Completed Experiments

- CNN baseline
- CNN V2
- Context V3
- Sampler-only CNN
- Binary-stage CNN
- Wav2Vec2 encoder
- Wav2Vec2 context
- Wav2Vec2 attention
- CNN attention
- CNN attention stability check

## Selected Model

Selected model: CNN attention phone error classifier.

Task: phone-level pronunciation error classification for `addition`, `deletion`, and `substitution` on the L2-ARCTIC Vietnamese clean v2 dataset.

Local checkpoint:

`ai-training/models/l2_arctic_error_type_cnn_attention.pt`

The checkpoint is local only and is not committed to Git.

## Key Metrics

CNN attention 3-seed stability metrics:

- Mean test accuracy: 0.6246, std 0.0235
- Mean test macro F1: 0.5124, std 0.0214
- Mean test addition F1: 0.1938, std 0.0415

CNN V2 reference:

- Test macro F1: 0.4835
- Test addition F1: 0.1240

CNN attention replaces CNN V2 as the current main model candidate because it improves both mean test macro F1 and mean test addition F1.

## Final Conclusion

Current AI training phase is complete at research baseline level. The project now has a selected model candidate and documented experiment history for phone-level pronunciation error classification.

Confidence remains classifier confidence, not pronunciation correctness.

## Next Recommended Phase

Move to AI Worker integration and model serving preparation:

- package/manage the selected checkpoint
- define inference input/output contract
- connect predicted error type and probabilities to a feedback format
- keep pronunciation scoring separate from classifier confidence
