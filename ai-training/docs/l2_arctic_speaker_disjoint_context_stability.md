# L2-ARCTIC Vietnamese Speaker-Disjoint Context Stability

## Purpose

This feature prepares a multi-seed stability check for Vietnamese speaker-disjoint CNN Attention with `context_0_10`.

The previous single-seed `context_0_10` run improved both macro F1 and addition F1 under Vietnamese leave-one-speaker-out evaluation. Before replacing the current model candidate, the result needs a seed-stability check because only four Vietnamese speakers are available and addition support is small.

Classifier confidence remains model confidence only. It is not a pronunciation correctness score.

## Previous Results

| Run | Mean macro F1 | Mean addition F1 | Mean deletion F1 | Mean substitution F1 |
|---|---:|---:|---:|---:|
| Baseline speaker-disjoint | 0.5022 | 0.0881 | 0.6881 | 0.7303 |
| Addition-focused sampler | 0.4715 | 0.0958 | 0.6347 | 0.6839 |
| Context `context_0_10` | 0.5178 +/- 0.0252 | 0.1246 +/- 0.0271 | 0.6801 +/- 0.0371 | 0.7486 +/- 0.0283 |

The sampler-only variant was not selected because it improved addition only slightly while reducing macro F1 and the other class scores. `context_0_10` is currently the leading speaker-disjoint robustness candidate, but it should be confirmed across seeds.

## Planned Stability Protocol

Script:

`ai-training/scripts/run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py`

Context mode:

`context_0_10`

Planned seeds:

- `42`
- `123`
- `2026`

Vietnamese leave-one-speaker-out folds:

- `HQTV`
- `PNV`
- `THV`
- `TLV`

Full run size:

`3 seeds x 4 folds = 12 training jobs`

For each seed/fold:

- Test set: all rows from the held-out Vietnamese speaker.
- Training set: original `train` rows from all other speakers, including non-Vietnamese speakers.
- Validation set: original `val` rows from all other speakers.
- The held-out Vietnamese speaker must not appear in training or validation.

## Commands

Lightweight dry run:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py --dry-run --max-seeds 1 --max-folds 1
```

Full stability check:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py --run-full
```

Optional shorter training check:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\run_l2_arctic_vietnamese_speaker_disjoint_context_stability.py --run-full --max-seeds 1 --max-folds 1 --epochs 1
```

Comparison:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\compare_vietnamese_speaker_disjoint_context_stability.py
```

The comparison script does not crash if full stability outputs are missing. It writes the available comparison rows and prints the missing full-run instruction.

## Expected Outputs

Full stability output files:

- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_stability_results.json`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_stability_per_seed_fold.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_stability_summary.csv`
- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_stability_per_class.csv`

Lightweight comparison output:

- `ai-training/datasets/l2-arctic/evaluation/vietnamese_speaker_disjoint_context_stability_comparison.csv`

## Checkpoint Note

Full training writes local checkpoints to:

`ai-training/models/l2_arctic_cnn_attention_speaker_disjoint_context_stability_seed_<SEED>_<SPEAKER>.pt`

These checkpoints are local artifacts and must not be committed.

## Interpretation

The stability summary should be interpreted at three levels:

- Overall seed-fold mean and standard deviation.
- Per-seed averages, to detect seed sensitivity.
- Per-held-out-speaker averages, to detect speaker sensitivity.

The main target is improved addition F1 without sacrificing macro F1. Deletion and substitution must also remain close to the speaker-disjoint baseline because a model that only improves addition by degrading the common classes is not a better classifier.

## Decision Rule

`context_0_10` can replace the current candidate only if the multi-seed stability result consistently improves macro F1 and addition F1 over the speaker-disjoint baseline without large degradation in deletion or substitution F1.

If the multi-seed result is unstable, `context_0_10` should remain an experiment and the next work should investigate richer segment features, phone-context features, or speaker-disjoint data augmentation rather than selecting the context model.

## Limitations

- Only four Vietnamese speakers are available.
- Addition remains sparse and high variance.
- The all-speaker training pool includes non-Vietnamese L1 groups.
- The classifier operates on known phone-error segments, not full end-to-end pronunciation assessment.
- Confidence is classifier confidence, not pronunciation correctness.
