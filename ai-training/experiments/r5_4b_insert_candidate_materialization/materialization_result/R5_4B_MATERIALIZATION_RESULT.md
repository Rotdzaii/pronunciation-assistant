# R5-4B Full INSERT Candidate Materialization Result

Final status: `R5_4B_INSERT_CANDIDATE_MATERIALIZATION_PASS`

## Environment

- Interpreter: `C:\Users\Admin\Documents\KLTN\pronunciation-assistant\ai-training\.venv\Scripts\python.exe`
- Python: 3.12.10
- PyTorch / CUDA: 2.12.0.dev20260408+cu128 / 12.8
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## Population and materialization

- Frozen word identities: 16582 / 16582
- Candidate rows: 2977040
- Alignable / impossible: 2976844 / 196
- Words affected / no finite INSERT: 65 / 0

## Reproduction

- BEST_INSERT identity: 16582 / 16582
- BEST_INSERT exact score: 16582 / 16582

## Materialization gates

- M1: PASS
- M2: PASS
- M3: PASS
- M4: PASS
- M5: PASS
- M6: PASS
- M7: PASS
- M8: PASS
- M9: PASS
- M10: PASS

Passed: 10 / 10

No Addition truth, recoverability metric, word-level performance metric, threshold search, VALIDATION, or TEST was used.
