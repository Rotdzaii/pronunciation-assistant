# AI Current Status and Roadmap

## 1. Purpose

This document clarifies what the current AI can do, what it cannot do yet, and what future AI phases should improve. It is intended for the capstone report and for future development planning after the current context AI flow validation.

## 2. Current AI Position

The current AI is a working research candidate integrated into the application pipeline, not a fully final production-grade pronunciation assessment model.

The selected current candidate is `CNN Attention with context_0_10`.

## 3. What Has Been Completed

- Dataset preparation using L2-ARCTIC
- Vietnamese-only baseline
- All-speaker expansion
- CNN baseline
- CNN Attention
- Speaker-disjoint evaluation
- Addition-focused experiment
- `context_0_10` experiment
- Multi-seed stability check
- AI Worker integration
- Backend webhook validation
- PGMQ validation
- `worker.py` once validation
- Demo validation checklist/result

## 4. Selected Candidate

The selected candidate is:

`CNN Attention with context_0_10`

Vietnamese speaker-disjoint multi-seed stability:

| Metric | Result |
|---|---:|
| Macro F1 | `0.5170 ± 0.0338` |
| Addition F1 | `0.1251 ± 0.0473` |
| Accuracy | `0.6618 ± 0.0324` |

It was selected because it improved speaker-disjoint macro F1 and addition F1 compared with the baseline while keeping the architecture lightweight enough for current system integration.

## 5. What The Current AI Can Do

- Classify broad phone error types: addition, deletion, and substitution
- Run as AI Worker scorer mode `cnn_attention_context`
- Process queued practice jobs
- Return structured AI results to the backend
- Preserve context metadata
- Support the teacher/student feedback pipeline

## 6. What The Current AI Cannot Fully Do Yet

- Cannot claim full Vietnamese-accent pronunciation diagnosis
- Cannot provide real GOP/CaGOP scoring yet
- Cannot rely on fallback alignment as precise phone timing
- Cannot treat classifier confidence as pronunciation correctness
- Cannot fully explain all pronunciation errors
- Cannot guarantee robust performance for all unseen Vietnamese learners

## 7. Why Training Looked Lightweight

Training looked lightweight because the current model is CNN Attention, not a large speech foundation model. The input segments are short, the dataset size is limited, and the project did not train Wav2Vec2 or Whisper from scratch.

The current goal was to establish a baseline research candidate and integrate it into the system pipeline. Heavier AI work remains future work.

## 8. Runtime Status

The current AI Worker runs with CPU torch. CPU inference is fast based on the local benchmark, and the current bottleneck appears to be audio decoding/preprocessing, especially WebM frontend audio, rather than model inference.

Benchmark values:

- `total_runtime_mean ≈ 1.013766s`
- `audio_prepare_time_mean ≈ 0.930055s`
- `inference_time_mean ≈ 0.014268s`

GPU runtime is future work if inference becomes the bottleneck later.

## 9. Missing Pieces For A More Complete Pronunciation Model

1. Real forced alignment
2. Real GOP/CaGOP
3. Better phoneme-level scoring
4. Larger and more diverse datasets
5. Speaker-independent evaluation with more speakers
6. Calibration of score/confidence
7. Stronger acoustic models or fine-tuning
8. Real user evaluation

## 10. Recommended Phase 4 Roadmap

### Phase 4A — Real forced alignment

- MFA or equivalent
- TextGrid parsing
- Phone-level timing reliability

### Phase 4B — Real GOP/CaGOP scoring

- Acoustic likelihood or posterior-based scoring
- Calibration
- Replace `heuristic_gop`

### Phase 4C — Dataset expansion

- Add more public pronunciation datasets if available
- Normalize schema
- Avoid simply mixing incompatible labels

### Phase 4D — Stronger model experiment

- Wav2Vec2/HuBERT/Whisper encoder fine-tuning or feature extraction
- Compare against current CNN Attention context candidate

### Phase 4E — Runtime optimization

- Audio decode/preprocessing improvement
- Local audio vs signed URL benchmark
- GPU only if inference becomes bottleneck

## 11. Capstone-Friendly Summary

Mô hình AI hiện tại của hệ thống là một ứng viên nghiên cứu đã được kiểm chứng và tích hợp end-to-end vào quy trình ứng dụng, từ hàng đợi xử lý, AI Worker, webhook backend đến kết quả hiển thị cho người dùng. Ứng viên hiện tại sử dụng CNN Attention với ngữ cảnh `context_0_10` và cho kết quả ổn định hơn baseline trong đánh giá speaker-disjoint trên nhóm người học tiếng Việt. Tuy nhiên, mô hình này chưa được tuyên bố là mô hình chẩn đoán phát âm hoàn chỉnh ở mức production. Các hướng phát triển tiếp theo cần tập trung vào forced alignment thật, GOP/CaGOP thật, mở rộng dữ liệu, thử nghiệm mô hình âm học mạnh hơn và tối ưu hóa runtime, đặc biệt là bước giải mã và tiền xử lý âm thanh.
