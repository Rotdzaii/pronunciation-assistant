# Phase 4A — Real Forced Alignment Plan

## 1. Purpose

Phase 4A addresses the current alignment limitation before adding more advanced pronunciation scoring. The existing fallback alignment is approximate: it splits audio duration across prompt words or canonical phones without detecting speech boundaries from the audio signal. Because of that, phone-level location reliability is limited.

Reliable GOP/CaGOP scoring requires dependable phone boundaries. Without real forced alignment, a GOP-like score may evaluate the wrong time window for a target phone, making phone-level diagnosis and feedback unreliable.

## 2. Current Alignment State

The AI Worker currently has an alignment scaffold, but fallback alignment remains the safe default.

- `fallback_aligner.py` creates approximate word or phone segments by evenly splitting audio duration.
- `alignment_contract.py` defines a normalized result shape with `alignment_status`, `alignment_method`, `segments`, `words`, `phones`, `note`, and `metadata`.
- `alignment_service.py` selects `ALIGNMENT_MODE=fallback`, `ALIGNMENT_MODE=mfa`, or `ALIGNMENT_MODE=none`, and can fall back when MFA fails.
- `mfa_aligner.py` is a local MFA wrapper scaffold. It expects MFA, a dictionary, and an acoustic model to already exist locally.
- `textgrid_parser.py` parses common MFA/Praat TextGrid word and phone tiers into the existing alignment contract.
- Metadata and notes currently mark fallback alignment as approximate and not real forced alignment.

## 3. Target Alignment Capability

The target capability is:

`audio + transcript/canonical phones -> word-level and phone-level timings`

Expected output fields:

- `alignment_status`
- `alignment_method`
- `is_forced_alignment`
- word segments
- phone segments
- start and end timestamps
- confidence or quality flags if the aligner provides them
- fallback reason if real forced alignment fails

The desired result is not a pronunciation score by itself. It is a reliable timing layer that later scoring and diagnosis modules can consume.

## 4. Candidate Tool: MFA

Montreal Forced Aligner (MFA) is a strong candidate because it is commonly used for forced alignment, supports acoustic models and pronunciation dictionaries, aligns known transcripts to audio, and outputs TextGrid files that can be parsed into word and phone intervals.

Expected local requirements:

- MFA installed locally
- acoustic model
- pronunciation dictionary
- converted WAV, preferably 16 kHz mono if required by the local MFA setup
- transcript text
- temporary working directory

This phase should not install MFA automatically or commit MFA models, dictionaries, generated TextGrid files, audio files, or local paths containing sensitive information.

## 5. Input Requirements

The worker needs these inputs for real forced alignment:

- audio file path after download or local preparation
- `target_word` or `target_sentence` / `prompt_text`
- normalized transcript text suitable for the aligner
- canonical pronunciation or dictionary lookup for target words and phones
- fallback behavior when transcript text is missing

Transcript normalization should remove unsupported characters, normalize whitespace, and preserve the words needed by the pronunciation dictionary. If only a word prompt is available, MFA may still run, but alignment quality may be weaker than with a full sentence or prompt context.

## 6. Output Contract

Real alignment should preserve the existing `alignment_contract.py` schema whenever possible:

- `status`
- `alignment_status`
- `method`
- `alignment_method`
- `segments`
- `words`
- `phones`
- `note`
- `metadata`

Recommended metadata additions or conventions:

- `alignment_method=mfa`
- `alignment_status=completed` for successful normalized results
- `alignment_status=failed` when MFA fails and fallback is disabled
- `alignment_status=fallback` or `completed` with explicit fallback metadata when fallback is used, depending on backend compatibility
- `is_forced_alignment=true` for MFA results
- `is_fallback=false` for MFA results
- `textgrid_path` local only, sanitized before webhook payload creation
- `alignment_warning` when fallback or degraded alignment is used
- `fallback_reason` when MFA fails and fallback is allowed

Local file paths such as `textgrid_path`, MFA corpus paths, temporary directories, dictionary paths, and acoustic model paths must not be sent to the backend payload.

## 7. Worker Integration Strategy

Implementation should keep the current safe default:

```dotenv
ALIGNMENT_MODE=fallback
ALLOW_ALIGNMENT_FALLBACK=true
```

Add real forced alignment behind:

```dotenv
ALIGNMENT_MODE=mfa
```

Planned behavior:

- Keep `ALIGNMENT_MODE=fallback` as the default for local development and demos.
- Use `ALIGNMENT_MODE=mfa` only when MFA, dictionary, acoustic model, audio preparation, and transcript normalization are configured.
- If MFA fails, use fallback only when `ALLOW_ALIGNMENT_FALLBACK=true`.
- If fallback is used, preserve explicit fallback metadata and warnings.
- Preserve the existing output contract so backend and frontend integrations do not need a breaking schema change.
- Strip or sanitize local file paths before sending webhook payloads to the backend.

## 8. Validation Plan

1. TextGrid parser unit/demo validation
2. MFA local validation with one safe audio file and transcript
3. Compare MFA timings with fallback timings
4. Run context scorer with MFA alignment
5. Validate backend payload metadata
6. Validate PGMQ / `worker.py` once with `ALIGNMENT_MODE=mfa`

Validation should record whether the result used true MFA alignment or fell back. It should also verify that no local paths, signed URLs, raw audio, checkpoints, or secrets are committed or sent in app-facing payloads.

## 9. Risks and Limitations

- MFA installation and environment setup can be complex.
- Pronunciation dictionary mismatch can cause missing words or poor phone alignment.
- A word-only prompt may not provide enough transcript context for robust alignment.
- Learner mispronunciation can reduce forced-alignment quality.
- Alignment is not the same as pronunciation correctness.
- Forced alignment still needs a scoring model, GOP, CaGOP, or hybrid scoring layer after this phase.

## 10. Recommended Implementation Branches

- `feature/ai-phase4a-mfa-local-validation`
- `feature/ai-worker-mfa-alignment-mode`
- `feature/ai-worker-mfa-context-scorer-validation`

## 11. Capstone-Friendly Summary

Phase 4A giải quyết hạn chế hiện tại của hệ thống là alignment fallback chỉ mang tính xấp xỉ, chưa phải forced alignment thật. Việc lập kế hoạch tích hợp forced alignment bằng MFA hoặc công cụ tương đương là bước cần thiết để xác định ranh giới từ và âm vị đáng tin cậy hơn, từ đó tạo nền tảng cho chấm điểm phoneme-level và GOP/CaGOP trong các giai đoạn tiếp theo.
