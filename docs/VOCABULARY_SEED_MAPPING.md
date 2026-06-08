# Vocabulary Seed Mapping

## Purpose

This document explains how one seed vocabulary item should map into the proposed pronunciation-focused vocabulary flow.

## 1. Seed Item To `vocabulary_items`

A seed JSON record is the draft content source for one `vocabulary_items` row.

Example mapping:

- `word` -> `vocabulary_items.word`
- `phonetic` -> `vocabulary_items.phonetic`
- `meaning_vi` -> `vocabulary_items.meaning_vi`
- `topic` -> `vocabulary_items.topic`
- `level` -> `vocabulary_items.level`
- `difficulty` -> `vocabulary_items.difficulty`
- `sample_sentence` -> `vocabulary_items.sample_sentence`
- `target_phonemes` -> `vocabulary_items.target_phonemes`
- `common_mistake_tags` -> `vocabulary_items.common_mistake_tags`
- `stress_pattern` -> `vocabulary_items.stress_pattern`

## 2. Seed Item To `vocabulary_sets`

Seed items can later be grouped into `vocabulary_sets` by:

- topic
- lesson
- pronunciation challenge
- teacher assignment objective

Examples:

- daily life starter set
- final consonants review set
- `/theta/` and `/eth/` challenge set
- technology speaking set

## 3. Seed Item To Practice Job Creation

When a student starts practice from a seed item:

- the selected word becomes the speaking target
- the application creates a pronunciation practice job
- the job uses the same pronunciation pipeline as other word practice flows

This keeps the vocabulary feature aligned with the existing AI architecture.

## 4. Seed Item To `target_word`

The most direct mapping is:

- `seed_item.word` -> `target_word`

This should remain simple in MVP because the AI worker already uses `target_word` for pronunciation-focused word attempts.

## 5. Seed Item To `practice_history`

Each attempt from a seed word should create or link to a `practice_history` row.

That row remains the primary record for:

- attempt status
- score
- problem phonemes
- predicted error type
- AI feedback

This preserves compatibility with the current pronunciation workflow.

## 6. Seed Item To `vocabulary_practice_history`

The vocabulary feature can later add a feature-specific link table or record such as `vocabulary_practice_history`.

Suggested mapping:

- `vocabulary_item_id`
- `practice_history_id`
- `student_id`
- `vocabulary_set_id`
- `vocabulary_assignment_id`
- `attempt_no`

This layer makes it possible to answer vocabulary-specific questions such as:

- which assigned words a student practiced
- which seed words are most frequently mispronounced
- which phoneme groups are weakest for a student
- which teacher-assigned sets require review

## 7. Summary

The seed file is best treated as structured source content for pronunciation practice. It should map cleanly into `vocabulary_items`, optionally into `vocabulary_sets`, and finally into the existing `practice_history` flow through `target_word` and later `vocabulary_practice_history`.
