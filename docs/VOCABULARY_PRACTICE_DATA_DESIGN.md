# Vocabulary Practice Data Design

## 1. Purpose

This feature is designed to support pronunciation-focused vocabulary practice inside the Pronunciation Assistant project.

It is not intended to become a general vocabulary learning app or a full dictionary product. The main goal is to help students practice speaking target words, receive pronunciation feedback, and allow teachers to organize word-based pronunciation activities more effectively.

## 2. Scope

In scope:

- vocabulary items
- word sets
- teacher assignments
- pronunciation focus tags
- common mistake tags
- practice history linked to AI scoring

Out of scope for now:

- full dictionary app
- complex gamification
- full spaced repetition engine
- marketplace/community content

## 3. Learning From Existing Apps

This feature can borrow selected ideas from existing learning products, but it should adapt them for pronunciation practice rather than general memorization.

- Duolingo: short guided lessons and visible progress
- Quizlet: word sets and simple flashcard-style grouping
- Anki: repeated review for difficult content
- Quizizz: classroom assignments and teacher-managed activities

Adaptation for this project:

- Duolingo-style structure can become short pronunciation drills with a small number of target words.
- Quizlet-style sets can become pronunciation practice sets grouped by topic, phoneme, or common mistakes.
- Anki-style review can later support repeated practice of words a student often mispronounces.
- Quizizz-style class flow can later support teacher assignment, classroom tracking, and lightweight competitive motivation.

The core difference is that the main output is not memorization score but pronunciation practice attempts, AI feedback, and teacher review of speaking difficulties.

## 4. Core Data Entities

### `vocabulary_items`

Purpose:

- stores individual vocabulary words used for pronunciation practice

Suggested fields:

- `id`
- `word`
- `phonetic`
- `language`
- `part_of_speech`
- `meaning_vi`
- `sample_sentence`
- `stress_pattern`
- `difficulty`
- `topic`
- `level`
- `target_phonemes`
- `common_mistake_tags`
- `is_active`
- `created_at`
- `updated_at`

Relationship to existing `practice_history`:

- a vocabulary item can be the source of a pronunciation practice attempt
- `practice_history` can later store `vocabulary_item_id` or equivalent metadata linkage

### `vocabulary_sets`

Purpose:

- stores teacher-created or system-defined sets of vocabulary items for a lesson or practice objective

Suggested fields:

- `id`
- `title`
- `description`
- `created_by`
- `set_type`
- `topic`
- `level`
- `focus_note`
- `is_public`
- `is_active`
- `created_at`
- `updated_at`

Relationship to existing `practice_history`:

- indirect relationship through assigned words and student practice attempts

### `vocabulary_set_items`

Purpose:

- joins vocabulary items to sets and preserves order or emphasis

Suggested fields:

- `id`
- `vocabulary_set_id`
- `vocabulary_item_id`
- `display_order`
- `required`
- `focus_phoneme_override`
- `created_at`

Relationship to existing `practice_history`:

- allows later reporting such as which items in a set produced the most pronunciation errors

### `vocabulary_assignments`

Purpose:

- stores teacher assignment of a vocabulary set to a class, group, or student

Suggested fields:

- `id`
- `vocabulary_set_id`
- `teacher_id`
- `class_id`
- `student_id`
- `title`
- `instruction_text`
- `assigned_at`
- `due_at`
- `status`
- `attempt_limit`
- `created_at`
- `updated_at`

Relationship to existing `practice_history`:

- practice attempts generated from an assignment should be traceable back to that assignment

### `vocabulary_practice_history`

Purpose:

- stores pronunciation-focused practice attempts at the vocabulary feature level

Suggested fields:

- `id`
- `student_id`
- `vocabulary_item_id`
- `vocabulary_set_id`
- `vocabulary_assignment_id`
- `practice_history_id`
- `attempt_no`
- `status`
- `score`
- `problem_phonemes`
- `predicted_error_type`
- `attempted_at`
- `review_flag`
- `created_at`

Relationship to existing `practice_history`:

- `practice_history` remains the main low-level pronunciation attempt record
- `vocabulary_practice_history` works as a feature-specific layer that links vocabulary context, assignment context, and AI output

## 5. Pronunciation-Specific Fields

Suggested pronunciation-oriented fields:

- `phonetic`
- `target_phonemes`
- `common_mistake_tags`
- `stress_pattern`
- `sample_sentence`
- `difficulty`
- `topic`
- `level`

Why they matter:

- `phonetic` helps display target pronunciation clearly
- `target_phonemes` supports phoneme-focused drills
- `common_mistake_tags` supports grouping by known learner errors
- `stress_pattern` is important for word-level pronunciation accuracy
- `sample_sentence` gives minimal speaking context
- `difficulty`, `topic`, and `level` help teachers organize practice content

## 6. Teacher Workflow

Example workflow:

Teacher creates or selects a word set
-> assigns it to students or class groups
-> students practice word pronunciation
-> AI scores attempts and returns pronunciation feedback
-> teacher reviews common mistakes and weak phoneme patterns

Teacher value:

- manage lesson-aligned vocabulary practice
- focus on common pronunciation errors in class
- identify which words or phonemes require additional teaching

## 7. Student Workflow

Example workflow:

Student opens assigned set
-> practices a target word
-> sees pronunciation feedback
-> retries difficult words
-> reviews personal weak phonemes

Student value:

- short and focused pronunciation practice
- repeated practice for hard words
- clearer connection between vocabulary learning and speaking improvement

## 8. AI Integration

AI integration should remain aligned with the current pronunciation pipeline:

- a vocabulary item can create a pronunciation practice job
- `target_word` comes from `vocabulary_items.word`
- the AI result links to `vocabulary_practice_history`
- `problem_phonemes` can later contribute to a student weakness profile

Recommended integration direction:

- keep `practice_history` as the main AI-attempt source of truth
- add vocabulary context around it rather than replacing the existing practice flow

## 9. Minimal MVP Recommendation

Recommended MVP:

- static seed vocabulary list
- word sets
- practice from a word item
- link result to `practice_history`
- teacher view later

Why this MVP is appropriate:

- keeps the feature aligned with pronunciation practice
- avoids overbuilding a dictionary or gamified vocabulary system
- provides enough structure for classroom use and capstone evaluation

## 10. Future Enhancements

- spaced repetition
- leaderboard/class challenge
- adaptive review based on weak phonemes
- teacher-created custom word sets
- sentence-level practice

These enhancements should only be added after the pronunciation-centered data flow is stable.

## 11. Capstone-Friendly Summary

Tính năng luyện từ vựng này được định hướng như một mô-đun hỗ trợ luyện phát âm, không phải ứng dụng học từ vựng tổng quát. Hệ thống tập trung vào danh sách từ, bộ từ do giáo viên giao, các lỗi phát âm thường gặp, luyện tập theo âm vị và liên kết lịch sử luyện tập với kết quả chấm điểm AI. Cách thiết kế này giúp sinh viên luyện phát âm theo từ mục tiêu, giúp giáo viên theo dõi các từ và âm vị mà người học thường phát âm sai, đồng thời phù hợp với định hướng nghiên cứu và phát triển hệ thống chẩn đoán lỗi phát âm tiếng Anh tự động dựa trên Deep Learning.
