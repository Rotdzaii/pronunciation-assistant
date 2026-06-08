# Vocabulary Seed List

## Purpose Of Seed Vocabulary

The seed vocabulary list provides a safe, curated starting set of words for pronunciation-focused practice inside the Pronunciation Assistant project.

Its purpose is to:

- support early feature validation without requiring a full dictionary
- give teachers and students a structured set of practice words
- cover common pronunciation challenges for Vietnamese learners
- connect directly to the existing pronunciation practice pipeline

This seed list is not intended to become a full lexical resource or a general vocabulary product.

## Selection Criteria

The seed vocabulary should be selected using the following criteria:

- simple and familiar English words
- suitable for school and classroom practice
- useful for short speaking drills
- broad enough to cover multiple topics
- strong pronunciation value, not only meaning memorization value
- representative of common learner pronunciation difficulties

The list should avoid:

- rare specialist vocabulary
- copyrighted third-party word lists copied directly
- overly long or difficult words in the first seed version

## Topics

Suggested seed topics:

- daily life
- school
- technology
- work
- travel
- pronunciation challenges

These topics keep the vocabulary relevant to practical communication while still supporting pronunciation-focused drills.

## Levels

Suggested seed levels:

- beginner
- elementary
- lower_intermediate

The first seed list should emphasize beginner and elementary words so students can focus on pronunciation instead of lexical difficulty.

## Pronunciation Focus

The seed list should intentionally cover these pronunciation focus groups:

- final consonants
- vowel length
- word stress
- consonant clusters
- `/theta/` and `/eth/`
- `/sh/` and `/ch/`
- `/i_short/` and `/i_long/`
- `/ae/` and `/e/`
- `/r/` and `/l/`

These focus groups are chosen because they are common and pedagogically useful for English pronunciation practice with Vietnamese learners.

## Common Mistake Tags

Suggested common mistake tags:

- `drop_final_consonant`
- `final_stop_confusion`
- `vowel_length_confusion`
- `word_stress_error`
- `consonant_cluster_reduction`
- `th_sound_substitution`
- `sh_ch_confusion`
- `short_long_i_confusion`
- `ae_e_confusion`
- `r_l_confusion`

These tags support:

- teacher review
- phoneme-focused grouping
- later adaptive review
- future analytics on frequent learner errors

## How These Words Connect To `practice_history`

Each seed word should be usable as a pronunciation practice target.

Suggested connection flow:

- a learner opens a seeded vocabulary item
- the system creates a pronunciation practice job
- `target_word` is populated from the seed item word
- the speaking attempt is stored in `practice_history`
- the vocabulary context can later be linked through `vocabulary_practice_history`

This means the seed list should be designed as structured practice content, not as static display-only data.

## Capstone-Friendly Notes

For capstone scope, the first seed list should remain:

- small enough to review manually
- simple enough to seed without external dependencies
- rich enough to demonstrate pronunciation-oriented data modeling

A good first version is around 30 to 50 words across several topics and pronunciation challenge groups.
