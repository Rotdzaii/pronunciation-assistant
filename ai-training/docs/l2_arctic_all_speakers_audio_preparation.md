# L2-ARCTIC All-Speaker Audio Preparation

## Purpose

The all-speaker phone error metadata has been created at:

`ai-training/datasets/l2-arctic/metadata/all_speakers_phone_error_classification_v2.csv`

The metadata contains rows for all 24 L2-ARCTIC speakers, but the current local raw folder only has the Vietnamese speaker folders extracted. The non-Vietnamese speakers are still stored as ZIP archives, so their expected `audio_path` values do not exist yet.

Before all-speaker training, the speaker archives must be extracted into the expected L2-ARCTIC folder structure.

## Current Issue

The metadata expects audio paths like:

`ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/<SPEAKER>/wav/<UTTERANCE>.wav`

Example:

`ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/ABA/wav/arctic_a0003.wav`

Currently, Vietnamese speakers such as `HQTV`, `PNV`, `THV`, and `TLV` are extracted. Other speakers are available as archives such as `ABA.zip`, `ASI.zip`, and `BWC.zip`.

## Dry-Run Inspection

Run this first. It does not extract files:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\prepare_l2_arctic_all_speakers_audio.py
```

This prints:

- extracted speakers
- archived speakers
- missing speaker folders
- expected audio path pattern
- per-speaker missing audio counts

## Extract Missing Speakers

Only run extraction when you intentionally want to create local raw audio folders. Do not commit the extracted files.

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\prepare_l2_arctic_all_speakers_audio.py --extract
```

The script extracts speaker archives into:

`ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0/`

It skips existing extracted speaker folders by default. Use `--allow-existing` only if you explicitly want ZIP extraction to write into an existing speaker folder.

## Validate Audio Availability

After dry-run or extraction, validate the metadata paths:

```powershell
.\ai-training\.venv\Scripts\python.exe ai-training\scripts\validate_l2_arctic_all_speakers_audio.py
```

The validator recomputes `audio_exists` from disk and saves:

- `ai-training/datasets/l2-arctic/evaluation/all_speakers_audio_availability.csv`
- `ai-training/datasets/l2-arctic/evaluation/all_speakers_missing_audio_by_speaker.csv`
- `ai-training/datasets/l2-arctic/evaluation/all_speakers_audio_validation.json`

## Commit Safety

Do not commit:

- `.wav` files
- raw audio folders
- raw archives
- extracted speaker folders
- checkpoints
- model files
- quarantine folders

Only scripts, documentation, and small metadata/review outputs should be committed.

## Next Step

Once validation reports `All audio ready for training: True`, the next recommended feature is:

`feature/ai-train-all-l2-arctic-cnn-attention`

That feature can train the selected CNN Attention baseline on the all-speaker L2-ARCTIC metadata and evaluate the Vietnamese subset separately.
