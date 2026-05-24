from pathlib import Path


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")

VIETNAMESE_SPEAKERS = ["HQTV", "PNV", "THV", "TLV"]


def count_files(folder: Path, pattern: str) -> int:
    if not folder.exists():
        return 0

    return len(list(folder.glob(pattern)))


def inspect_speaker(speaker_id: str):
    speaker_dir = DATASET_ROOT / speaker_id

    wav_dir = speaker_dir / "wav"
    transcript_dir = speaker_dir / "transcript"
    annotation_dir = speaker_dir / "annotation"
    textgrid_dir = speaker_dir / "textgrid"

    print("=" * 60)
    print(f"Speaker: {speaker_id}")
    print(f"Path: {speaker_dir}")

    if not speaker_dir.exists():
        print("Status: MISSING")
        return

    print("Status: FOUND")
    print(f"WAV files:        {count_files(wav_dir, '*.wav')}")
    print(f"Transcript files: {count_files(transcript_dir, '*.txt')}")
    print(f"Annotation files: {count_files(annotation_dir, '*.TextGrid')}")
    print(f"TextGrid files:   {count_files(textgrid_dir, '*.TextGrid')}")

    sample_wavs = sorted(wav_dir.glob("*.wav"))[:5]

    print()
    print("Sample wav files:")

    for wav_path in sample_wavs:
        print(f"- {wav_path.name}")


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    print("L2-ARCTIC dataset root:")
    print(DATASET_ROOT)
    print()

    for speaker_id in VIETNAMESE_SPEAKERS:
        inspect_speaker(speaker_id)


if __name__ == "__main__":
    main()