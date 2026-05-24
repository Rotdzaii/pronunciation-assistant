from pathlib import Path


DATASET_ROOT = Path("ai-training/datasets/cmu-arctic/raw")

SPEAKERS = {
    "cmu_us_bdl_arctic": {"speaker_id": "bdl", "gender": "M"},
    "cmu_us_slt_arctic": {"speaker_id": "slt", "gender": "F"},
}


def count_files(folder: Path, pattern: str) -> int:
    if not folder.exists():
        return 0

    return len(list(folder.glob(pattern)))


def preview_txt_done_data(txt_done_path: Path, limit: int = 5):
    if not txt_done_path.exists():
        print(f"Missing transcript file: {txt_done_path}")
        return

    print("Sample transcript lines:")

    lines = txt_done_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines[:limit]:
        print(f"- {line}")


def inspect_speaker(folder_name: str, speaker_info: dict):
    speaker_dir = DATASET_ROOT / folder_name
    wav_dir = speaker_dir / "wav"
    etc_dir = speaker_dir / "etc"
    txt_done_path = etc_dir / "txt.done.data"

    print("=" * 60)
    print(f"Folder: {folder_name}")
    print(f"Speaker ID: {speaker_info['speaker_id']}")
    print(f"Gender: {speaker_info['gender']}")
    print(f"Path: {speaker_dir}")

    if not speaker_dir.exists():
        print("Status: MISSING")
        return

    print("Status: FOUND")
    print(f"WAV files: {count_files(wav_dir, '*.wav')}")
    print(f"txt.done.data exists: {txt_done_path.exists()}")

    sample_wavs = sorted(wav_dir.glob("*.wav"))[:5]

    print()
    print("Sample wav files:")

    for wav_path in sample_wavs:
        print(f"- {wav_path.name}")

    print()
    preview_txt_done_data(txt_done_path)


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    print("CMU ARCTIC dataset root:")
    print(DATASET_ROOT)
    print()

    for folder_name, speaker_info in SPEAKERS.items():
        inspect_speaker(folder_name, speaker_info)


if __name__ == "__main__":
    main()