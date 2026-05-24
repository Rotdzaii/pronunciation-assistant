from pathlib import Path
import csv


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
OUTPUT_CSV = Path("ai-training/datasets/l2-arctic/metadata/vietnamese_speakers_metadata.csv")

VIETNAMESE_SPEAKERS = {
    "HQTV": {"l1": "Vietnamese", "gender": "M"},
    "PNV": {"l1": "Vietnamese", "gender": "F"},
    "THV": {"l1": "Vietnamese", "gender": "F"},
    "TLV": {"l1": "Vietnamese", "gender": "M"},
}


def read_transcript(transcript_path: Path) -> str:
    if not transcript_path.exists():
        return ""

    text = transcript_path.read_text(encoding="utf-8", errors="ignore").strip()
    text = " ".join(text.split())

    return text


def get_split(index: int) -> str:
    """
    Simple deterministic split per speaker.

    First 80%  -> train
    Next 10%   -> val
    Last 10%   -> test
    """
    if index < 906:
        return "train"

    if index < 1019:
        return "val"

    return "test"


def build_rows():
    rows = []

    for speaker_id, speaker_info in VIETNAMESE_SPEAKERS.items():
        speaker_dir = DATASET_ROOT / speaker_id
        wav_dir = speaker_dir / "wav"
        transcript_dir = speaker_dir / "transcript"

        if not speaker_dir.exists():
            raise FileNotFoundError(f"Speaker folder not found: {speaker_dir}")

        wav_files = sorted(wav_dir.glob("*.wav"))

        for index, wav_path in enumerate(wav_files):
            utterance_id = wav_path.stem
            transcript_path = transcript_dir / f"{utterance_id}.txt"
            target_text = read_transcript(transcript_path)

            rows.append(
                {
                    "dataset": "l2_arctic",
                    "speaker_id": speaker_id,
                    "l1": speaker_info["l1"],
                    "gender": speaker_info["gender"],
                    "split": get_split(index),
                    "audio_path": str(wav_path).replace("\\", "/"),
                    "target_text": target_text,
                    "label": 1,
                    "label_name": "non_native",
                    "source_type": "l2_learner",
                    "utterance_id": utterance_id,
                }
            )

    return rows


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    rows = build_rows()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "speaker_id",
        "l1",
        "gender",
        "split",
        "audio_path",
        "target_text",
        "label",
        "label_name",
        "source_type",
        "utterance_id",
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metadata to: {OUTPUT_CSV}")
    print(f"Total rows: {len(rows)}")

    split_counts = {}

    for row in rows:
        key = row["split"]
        split_counts[key] = split_counts.get(key, 0) + 1

    print("Split counts:")

    for split, count in split_counts.items():
        print(f"- {split}: {count}")


if __name__ == "__main__":
    main()