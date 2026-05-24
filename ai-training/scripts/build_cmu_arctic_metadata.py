from pathlib import Path
import csv
import re


DATASET_ROOT = Path("ai-training/datasets/cmu-arctic/raw")
OUTPUT_CSV = Path("ai-training/datasets/cmu-arctic/metadata/native_speakers_metadata.csv")

SPEAKERS = {
    "cmu_us_bdl_arctic": {"speaker_id": "bdl", "l1": "English", "gender": "M"},
    "cmu_us_slt_arctic": {"speaker_id": "slt", "l1": "English", "gender": "F"},
}


def parse_txt_done_data(txt_done_path: Path) -> dict[str, str]:
    """
    Parse lines like:
    ( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
    """
    if not txt_done_path.exists():
        print(f"WARNING: Transcript file not found: {txt_done_path}")
        return {}

    prompts = {}

    lines = txt_done_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    pattern = re.compile(r'^\(\s*(\S+)\s+"(.+)"\s*\)$')

    for line in lines:
        line = line.strip()
        match = pattern.match(line)

        if not match:
            continue

        utterance_id = match.group(1)
        text = match.group(2)

        # Normalize spaces and remove punctuation that may make matching noisy later.
        text = " ".join(text.split())
        text = text.replace(",", "")
        text = text.replace(".", "")
        text = text.replace(";", "")
        text = text.replace(":", "")
        text = text.replace("?", "")
        text = text.replace("!", "")

        prompts[utterance_id] = text

    return prompts


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
    missing_audio_paths = []

    for folder_name, speaker_info in SPEAKERS.items():
        speaker_dir = DATASET_ROOT / folder_name
        wav_dir = speaker_dir / "wav"
        txt_done_path = speaker_dir / "etc" / "txt.done.data"

        if not speaker_dir.exists():
            raise FileNotFoundError(f"Speaker folder not found: {speaker_dir}")

        if not wav_dir.exists():
            missing_audio_paths.append(str(wav_dir))
            continue

        prompts = parse_txt_done_data(txt_done_path)
        wav_files = sorted(wav_dir.glob("*.wav"))

        if not wav_files:
            missing_audio_paths.append(str(wav_dir / "*.wav"))
            continue

        for index, wav_path in enumerate(wav_files):
            utterance_id = wav_path.stem
            target_text = prompts.get(utterance_id, "")

            rows.append(
                {
                    "dataset": "cmu_arctic",
                    "speaker_id": speaker_info["speaker_id"],
                    "l1": speaker_info["l1"],
                    "gender": speaker_info["gender"],
                    "split": get_split(index),
                    "audio_path": str(wav_path).replace("\\", "/"),
                    "target_text": target_text,
                    "label": 0,
                    "label_name": "native_reference",
                    "source_type": "native_reference",
                    "utterance_id": utterance_id,
                }
            )

    if missing_audio_paths:
        preview = "\n".join(f"- {path}" for path in missing_audio_paths)
        raise FileNotFoundError(f"Missing CMU ARCTIC audio files or folders:\n{preview}")

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
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    print("Split counts:")

    for split, count in split_counts.items():
        print(f"- {split}: {count}")

    empty_transcripts = [row for row in rows if not row["target_text"]]
    print(f"Empty transcripts: {len(empty_transcripts)}")

    if empty_transcripts:
        print("WARNING: Some CMU ARCTIC rows have empty target_text.")

        if len(empty_transcripts) == 1:
            row = empty_transcripts[0]
            print(
                "Keeping row with missing prompt text: "
                f"utterance_id={row['utterance_id']} "
                f"speaker_id={row['speaker_id']} "
                f"audio_path={row['audio_path']}"
            )
        else:
            print("Rows with empty target_text:")
            for row in empty_transcripts:
                print(
                    "- "
                    f"utterance_id={row['utterance_id']} "
                    f"speaker_id={row['speaker_id']} "
                    f"audio_path={row['audio_path']}"
                )


if __name__ == "__main__":
    main()
