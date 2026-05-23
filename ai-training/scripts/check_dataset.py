from pathlib import Path

import pandas as pd


LABEL_FILE = Path("data/labels/train_labels.csv")


def main():
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_FILE}")

    df = pd.read_csv(LABEL_FILE)

    required_columns = {"audio_path", "label", "label_name", "target_text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    print("Dataset rows:", len(df))
    print()
    print("Label distribution:")
    print(df["label_name"].value_counts())

    print()
    print("Checking audio files...")

    missing_files = []

    for path in df["audio_path"]:
        audio_path = Path(path)
        if not audio_path.exists():
            missing_files.append(str(audio_path))

    if missing_files:
        print("Missing audio files:")
        for item in missing_files:
            print("-", item)
    else:
        print("All audio files exist.")


if __name__ == "__main__":
    main()