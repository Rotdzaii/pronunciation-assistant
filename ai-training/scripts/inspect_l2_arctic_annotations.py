from pathlib import Path
import re


DATASET_ROOT = Path("ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0")
VIETNAMESE_SPEAKERS = ("HQTV", "PNV", "THV", "TLV")


def safe_preview(path: Path, max_lines: int = 80):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[:max_lines]

    for line in lines:
        print(line)


def print_tier_names(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    tier_names = re.findall(r'name = "(.*?)"', text)
    print("Tier names:", ", ".join(tier_names) if tier_names else "(none found)")


def inspect_speaker(speaker_id: str):
    speaker_dir = DATASET_ROOT / speaker_id
    annotation_dir = speaker_dir / "annotation"
    textgrid_dir = speaker_dir / "textgrid"

    annotation_files = sorted(annotation_dir.glob("*.TextGrid")) if annotation_dir.exists() else []
    textgrid_files = sorted(textgrid_dir.glob("*.TextGrid")) if textgrid_dir.exists() else []

    print("=" * 80)
    print(f"Speaker: {speaker_id}")
    print(f"annotation/*.TextGrid: {len(annotation_files)}")
    print(f"textgrid/*.TextGrid: {len(textgrid_files)}")
    print("Sample annotation filenames:")

    for path in annotation_files[:10]:
        print(f"- {path.name}")

    if not annotation_files:
        print("No annotation TextGrid files found.")
        return

    sample_path = annotation_files[0]
    print()
    print(f"Preview: {sample_path}")
    print_tier_names(sample_path)
    safe_preview(sample_path)


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    for speaker_id in VIETNAMESE_SPEAKERS:
        inspect_speaker(speaker_id)


if __name__ == "__main__":
    main()
