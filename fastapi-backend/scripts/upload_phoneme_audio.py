#!/usr/bin/env python3
"""
fastapi-backend/scripts/upload_phoneme_audio.py

After manually downloading the 36 phoneme OGG files from Wikimedia Commons,
run this script to upload them to Supabase Storage and update the manifest.

Usage (from fastapi-backend/):
    .venv\\Scripts\\python scripts\\upload_phoneme_audio.py [FOLDER]

FOLDER defaults to  downloads/phonemes/  relative to fastapi-backend/.
Each .ogg file in FOLDER whose name matches a pending phoneme entry is
uploaded to  pronunciation-audio/phonemes/{filename}  and the manifest is
updated so the backend immediately recognises it.

Files already present in manifest["phonemes"] are skipped (idempotent).
"""

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from supabase import create_client


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parents[1]          # fastapi-backend/
MANIFEST_PATH = BASE_DIR / "app" / "data" / "pronunciation_manifest.json"
BUCKET       = "pronunciation-audio"


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _required_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"ERROR: {key} not set in fastapi-backend/.env", file=sys.stderr)
        sys.exit(1)
    return val


def _bootstrap():
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")
    return _required_env("SUPABASE_URL"), _required_env("SUPABASE_SERVICE_ROLE_KEY")


# ── Upload helper ─────────────────────────────────────────────────────────────

def upload_file(supabase, storage_path: str, data: bytes) -> bool:
    try:
        supabase.storage.from_(BUCKET).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": "audio/ogg", "upsert": "true"},
        )
        return True
    except Exception as exc:
        print(f"    UPLOAD ERROR {storage_path}: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "downloads" / "phonemes"

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}", file=sys.stderr)
        print( "       Create it and place the downloaded .ogg files inside.", file=sys.stderr)
        sys.exit(1)

    ogg_files = sorted(folder.glob("*.ogg"))
    if not ogg_files:
        print(f"No .ogg files found in {folder}")
        sys.exit(0)

    print(f"Found {len(ogg_files)} .ogg file(s) in {folder}")

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        print( "       Run download_audio.py first to create it.", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pending: dict = manifest.get("phoneme_manual_upload", {})

    if not pending:
        print("No pending phonemes in manifest — nothing to do.")
        sys.exit(0)

    # Build reverse map: filename → phoneme symbol
    filename_to_phoneme: dict[str, str] = {
        info["filename"]: phoneme
        for phoneme, info in pending.items()
    }

    supabase_url, service_key = _bootstrap()
    supabase = create_client(supabase_url, service_key)

    sep = "=" * 60
    print(f"\n{sep}")
    print("Uploading phoneme OGG files to Supabase Storage")
    print(sep)

    uploaded = 0
    skipped  = 0
    failed   = 0

    for ogg_path in ogg_files:
        filename = ogg_path.name

        # Already in manifest["phonemes"]?
        phoneme = filename_to_phoneme.get(filename)
        if phoneme and phoneme in manifest.get("phonemes", {}):
            print(f"  {filename:55s} already uploaded — skip")
            skipped += 1
            continue

        if phoneme is None:
            print(f"  {filename:55s} NOT in pending list — skip")
            skipped += 1
            continue

        storage_path = f"phonemes/{filename}"
        print(f"  {phoneme:6s} {filename}", end=" … ", flush=True)

        data = ogg_path.read_bytes()
        if upload_file(supabase, storage_path, data):
            info = pending[phoneme]
            manifest.setdefault("phonemes", {})[phoneme] = {
                "storage_path": storage_path,
                "source_url":   info.get("page_url", f"https://commons.wikimedia.org/wiki/File:{filename}"),
                "direct_url":   info.get("direct_url", ""),
                "license":      info.get("license", "CC BY-SA 3.0"),
            }
            # Remove from pending
            del manifest["phoneme_manual_upload"][phoneme]
            print("OK")
            uploaded += 1
        else:
            failed += 1

    # Remove phoneme_manual_upload key if empty
    if not manifest.get("phoneme_manual_upload"):
        manifest.pop("phoneme_manual_upload", None)

    # Write updated manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest updated → {MANIFEST_PATH}")

    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped}")
    print(f"  Failed   : {failed}")

    still_pending = len(manifest.get("phoneme_manual_upload", {}))
    if still_pending:
        print(f"  Still pending: {still_pending} phoneme(s) — place their .ogg files in {folder} and re-run")
    else:
        print("  All phonemes uploaded — PhonemeDetailModal audio is fully enabled!")


if __name__ == "__main__":
    main()
