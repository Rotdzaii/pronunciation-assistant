#!/usr/bin/env python3
"""
fastapi-backend/scripts/retry_phoneme_download.py

Retry downloading the 36 phoneme OGG files from Wikimedia Commons using
browser-like request behaviour to avoid IP-based bot filtering.

Saves files to  fastapi-backend/downloads/phonemes/  then calls
upload_phoneme_audio.py to push them to Supabase Storage.

Usage (from fastapi-backend/):
    .venv\\Scripts\\python scripts\\retry_phoneme_download.py
"""

import asyncio
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parents[1]      # fastapi-backend/
MANIFEST_PATH = BASE_DIR / "app" / "data" / "pronunciation_manifest.json"
DOWNLOAD_DIR  = BASE_DIR / "downloads" / "phonemes"

# ── Phoneme map (must stay in sync with download_audio.py and phonemeAudio.ts) ─

IPA_COMMONS_FILE: dict[str, str] = {
    "/iː/": "Close_front_unrounded_vowel.ogg",
    "/ɪ/":  "Near-close_near-front_unrounded_vowel.ogg",
    "/e/":  "Close-mid_front_unrounded_vowel.ogg",
    "/æ/":  "Near-open_front_unrounded_vowel.ogg",
    "/ɑː/": "Open_back_unrounded_vowel.ogg",
    "/ɒ/":  "Open_back_rounded_vowel.ogg",
    "/ɔː/": "Open-mid_back_rounded_vowel.ogg",
    "/ʊ/":  "Near-close_near-back_rounded_vowel.ogg",
    "/uː/": "Close_back_rounded_vowel.ogg",
    "/ʌ/":  "Open-mid_back_unrounded_vowel.ogg",
    "/ɜː/": "Open-mid_central_unrounded_vowel.ogg",
    "/ə/":  "Mid-central_vowel.ogg",
    "/p/":  "Voiceless_bilabial_plosive.ogg",
    "/b/":  "Voiced_bilabial_plosive.ogg",
    "/t/":  "Voiceless_alveolar_plosive.ogg",
    "/d/":  "Voiced_alveolar_plosive.ogg",
    "/k/":  "Voiceless_velar_plosive.ogg",
    "/ɡ/":  "Voiced_velar_plosive.ogg",
    "/f/":  "Voiceless_labiodental_fricative.ogg",
    "/v/":  "Voiced_labiodental_fricative.ogg",
    "/θ/":  "Voiceless_dental_fricative.ogg",
    "/ð/":  "Voiced_dental_fricative.ogg",
    "/s/":  "Voiceless_alveolar_sibilant.ogg",
    "/z/":  "Voiced_alveolar_sibilant.ogg",
    "/ʃ/":  "Voiceless_palato-alveolar_sibilant.ogg",
    "/ʒ/":  "Voiced_palato-alveolar_sibilant.ogg",
    "/h/":  "Voiceless_glottal_fricative.ogg",
    "/tʃ/": "Voiceless_palato-alveolar_affricate.ogg",
    "/dʒ/": "Voiced_palato-alveolar_affricate.ogg",
    "/m/":  "Bilabial_nasal.ogg",
    "/n/":  "Alveolar_nasal.ogg",
    "/ŋ/":  "Velar_nasal.ogg",
    "/l/":  "Alveolar_lateral_approximant.ogg",
    "/r/":  "Alveolar_approximant.ogg",
    "/j/":  "Palatal_approximant.ogg",
    "/w/":  "Voiced_labio-velar_approximant.ogg",
}

# Realistic browser User-Agent (Chrome 124 on Windows 10)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "audio/ogg,audio/*;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://commons.wikimedia.org/",
    "Connection": "keep-alive",
    "DNT": "1",
}


def commons_direct_url(filename: str) -> str:
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/{filename}"


def already_uploaded(phoneme: str) -> bool:
    if not MANIFEST_PATH.exists():
        return False
    try:
        m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return phoneme in m.get("phonemes", {})
    except Exception:
        return False


def already_downloaded(filename: str) -> bool:
    return (DOWNLOAD_DIR / filename).exists()


async def fetch_one(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    phoneme: str,
) -> bool:
    """Attempt to download url → DOWNLOAD_DIR/filename.  Returns True on success."""

    async def _try(attempt: int) -> bytes | None:
        try:
            r = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True, timeout=45.0)
            if r.status_code == 200:
                return r.content
            if r.status_code == 403 and attempt == 1:
                wait = random.uniform(30.0, 60.0)
                print(f"    403 on attempt 1 — waiting {wait:.0f}s before retry …", flush=True)
                await asyncio.sleep(wait)
                return None   # signal to retry
            print(f"    HTTP {r.status_code} (attempt {attempt})")
            return b""        # non-retryable failure — empty sentinel
        except Exception as exc:
            print(f"    FETCH ERROR (attempt {attempt}): {exc}")
            return b""        # treat as non-retryable

    data = await _try(1)
    if data is None:           # 403 first attempt → retry
        data = await _try(2)

    if not data:               # empty sentinel or second failure
        return False

    (DOWNLOAD_DIR / filename).write_bytes(data)
    return True


async def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str]] = []   # (phoneme, filename)
    for phoneme, filename in IPA_COMMONS_FILE.items():
        if already_uploaded(phoneme):
            print(f"  {phoneme:8s} already in Supabase manifest — skip")
            continue
        if already_downloaded(filename):
            print(f"  {phoneme:8s} already in downloads/ — will upload")
            continue
        pending.append((phoneme, filename))

    print(f"\n{len(pending)} phoneme(s) to download (sequential, browser-like headers)")
    print(f"Downloads → {DOWNLOAD_DIR}\n")

    succeeded: list[str] = []
    failed: list[str]    = []

    async with httpx.AsyncClient() as client:
        for i, (phoneme, filename) in enumerate(pending):
            url = commons_direct_url(filename)
            print(f"  [{i+1}/{len(pending)}] {phoneme:8s} {filename}", end=" … ", flush=True)

            ok = await fetch_one(client, url, filename, phoneme)
            if ok:
                print("OK")
                succeeded.append(phoneme)
            else:
                print("FAILED")
                failed.append(phoneme)

            # Randomized inter-request delay (skip after last file)
            if i < len(pending) - 1:
                delay = random.uniform(3.0, 8.0)
                print(f"    sleeping {delay:.1f}s …")
                await asyncio.sleep(delay)

    sep = "=" * 60
    print(f"\n{sep}")
    print("DOWNLOAD SUMMARY")
    print(sep)
    already_on_disk = [
        p for p, f in IPA_COMMONS_FILE.items()
        if already_downloaded(f) and not already_uploaded(p) and (p, f) not in pending
    ] + succeeded
    print(f"  Files in downloads/ ready to upload : {len(already_on_disk) + len(succeeded)}")
    print(f"  Downloaded this run                 : {len(succeeded)}")
    if failed:
        print(f"  Still failed after retry            : {len(failed)}")
        for p in failed:
            print(f"    - {p}  ({IPA_COMMONS_FILE[p]})")
    else:
        print("  No failures — all pending files downloaded!")

    # ── Auto-run upload_phoneme_audio.py ─────────────────────────────────────
    ogg_count = len(list(DOWNLOAD_DIR.glob("*.ogg")))
    if ogg_count == 0:
        print("\nNo .ogg files in downloads/ — skipping upload step.")
        return

    print(f"\n{sep}")
    print(f"Running upload_phoneme_audio.py ({ogg_count} file(s) in downloads/) …")
    print(sep)

    upload_script = Path(__file__).parent / "upload_phoneme_audio.py"
    result = subprocess.run(
        [sys.executable, str(upload_script), str(DOWNLOAD_DIR)],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"\nERROR: upload_phoneme_audio.py exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    asyncio.run(main())
