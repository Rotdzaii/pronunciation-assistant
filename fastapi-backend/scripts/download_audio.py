#!/usr/bin/env python3
"""
fastapi-backend/scripts/download_audio.py

One-time script: pre-download pronunciation audio to Supabase Storage.
Run from project root or fastapi-backend/:

    .venv\\Scripts\\python scripts\\download_audio.py

Reads creds from fastapi-backend/.env (same pattern as seed_demo_data.py).
Writes manifest → fastapi-backend/app/data/pronunciation_manifest.json
Writes attribution → fastapi-backend/scripts/ATTRIBUTION.md
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

from supabase import create_client


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parents[1]   # fastapi-backend/
MANIFEST_OUT = BASE_DIR / "app" / "data" / "pronunciation_manifest.json"
ATTRIBUTION  = BASE_DIR / "scripts" / "ATTRIBUTION.md"

BUCKET = "pronunciation-audio"

# ── Phoneme map (mirrors frontend/lib/phonemeAudio.ts) ────────────────────────

IPA_COMMONS_FILE: dict[str, str] = {
    # Monophthong vowels
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
    # Consonants — stops
    "/p/":  "Voiceless_bilabial_plosive.ogg",
    "/b/":  "Voiced_bilabial_plosive.ogg",
    "/t/":  "Voiceless_alveolar_plosive.ogg",
    "/d/":  "Voiced_alveolar_plosive.ogg",
    "/k/":  "Voiceless_velar_plosive.ogg",
    "/ɡ/":  "Voiced_velar_plosive.ogg",
    # Consonants — fricatives
    "/f/":  "Voiceless_labiodental_fricative.ogg",
    "/v/":  "Voiced_labiodental_fricative.ogg",
    "/θ/":  "Voiceless_dental_fricative.ogg",
    "/ð/":  "Voiced_dental_fricative.ogg",
    "/s/":  "Voiceless_alveolar_sibilant.ogg",
    "/z/":  "Voiced_alveolar_sibilant.ogg",
    "/ʃ/":  "Voiceless_palato-alveolar_sibilant.ogg",
    "/ʒ/":  "Voiced_palato-alveolar_sibilant.ogg",
    "/h/":  "Voiceless_glottal_fricative.ogg",
    # Consonants — affricates
    "/tʃ/": "Voiceless_palato-alveolar_affricate.ogg",
    "/dʒ/": "Voiced_palato-alveolar_affricate.ogg",
    # Consonants — nasals
    "/m/":  "Bilabial_nasal.ogg",
    "/n/":  "Alveolar_nasal.ogg",
    "/ŋ/":  "Velar_nasal.ogg",
    # Consonants — approximants
    "/l/":  "Alveolar_lateral_approximant.ogg",
    "/r/":  "Alveolar_approximant.ogg",
    "/j/":  "Palatal_approximant.ogg",
    "/w/":  "Voiced_labio-velar_approximant.ogg",
}

# Representative words for the 8 diphthongs (no standalone IPA Commons file).
DIPHTHONG_WORDS: dict[str, str] = {
    "/eɪ/": "day",
    "/aɪ/": "time",
    "/ɔɪ/": "boy",
    "/aʊ/": "now",
    "/əʊ/": "go",
    "/ɪə/": "here",
    "/eə/": "there",
    "/ʊə/": "sure",
}

COMMONS_BASE  = "https://commons.wikimedia.org/wiki/Special:FilePath/"
FREE_DICT_API = "https://api.dictionaryapi.dev/api/v2/entries/en/"

# Wikimedia requires a descriptive User-Agent for automated file downloads.
# See: https://www.mediawiki.org/wiki/API:Etiquette
WIKIMEDIA_UA = (
    "PronunciationAssistantThesis/1.0 "
    "(KLTN pronunciation app; educational use) "
    "python-httpx/0.28"
)


# ── Environment / Supabase bootstrap ─────────────────────────────────────────

def _required_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"ERROR: {key} is not set in fastapi-backend/.env", file=sys.stderr)
        sys.exit(1)
    return val


def _bootstrap():
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")
    url  = _required_env("SUPABASE_URL")
    key  = _required_env("SUPABASE_SERVICE_ROLE_KEY")
    return url, create_client(url, key)


# ── Supabase Storage helpers ──────────────────────────────────────────────────

def ensure_bucket(supabase) -> None:
    try:
        supabase.storage.create_bucket(BUCKET, options={"public": True})
        print(f"  Created bucket '{BUCKET}'")
    except Exception as exc:
        msg = str(exc).lower()
        if "already exists" in msg or "duplicate" in msg or "violates" in msg:
            print(f"  Bucket '{BUCKET}' already exists — OK")
        else:
            raise


def upload(supabase, storage_path: str, data: bytes, content_type: str) -> bool:
    try:
        supabase.storage.from_(BUCKET).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return True
    except Exception as exc:
        print(f"    UPLOAD ERROR {storage_path}: {exc}")
        return False


# ── HTTP download helpers ─────────────────────────────────────────────────────

async def fetch_bytes(
    client: httpx.AsyncClient,
    url: str,
    headers: dict | None = None,
) -> bytes | None:
    try:
        r = await client.get(url, follow_redirects=True, timeout=30.0, headers=headers or {})
        if r.status_code == 200:
            return r.content
        print(f"    HTTP {r.status_code} fetching {url}")
        return None
    except Exception as exc:
        print(f"    FETCH ERROR {url}: {exc}")
        return None


def commons_direct_url(filename: str) -> str:
    """Compute the stable upload.wikimedia.org URL for a Commons file.

    Wikimedia derives the storage path from the MD5 hash of the normalised
    filename (spaces→underscores, first letter upper-cased).  For the OGG files
    used here the filename is already in canonical form so we hash it directly.

    Formula: https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[0:2]}/{filename}
    where h = md5(filename, utf-8).hexdigest()
    """
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/{filename}"


async def fetch_word_audio_url(client: httpx.AsyncClient, word: str) -> str | None:
    try:
        r = await client.get(f"{FREE_DICT_API}{word}", timeout=15.0, follow_redirects=True)
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            print(f"    HTTP {r.status_code} for word '{word}'")
            return None
        for entry in r.json():
            for phonetic in entry.get("phonetics", []):
                url = phonetic.get("audio") or ""
                if url:
                    # some entries return protocol-relative URLs
                    return ("https:" + url) if url.startswith("//") else url
        return None
    except Exception as exc:
        print(f"    FETCH ERROR word '{word}': {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    supabase_url, supabase = _bootstrap()
    print(f"Supabase: {supabase_url}")

    ensure_bucket(supabase)

    # Load existing manifest so re-runs are incremental (already-OK entries kept).
    existing: dict = {}
    if MANIFEST_OUT.exists():
        try:
            existing = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
            print(f"  Loaded existing manifest: {len(existing.get('phonemes',{}))} phonemes, {len(existing.get('words',{}))} words")
        except Exception:
            pass

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "supabase_url": supabase_url,
        "bucket": BUCKET,
        "phonemes": dict(existing.get("phonemes", {})),
        "words":    dict(existing.get("words", {})),
        "failures": [],
    }

    async with httpx.AsyncClient() as client:

        # ── Step 1: Phoneme OGGs from Wikimedia Commons ───────────────────────
        sep = "=" * 60
        print(f"\n{sep}")
        print("Step 1 — Phoneme audio from Wikimedia Commons")
        print(sep)

        for phoneme, filename in IPA_COMMONS_FILE.items():
            if phoneme in manifest["phonemes"]:
                print(f"  {phoneme:8s} already in manifest — skip")
                continue
            print(f"  {phoneme:8s} {filename}", end=" … ", flush=True)

            # Build direct URL via deterministic MD5 formula (no API call needed)
            direct_url = commons_direct_url(filename)
            data = await fetch_bytes(client, direct_url, headers={"User-Agent": WIKIMEDIA_UA})
            if data and upload(supabase, f"phonemes/{filename}", data, "audio/ogg"):
                manifest["phonemes"][phoneme] = {
                    "storage_path": f"phonemes/{filename}",
                    "source_url": f"https://commons.wikimedia.org/wiki/File:{filename}",
                    "direct_url": direct_url,
                    "license": "CC BY-SA 3.0",
                }
                print("OK")
            else:
                manifest["failures"].append(f"phoneme:{phoneme}")
                print("FAILED")

        # ── Step 2: Vocabulary words from DB ─────────────────────────────────
        print(f"\n{sep}")
        print("Step 2 — Fetch distinct words from vocabulary_items")
        print(sep)

        try:
            resp = supabase.table("vocabulary_items").select("word").eq("is_active", True).execute()
            db_words: set[str] = {
                r["word"].strip().lower()
                for r in (resp.data or [])
                if r.get("word")
            }
            print(f"  {len(db_words)} distinct words from DB")
        except Exception as exc:
            print(f"  ERROR querying DB: {exc}")
            db_words = set()

        # ── Step 3: All words (DB + diphthong reps) ──────────────────────────
        diphthong_reps = set(DIPHTHONG_WORDS.values())
        all_words = sorted(db_words | diphthong_reps)

        print(f"\n{sep}")
        print(f"Step 3 — Word audio from Free Dictionary API ({len(all_words)} words)")
        print(sep)

        for word in all_words:
            if word in manifest["words"]:
                print(f"  {word:20s} already in manifest — skip")
                continue
            print(f"  {word:20s}", end=" … ", flush=True)
            audio_url = await fetch_word_audio_url(client, word)
            if not audio_url:
                print("no audio found")
                manifest["failures"].append(f"word:{word}")
                continue

            ext = ".ogg" if audio_url.endswith(".ogg") else ".mp3"
            content_type = "audio/ogg" if ext == ".ogg" else "audio/mpeg"
            storage_path = f"words/{word}{ext}"

            data = await fetch_bytes(client, audio_url)
            if data and upload(supabase, storage_path, data, content_type):
                manifest["words"][word] = {
                    "storage_path": storage_path,
                    "source_url": f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
                    "audio_source_url": audio_url,
                    "license": "CC BY-SA 3.0",
                }
                print("OK")
            else:
                manifest["failures"].append(f"word:{word}")
                print("FAILED")

    # ── Embed computed phoneme URLs in manifest for manual follow-up ──────────
    # These direct upload.wikimedia.org URLs may be blocked from the current
    # machine/IP.  Open them in a browser to verify, then download manually
    # and upload to pronunciation-audio/phonemes/ in Supabase Storage.
    phoneme_manual: dict = {}
    for phoneme, filename in IPA_COMMONS_FILE.items():
        if phoneme not in manifest["phonemes"]:
            phoneme_manual[phoneme] = {
                "filename": filename,
                "storage_path": f"phonemes/{filename}",
                "direct_url": commons_direct_url(filename),
                "page_url": f"https://commons.wikimedia.org/wiki/File:{filename}",
                "license": "CC BY-SA 3.0",
            }
    if phoneme_manual:
        manifest["phoneme_manual_upload"] = phoneme_manual

    # ── Write manifest ─────────────────────────────────────────────────────────
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest → {MANIFEST_OUT}")

    # ── Write attribution ──────────────────────────────────────────────────────
    _write_attribution(manifest)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Phonemes : {len(manifest['phonemes'])}/{len(IPA_COMMONS_FILE)} OK")
    print(f"  Words    : {len(manifest['words'])}/{len(all_words)} OK")
    phoneme_fails = [f for f in manifest["failures"] if f.startswith("phoneme:")]
    word_fails    = [f for f in manifest["failures"] if f.startswith("word:")]
    if word_fails:
        print(f"\n  Words with no audio (manual review):")
        for item in word_fails:
            print(f"    - {item.removeprefix('word:')}")
    if phoneme_fails:
        print(f"\n  Phoneme audio blocked (Wikimedia unreachable from this IP).")
        print( "  Direct URLs for manual download saved to manifest['phoneme_manual_upload'].")
        print( "  These also appear in ATTRIBUTION.md under 'Manual Upload Required'.")


def _write_attribution(manifest: dict) -> None:
    lines = [
        "# Pronunciation Audio — Attribution",
        "",
        f"_Generated: {manifest['generated_at']}_",
        f"_Bucket: `{manifest['bucket']}` on {manifest['supabase_url']}_",
        "",
        "---",
        "",
        "## Âm vị IPA (Phoneme Audio)",
        "",
        "**Nguồn:** Wikimedia Commons  ",
        "**Giấy phép:** CC BY-SA 3.0 / CC BY-SA 4.0 (per individual file)  ",
        "**URL gốc:** https://commons.wikimedia.org/",
        "",
        "| Âm vị | Tên file gốc | URL nguồn |",
        "|-------|-------------|-----------|",
    ]
    for phoneme, info in sorted(manifest["phonemes"].items()):
        filename = Path(info["storage_path"]).name
        lines.append(f"| `{phoneme}` | `{filename}` | {info['source_url']} |")

    lines += [
        "",
        "---",
        "",
        "## Phát âm từ vựng (Word Audio)",
        "",
        "**Nguồn:** Free Dictionary API (https://dictionaryapi.dev)  ",
        "**Giấy phép:** CC BY-SA 3.0  ",
        "**Ghi chú:** Âm thanh được cung cấp từ các từ điển công khai (LD, Google, etc.)",
        "",
        "| Từ | URL tra cứu | File lưu trữ |",
        "|----|------------|--------------|",
    ]
    for word, info in sorted(manifest["words"].items()):
        lines.append(f"| {word} | {info['source_url']} | `{info['storage_path']}` |")

    word_fails = [f for f in manifest.get("failures", []) if f.startswith("word:")]
    if word_fails:
        lines += [
            "",
            "---",
            "",
            "## Từ không tìm thấy audio (cần xử lý thủ công)",
            "",
        ]
        for item in word_fails:
            lines.append(f"- `{item.removeprefix('word:')}`")

    manual = manifest.get("phoneme_manual_upload", {})
    if manual:
        lines += [
            "",
            "---",
            "",
            "## Phoneme audio — Cần upload thủ công",
            "",
            "Wikimedia Commons bị chặn từ máy chạy script. Tải các file sau trong trình duyệt",
            "rồi upload lên Supabase Storage bucket `pronunciation-audio` đúng đường dẫn.",
            "",
            "| Âm vị | Direct URL | Storage path |",
            "|-------|-----------|--------------|",
        ]
        for phoneme, info in sorted(manual.items()):
            lines.append(
                f"| `{phoneme}` | [{info['filename']}]({info['direct_url']}) "
                f"| `{info['storage_path']}` |"
            )

    ATTRIBUTION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Attribution → {ATTRIBUTION}")


if __name__ == "__main__":
    asyncio.run(main())
