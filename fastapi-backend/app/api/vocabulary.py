import csv
import io
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import (
    CurrentUser,
    get_current_user,
    get_supabase_authenticated_client,
    get_supabase_service_client,
    require_roles,
)
from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vocabulary", tags=["Vocabulary"])

# ── Pronunciation manifest (populated by scripts/download_audio.py) ───────────

def _load_pronunciation_manifest() -> dict:
    path = Path(__file__).parent.parent / "data" / "pronunciation_manifest.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info(
                "Pronunciation manifest loaded: %d phonemes, %d words, %d failures",
                len(data.get("phonemes", {})),
                len(data.get("words", {})),
                len(data.get("failures", [])),
            )
            return data
    except Exception:
        logger.warning("Could not load pronunciation manifest from %s", path, exc_info=True)
    return {"phonemes": {}, "words": {}, "bucket": "pronunciation-audio"}


_pronunciation_manifest = _load_pronunciation_manifest()

# In-memory cache for live fallback calls (only used when
# ENABLE_LIVE_PRONUNCIATION_FALLBACK=true and word is absent from manifest).
_live_pronunciation_cache: dict[str, str | None] = {}

VOCABULARY_ITEM_COLUMNS = (
    "id,word,phonetic,meaning_vi,topic,level,difficulty,sample_sentence,"
    "target_phonemes,common_mistake_tags,stress_pattern"
)

VOCABULARY_SET_COLUMNS = (
    "id,title,description,topic,level,is_public,is_active"
)


def _normalize_expected_phones(phones: list[str], *, variant: str | None) -> list[str]:
    """Use one en-US rhotic-vowel target for the public pronunciation contract."""
    canonical = ["ɹ" if phone == "r" else phone for phone in phones]
    if str(variant or "").replace("_", "-").casefold() != "en-us":
        return canonical

    normalized: list[str] = []
    index = 0
    while index < len(canonical):
        if canonical[index] == "ə" and index + 1 < len(canonical) and canonical[index + 1] == "ɹ":
            normalized.append("ɚ")
            index += 2
            continue
        normalized.append(canonical[index])
        index += 1
    return normalized


def resolve_stored_pronunciation(supabase_client: Any, word: str) -> dict[str, Any]:
    """Resolve the public pronunciation contract from stored vocabulary data.

    MFA output is intentionally not a source here: it aligns the learner audio,
    while ``vocabulary_items.phonetic`` and ``target_phonemes`` define the
    expected pronunciation displayed to learners.
    """
    unavailable = {
        "display_pronunciation": None,
        "expected_phones": [],
        "pronunciation_variant": None,
        "pronunciation_source": "unavailable",
    }
    normalized_word = word.strip()
    if not normalized_word:
        return unavailable

    try:
        rows = (
            supabase_client.table("vocabulary_items")
            .select("phonetic,target_phonemes")
            .eq("word", normalized_word)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception:
        logger.warning("Unable to resolve stored pronunciation for practice result", exc_info=True)
        return unavailable

    if not rows:
        return unavailable

    row = rows[0]
    phonetic = row.get("phonetic")
    display_pronunciation = phonetic.strip() if isinstance(phonetic, str) and phonetic.strip() else None
    raw_phones = row.get("target_phonemes")
    expected_phones = _normalize_expected_phones([
        phone.strip()
        for phone in raw_phones
        if isinstance(phone, str) and phone.strip()
    ] if isinstance(raw_phones, list) else [], variant="en-US")
    if not display_pronunciation and not expected_phones:
        return unavailable

    return {
        "display_pronunciation": display_pronunciation,
        "expected_phones": expected_phones,
        "pronunciation_variant": "en-US",
        "pronunciation_source": "vocabulary_items",
    }


class VocabularyItemResponse(BaseModel):
    id: UUID
    word: str
    phonetic: str | None = None
    meaning_vi: str | None = None
    topic: str | None = None
    level: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=5)
    sample_sentence: str | None = None
    target_phonemes: list[Any]
    common_mistake_tags: list[Any]
    stress_pattern: str | None = None


class VocabularyItemsResponse(BaseModel):
    items: list[VocabularyItemResponse]
    limit: int
    offset: int


class VocabularySetResponse(BaseModel):
    id: UUID
    title: str
    description: str | None = None
    topic: str | None = None
    level: str | None = None
    is_public: bool
    is_active: bool
    item_count: int | None = None


class VocabularySetsResponse(BaseModel):
    items: list[VocabularySetResponse]
    limit: int
    offset: int


class VocabularySetDetailResponse(BaseModel):
    id: UUID
    title: str
    description: str | None = None
    topic: str | None = None
    level: str | None = None
    is_public: bool
    is_active: bool
    items: list[VocabularyItemResponse]


class VocabularySetCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    topic: str | None = None
    level: str | None = None
    is_public: bool = True


class VocabularySetUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    topic: str | None = None
    level: str | None = None
    is_public: bool | None = None
    is_active: bool | None = None


class VocabularyItemCreate(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)
    phonetic: str | None = None
    meaning_vi: str | None = None
    topic: str | None = None
    level: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=5)
    sample_sentence: str | None = None
    target_phonemes: list[Any] = Field(default_factory=list)
    common_mistake_tags: list[Any] = Field(default_factory=list)
    stress_pattern: str | None = None


class VocabularyItemUpdate(BaseModel):
    word: str | None = Field(default=None, min_length=1, max_length=200)
    phonetic: str | None = None
    meaning_vi: str | None = None
    topic: str | None = None
    level: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=5)
    sample_sentence: str | None = None
    target_phonemes: list[Any] | None = None
    common_mistake_tags: list[Any] | None = None
    stress_pattern: str | None = None
    is_active: bool | None = None


class VocabularyImportResponse(BaseModel):
    imported_count: int
    skipped_count: int
    errors: list[str]


def _clean_filter(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _get_vocabulary_client(current_user: CurrentUser, settings: Settings):
    return get_supabase_authenticated_client(settings, current_user._access_token)


def _require_set_owner(supabase_client: Any, set_id: str, current_user: CurrentUser) -> dict[str, Any]:
    rows = (
        supabase_client.table("vocabulary_sets")
        .select("id,created_by,is_active")
        .eq("id", set_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vocabulary set not found")
    row = rows[0]
    if current_user.app_role != "admin" and row.get("created_by") != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to modify this vocabulary set")
    return row


@router.get("/items", response_model=VocabularyItemsResponse)
def list_vocabulary_items(
    topic: str | None = Query(default=None),
    level: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> VocabularyItemsResponse:
    supabase_client = _get_vocabulary_client(current_user, settings)

    try:
        query = (
            supabase_client.table("vocabulary_items")
            .select(VOCABULARY_ITEM_COLUMNS)
            .eq("is_active", True)
            .order("word")
            .range(offset, offset + limit - 1)
        )

        topic_filter = _clean_filter(topic)
        if topic_filter:
            query = query.eq("topic", topic_filter)

        level_filter = _clean_filter(level)
        if level_filter:
            query = query.eq("level", level_filter)

        response = query.execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load vocabulary items",
        ) from exc

    return VocabularyItemsResponse(
        items=[VocabularyItemResponse(**item) for item in response.data or []],
        limit=limit,
        offset=offset,
    )


@router.get("/sets", response_model=VocabularySetsResponse)
def list_vocabulary_sets(
    topic: str | None = Query(default=None),
    level: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> VocabularySetsResponse:
    supabase_client = _get_vocabulary_client(current_user, settings)

    try:
        query = (
            supabase_client.table("vocabulary_sets")
            .select(VOCABULARY_SET_COLUMNS)
            .eq("is_active", True)
            .eq("is_public", True)
            .order("title")
            .range(offset, offset + limit - 1)
        )

        topic_filter = _clean_filter(topic)
        if topic_filter:
            query = query.eq("topic", topic_filter)

        level_filter = _clean_filter(level)
        if level_filter:
            query = query.eq("level", level_filter)

        response = query.execute()
        sets = response.data or []

        item_counts: dict[str, int] = {}
        set_ids = [item["id"] for item in sets]
        if set_ids:
            links_response = (
                supabase_client.table("vocabulary_set_items")
                .select("set_id")
                .in_("set_id", set_ids)
                .execute()
            )
            for link in links_response.data or []:
                set_id = str(link["set_id"])
                item_counts[set_id] = item_counts.get(set_id, 0) + 1
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load vocabulary sets",
        ) from exc

    return VocabularySetsResponse(
        items=[
            VocabularySetResponse(
                **item,
                item_count=item_counts.get(str(item["id"]), 0),
            )
            for item in sets
        ],
        limit=limit,
        offset=offset,
    )


@router.get("/sets/{set_id}", response_model=VocabularySetDetailResponse)
def get_vocabulary_set(
    set_id: UUID,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> VocabularySetDetailResponse:
    supabase_client = _get_vocabulary_client(current_user, settings)

    try:
        set_response = (
            supabase_client.table("vocabulary_sets")
            .select(VOCABULARY_SET_COLUMNS)
            .eq("id", str(set_id))
            .eq("is_active", True)
            .eq("is_public", True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load vocabulary set",
        ) from exc

    sets = set_response.data or []
    if not sets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vocabulary set not found",
        )

    try:
        links_response = (
            supabase_client.table("vocabulary_set_items")
            .select("item_id,sort_order")
            .eq("set_id", str(set_id))
            .order("sort_order")
            .execute()
        )
        links = links_response.data or []
        item_ids = [link["item_id"] for link in links]

        items_by_id: dict[str, dict[str, Any]] = {}
        if item_ids:
            items_response = (
                supabase_client.table("vocabulary_items")
                .select(VOCABULARY_ITEM_COLUMNS)
                .eq("is_active", True)
                .in_("id", item_ids)
                .execute()
            )
            items_by_id = {
                str(item["id"]): item for item in items_response.data or []
            }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load vocabulary set items",
        ) from exc

    ordered_items = [
        VocabularyItemResponse(**items_by_id[str(link["item_id"])])
        for link in links
        if str(link["item_id"]) in items_by_id
    ]

    return VocabularySetDetailResponse(
        **sets[0],
        items=ordered_items,
    )


@router.post("/sets", response_model=VocabularySetResponse, status_code=status.HTTP_201_CREATED)
def create_vocabulary_set(
    payload: VocabularySetCreate,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> VocabularySetResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        row = (
            supabase_client.table("vocabulary_sets")
            .insert({
                "title": payload.title,
                "description": payload.description,
                "topic": payload.topic,
                "level": payload.level,
                "is_public": payload.is_public,
                "is_active": True,
                "created_by": current_user.id,
            })
            .execute()
            .data[0]
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create vocabulary set") from exc
    return VocabularySetResponse(**row, item_count=0)


@router.patch("/sets/{set_id}", response_model=VocabularySetResponse)
def update_vocabulary_set(
    set_id: UUID,
    payload: VocabularySetUpdate,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> VocabularySetResponse:
    supabase_client = get_supabase_service_client(settings)
    _require_set_owner(supabase_client, str(set_id), current_user)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")
    try:
        rows = (
            supabase_client.table("vocabulary_sets")
            .update(updates)
            .eq("id", str(set_id))
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to update vocabulary set") from exc
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vocabulary set not found")
    return VocabularySetResponse(**rows[0])


@router.delete("/sets/{set_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vocabulary_set(
    set_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> None:
    supabase_client = get_supabase_service_client(settings)
    _require_set_owner(supabase_client, str(set_id), current_user)
    try:
        supabase_client.table("vocabulary_sets").update({"is_active": False}).eq("id", str(set_id)).execute()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to delete vocabulary set") from exc


@router.post("/sets/{set_id}/import", response_model=VocabularyImportResponse)
def import_vocabulary_set_items(
    set_id: UUID,
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> VocabularyImportResponse:
    supabase_client = get_supabase_service_client(settings)
    _require_set_owner(supabase_client, str(set_id), current_user)
    try:
        content = file.file.read().decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to read uploaded file") from exc

    reader = csv.DictReader(io.StringIO(content))
    imported = 0
    skipped = 0
    errors: list[str] = []

    for i, row in enumerate(reader, start=2):
        word = (row.get("word") or "").strip()
        if not word:
            errors.append(f"Row {i}: missing word")
            skipped += 1
            continue
        try:
            difficulty_raw = (row.get("difficulty") or "").strip()
            difficulty = int(difficulty_raw) if difficulty_raw.isdigit() else None

            item_rows = (
                supabase_client.table("vocabulary_items")
                .insert({
                    "word": word,
                    "phonetic": (row.get("phonetic") or "").strip() or None,
                    "meaning_vi": (row.get("meaning_vi") or "").strip() or None,
                    "topic": (row.get("topic") or "").strip() or None,
                    "level": (row.get("level") or "").strip() or None,
                    "difficulty": difficulty,
                    "sample_sentence": (row.get("sample_sentence") or "").strip() or None,
                    "target_phonemes": [],
                    "common_mistake_tags": [],
                    "is_active": True,
                    "created_by": current_user.id,
                })
                .execute()
                .data
            )
            if not item_rows:
                skipped += 1
                continue
            item_id = item_rows[0]["id"]
            existing = (
                supabase_client.table("vocabulary_set_items")
                .select("id")
                .eq("set_id", str(set_id))
                .eq("item_id", item_id)
                .execute()
                .data
            )
            if existing:
                skipped += 1
                continue
            max_order_rows = (
                supabase_client.table("vocabulary_set_items")
                .select("sort_order")
                .eq("set_id", str(set_id))
                .order("sort_order", desc=True)
                .limit(1)
                .execute()
                .data
            )
            next_order = (max_order_rows[0]["sort_order"] + 1) if max_order_rows else 0
            supabase_client.table("vocabulary_set_items").insert({
                "set_id": str(set_id),
                "item_id": item_id,
                "sort_order": next_order,
            }).execute()
            imported += 1
        except Exception as exc:
            errors.append(f"Row {i} ({word}): {exc}")
            skipped += 1

    return VocabularyImportResponse(imported_count=imported, skipped_count=skipped, errors=errors[:20])


@router.get("/sets/{set_id}/export")
def export_vocabulary_set(
    set_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    supabase_client = get_supabase_service_client(settings)
    _require_set_owner(supabase_client, str(set_id), current_user)
    try:
        links = (
            supabase_client.table("vocabulary_set_items")
            .select("item_id,sort_order")
            .eq("set_id", str(set_id))
            .order("sort_order")
            .execute()
            .data
            or []
        )
        item_ids = [link["item_id"] for link in links]
        items: list[dict[str, Any]] = []
        if item_ids:
            items = (
                supabase_client.table("vocabulary_items")
                .select(VOCABULARY_ITEM_COLUMNS)
                .in_("id", item_ids)
                .execute()
                .data
                or []
            )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to export vocabulary set") from exc

    items_by_id = {str(item["id"]): item for item in items}
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["word", "phonetic", "meaning_vi", "topic", "level", "difficulty", "sample_sentence"],
    )
    writer.writeheader()
    for link in links:
        item = items_by_id.get(str(link["item_id"]))
        if item:
            writer.writerow({
                "word": item.get("word", ""),
                "phonetic": item.get("phonetic", ""),
                "meaning_vi": item.get("meaning_vi", ""),
                "topic": item.get("topic", ""),
                "level": item.get("level", ""),
                "difficulty": item.get("difficulty", ""),
                "sample_sentence": item.get("sample_sentence", ""),
            })

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=vocabulary_set_{set_id}.csv"},
    )


@router.post("/items", response_model=VocabularyItemResponse, status_code=status.HTTP_201_CREATED)
def create_vocabulary_item(
    payload: VocabularyItemCreate,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> VocabularyItemResponse:
    supabase_client = get_supabase_service_client(settings)
    try:
        row = (
            supabase_client.table("vocabulary_items")
            .insert({
                "word": payload.word,
                "phonetic": payload.phonetic,
                "meaning_vi": payload.meaning_vi,
                "topic": payload.topic,
                "level": payload.level,
                "difficulty": payload.difficulty,
                "sample_sentence": payload.sample_sentence,
                "target_phonemes": payload.target_phonemes,
                "common_mistake_tags": payload.common_mistake_tags,
                "stress_pattern": payload.stress_pattern,
                "is_active": True,
                "created_by": current_user.id,
            })
            .execute()
            .data[0]
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to create vocabulary item") from exc
    return VocabularyItemResponse(**row)


@router.patch("/items/{item_id}", response_model=VocabularyItemResponse)
def update_vocabulary_item(
    item_id: UUID,
    payload: VocabularyItemUpdate,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> VocabularyItemResponse:
    supabase_client = get_supabase_service_client(settings)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")
    try:
        rows = (
            supabase_client.table("vocabulary_items")
            .update(updates)
            .eq("id", str(item_id))
            .execute()
            .data
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to update vocabulary item") from exc
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vocabulary item not found")
    return VocabularyItemResponse(**rows[0])


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vocabulary_item(
    item_id: UUID,
    current_user: CurrentUser = Depends(require_roles(["teacher"])),
    settings: Settings = Depends(get_settings),
) -> None:
    supabase_client = get_supabase_service_client(settings)
    try:
        supabase_client.table("vocabulary_items").update({"is_active": False}).eq("id", str(item_id)).execute()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to delete vocabulary item") from exc


@router.get("/pronunciation/{word}")
async def get_word_pronunciation(
    word: str,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict[str, str | None]:
    """Return a Supabase Storage URL for the pre-downloaded pronunciation of
    *word*.  Falls back to a live Free Dictionary API call only when
    ENABLE_LIVE_PRONUNCIATION_FALLBACK=true (disabled by default for demo).
    """
    word_lower = word.strip().lower()

    # ── Manifest lookup (no network, always attempted first) ──────────────────
    word_entry = _pronunciation_manifest.get("words", {}).get(word_lower)
    if word_entry:
        base   = (settings.supabase_url or "").rstrip("/")
        bucket = _pronunciation_manifest.get("bucket", settings.pronunciation_audio_bucket)
        url    = f"{base}/storage/v1/object/public/{bucket}/{word_entry['storage_path']}"
        return {"audio_url": url}

    # ── Live fallback: opt-in only ────────────────────────────────────────────
    if not settings.enable_live_pronunciation_fallback:
        return {"audio_url": None}

    if word_lower in _live_pronunciation_cache:
        return {"audio_url": _live_pronunciation_cache[word_lower]}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_lower}"
            )

        if response.status_code == 404:
            _live_pronunciation_cache[word_lower] = None
            return {"audio_url": None}

        if response.status_code != 200:
            return {"audio_url": None}

        audio_url: str | None = None
        for entry in response.json():
            for phonetic in entry.get("phonetics", []):
                url = phonetic.get("audio") or ""
                if url:
                    audio_url = ("https:" + url) if url.startswith("//") else url
                    break
            if audio_url:
                break

        _live_pronunciation_cache[word_lower] = audio_url
        return {"audio_url": audio_url}

    except Exception:
        return {"audio_url": None}
