from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.auth import (
    CurrentUser,
    get_current_user,
    get_supabase_authenticated_client,
)
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/vocabulary", tags=["Vocabulary"])

VOCABULARY_ITEM_COLUMNS = (
    "id,word,phonetic,meaning_vi,topic,level,difficulty,sample_sentence,"
    "target_phonemes,common_mistake_tags,stress_pattern"
)

VOCABULARY_SET_COLUMNS = (
    "id,title,description,topic,level,is_public,is_active"
)


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


def _clean_filter(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _get_vocabulary_client(current_user: CurrentUser, settings: Settings):
    return get_supabase_authenticated_client(settings, current_user._access_token)


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
