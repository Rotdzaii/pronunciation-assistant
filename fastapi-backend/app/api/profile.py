from datetime import UTC, datetime
import logging
from pathlib import Path
from time import sleep
from typing import Callable, TypeVar

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user, get_supabase_service_client
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/profile", tags=["Profile"])
logger = logging.getLogger(__name__)

PROFILE_SELECT = "id,email,display_name,app_role,avatar_url,avatar_path,updated_at"
PROFILE_BASE_SELECT = "id,email,display_name,app_role"
AVATAR_BUCKET = "avatars"
ALLOWED_AVATAR_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_AVATAR_BYTES = 2 * 1024 * 1024
PROFILE_RETRY_DELAYS = (0.2, 0.5, 1.0)
T = TypeVar("T")
NETWORK_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.NetworkError,
)


class UserProfile(BaseModel):
    id: str
    email: str | None = None
    display_name: str
    app_role: str | None = None
    avatar_url: str | None = None
    avatar_path: str | None = None
    updated_at: str | None = None


class UpdateProfilePayload(BaseModel):
    display_name: str = Field(min_length=2, max_length=80)


class ProfileAvatarUploadResponse(BaseModel):
    profile: UserProfile
    avatar_url: str
    avatar_path: str


def _is_network_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, NETWORK_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_missing_column_error(exc: BaseException, columns: tuple[str, ...]) -> bool:
    message = str(exc).lower()
    return any(column.lower() in message for column in columns) and (
        "column" in message or "schema cache" in message or "could not find" in message
    )


def _is_missing_bucket_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "bucket" in message and ("not found" in message or "does not exist" in message)


def _run_with_network_retry(
    operation: Callable[[], T],
    *,
    unavailable_detail: str,
    log_label: str,
    user_id: str,
) -> T:
    last_network_error: BaseException | None = None

    for attempt, delay in enumerate(PROFILE_RETRY_DELAYS, start=1):
        try:
            return operation()
        except HTTPException:
            raise
        except Exception as exc:
            if not _is_network_exception(exc):
                raise

            last_network_error = exc
            logger.warning(
                "%s transient failure user_id=%s attempt=%s/%s",
                log_label,
                user_id,
                attempt,
                len(PROFILE_RETRY_DELAYS),
            )
            if attempt < len(PROFILE_RETRY_DELAYS):
                sleep(delay)

    logger.exception(
        "%s unavailable user_id=%s",
        log_label,
        user_id,
        exc_info=(
            type(last_network_error),
            last_network_error,
            last_network_error.__traceback__,
        )
        if last_network_error is not None
        else None,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=unavailable_detail,
    ) from last_network_error


def _display_name(profile: dict) -> str:
    value = profile.get("display_name")
    if isinstance(value, str) and value.strip():
        return value.strip()

    email = profile.get("email")
    if isinstance(email, str) and email.strip():
        return email.split("@", 1)[0]

    return "Phoenix User"


def _profile_response(profile: dict) -> UserProfile:
    return UserProfile(
        id=str(profile["id"]),
        email=profile.get("email"),
        display_name=_display_name(profile),
        app_role=profile.get("app_role"),
        avatar_url=profile.get("avatar_url"),
        avatar_path=profile.get("avatar_path"),
        updated_at=profile.get("updated_at"),
    )


def _query_profile_rows(settings: Settings, user_id: str, select_columns: str) -> list[dict]:
    def query() -> list[dict]:
        supabase_client = get_supabase_service_client(settings)
        response = (
            supabase_client.table("profiles")
            .select(select_columns)
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        return response.data or []

    return _run_with_network_retry(
        query,
        unavailable_detail="Profile service temporarily unavailable",
        log_label="Profile lookup",
        user_id=user_id,
    )


def _load_profile(settings: Settings, user_id: str) -> UserProfile:
    try:
        data = _query_profile_rows(settings, user_id, PROFILE_SELECT)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise

        logger.warning(
            "Profile lookup with optional avatar columns failed user_id=%s; retrying with base columns",
            user_id,
            exc_info=True,
        )
        try:
            data = _query_profile_rows(settings, user_id, PROFILE_BASE_SELECT)
        except Exception as fallback_exc:
            if isinstance(fallback_exc, HTTPException):
                raise
            logger.exception("Unable to load profile user_id=%s", user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to load profile",
            ) from fallback_exc

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        )
    return _profile_response(data[0])


def _public_avatar_url(settings: Settings, avatar_path: str) -> str:
    if not settings.supabase_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase storage is not configured",
        )
    return f"{settings.supabase_url.rstrip('/')}/storage/v1/object/public/{AVATAR_BUCKET}/{avatar_path}"


def _upload_avatar_to_storage(
    settings: Settings,
    user_id: str,
    avatar_path: str,
    content: bytes,
    content_type: str,
) -> None:
    def upload() -> None:
        supabase_client = get_supabase_service_client(settings)
        bucket = supabase_client.storage.from_(AVATAR_BUCKET)
        try:
            bucket.upload(
                path=avatar_path,
                file=content,
                file_options={
                    "content-type": content_type,
                    "upsert": True,
                },
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "duplicate" not in message:
                raise
            bucket.update(
                path=avatar_path,
                file=content,
                file_options={"content-type": content_type},
            )

    try:
        _run_with_network_retry(
            upload,
            unavailable_detail="Avatar storage temporarily unavailable",
            log_label="Avatar storage upload",
            user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        if _is_missing_bucket_error(exc):
            logger.exception("Avatar storage bucket is missing bucket=%s user_id=%s", AVATAR_BUCKET, user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Avatar storage bucket is not configured",
            ) from exc
        logger.exception("Unable to upload avatar to storage user_id=%s path=%s", user_id, avatar_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to upload avatar to storage",
        ) from exc


def _update_profile_avatar(
    settings: Settings,
    user_id: str,
    avatar_url: str,
    avatar_path: str,
) -> None:
    def update_avatar() -> None:
        supabase_client = get_supabase_service_client(settings)
        response = (
            supabase_client.table("profiles")
            .update(
                {
                    "avatar_url": avatar_url,
                    "avatar_path": avatar_path,
                    "updated_at": datetime.now(tz=UTC).isoformat(),
                }
            )
            .eq("id", user_id)
            .execute()
        )
        if not (response.data or []):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found",
            )

    try:
        _run_with_network_retry(
            update_avatar,
            unavailable_detail="Profile service temporarily unavailable",
            log_label="Profile avatar update",
            user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        if _is_missing_column_error(exc, ("avatar_url", "avatar_path", "updated_at")):
            logger.exception("Profile avatar columns are missing user_id=%s", user_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Profile avatar columns are not available. Run the avatar profile migration.",
            ) from exc
        logger.exception("Unable to update profile avatar user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update profile avatar",
        ) from exc


@router.get("/me", response_model=UserProfile)
def read_my_profile(
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> UserProfile:
    return _load_profile(settings, current_user.id)


@router.patch("/me", response_model=UserProfile)
def update_my_profile(
    payload: UpdateProfilePayload,
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> UserProfile:
    display_name = payload.display_name.strip()
    if len(display_name) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Display name must be at least 2 characters",
        )
    if len(display_name) > 80:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Display name must be 80 characters or fewer",
        )

    supabase_client = get_supabase_service_client(settings)
    try:
        response = (
            supabase_client.table("profiles")
            .update(
                {
                    "display_name": display_name,
                    "updated_at": datetime.now(tz=UTC).isoformat(),
                }
            )
            .eq("id", current_user.id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update profile",
        ) from exc

    if not (response.data or []):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        )
    return _load_profile(settings, current_user.id)


@router.post("/avatar", response_model=ProfileAvatarUploadResponse)
async def upload_my_avatar(
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> ProfileAvatarUploadResponse:
    content_type = file.content_type or ""
    extension = ALLOWED_AVATAR_TYPES.get(content_type)
    if extension is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Avatar must be a JPEG, PNG, or WebP image",
        )

    content = await file.read()
    if len(content) > MAX_AVATAR_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Avatar image must be 2MB or smaller",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Avatar image is empty",
        )

    original_ext = Path(file.filename or "").suffix.lower()
    if original_ext in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg" if original_ext == ".jpeg" else original_ext
    avatar_path = f"{current_user.id}/avatar{extension}"
    avatar_url = _public_avatar_url(settings, avatar_path)

    _upload_avatar_to_storage(settings, current_user.id, avatar_path, content, content_type)
    _update_profile_avatar(settings, current_user.id, avatar_url, avatar_path)

    profile = _load_profile(settings, current_user.id)
    return ProfileAvatarUploadResponse(
        profile=profile,
        avatar_url=profile.avatar_url or avatar_url,
        avatar_path=profile.avatar_path or avatar_path,
    )
