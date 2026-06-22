from collections.abc import Callable, Sequence
import logging
from time import sleep
from typing import Any, Literal

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, PrivateAttr
from supabase import Client, create_client

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

AppRole = Literal["student", "teacher", "admin"]
PROFILE_LOOKUP_RETRY_DELAYS = (0.2, 0.5, 1.0)
NETWORK_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.NetworkError,
)


class CurrentUser(BaseModel):
    _access_token: str = PrivateAttr(default="")

    id: str
    email: str | None = None
    auth_role: str | None = None
    app_role: AppRole | None = None
    full_name: str | None = None
    name: str | None = None
    can_use_student: bool = False
    can_use_teacher: bool = False


def _auth_error(detail: str = "Invalid or expired token") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _permission_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def _settings_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Supabase authentication is not configured",
    )


async def _get_supabase_auth_user(token: str, settings: Settings) -> dict[str, Any]:
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise _settings_error()

    auth_url = settings.supabase_url.rstrip("/") + "/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": settings.supabase_anon_key,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(auth_url, headers=headers)

    logger.debug("Supabase Auth /auth/v1/user status=%s", response.status_code)

    if response.status_code != status.HTTP_200_OK:
        logger.warning(
            "Supabase token verification failed status=%s body=%s",
            response.status_code,
            response.text[:300],
        )
        raise _auth_error("Invalid or expired token")

    try:
        user_data = response.json()
    except ValueError as exc:
        logger.warning("Supabase token verification returned non-JSON response", exc_info=True)
        raise _auth_error("Invalid or expired token") from exc

    if not isinstance(user_data, dict):
        logger.warning("Supabase token verification returned unexpected payload type=%s", type(user_data).__name__)
        raise _auth_error("Invalid or expired token")

    if not user_data.get("id"):
        logger.warning("Supabase token verification returned payload without user id")
        raise _auth_error("Invalid or expired token")

    return user_data


def get_supabase_service_client(settings: Settings) -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise _settings_error()

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_supabase_authenticated_client(settings: Settings, token: str) -> Client:
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise _settings_error()

    client = create_client(settings.supabase_url, settings.supabase_anon_key)
    client.postgrest.auth(token)
    return client


def _is_network_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, NETWORK_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__
    return False


def get_profile_app_role(
    client: Client,
    user_id: str,
    client_factory: Callable[[], Client] | None = None,
) -> AppRole | None:
    last_network_error: BaseException | None = None

    for attempt, delay in enumerate(PROFILE_LOOKUP_RETRY_DELAYS, start=1):
        active_client = client
        if attempt == len(PROFILE_LOOKUP_RETRY_DELAYS) and client_factory is not None:
            active_client = client_factory()

        try:
            response = (
                active_client.table("profiles")
                .select("app_role")
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            break
        except HTTPException:
            raise
        except Exception as exc:
            if not _is_network_exception(exc):
                logger.exception("Unable to load profile role for authenticated user_id=%s", user_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to load user profile",
                ) from exc

            last_network_error = exc
            logger.warning(
                "Profile role lookup transient failure user_id=%s attempt=%s/%s",
                user_id,
                attempt,
                len(PROFILE_LOOKUP_RETRY_DELAYS),
            )
            if attempt < len(PROFILE_LOOKUP_RETRY_DELAYS):
                sleep(delay)
    else:
        logger.error(
            "Authentication profile service temporarily unavailable user_id=%s",
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
            detail="Authentication profile service temporarily unavailable",
        ) from last_network_error

    data = response.data or []
    if not data:
        logger.warning("Authenticated user has no profile row user_id=%s", user_id)
        raise _auth_error("User profile not found")

    app_role = data[0].get("app_role")
    if app_role in ("student", "teacher", "admin"):
        return app_role

    logger.warning("Authenticated user has missing or invalid app_role user_id=%s app_role=%r", user_id, app_role)
    raise _permission_error("User role is not allowed")


async def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    authorization = request.headers.get("authorization")
    logger.debug("Authorization header exists=%s", bool(authorization))

    if not authorization:
        raise _auth_error("Missing Authorization bearer token")

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        raise _auth_error("Missing Authorization bearer token")

    scheme, token = parts
    token = token.strip()

    logger.debug("Authorization scheme=%s", scheme)

    if scheme.lower() != "bearer" or not token:
        raise _auth_error("Missing Authorization bearer token")

    try:
        user_data = await _get_supabase_auth_user(token, settings)
    except httpx.HTTPError as exc:
        logger.warning("Unable to verify bearer token with Supabase Auth", exc_info=True)
        raise _auth_error("Unable to verify bearer token") from exc

    user_id = str(user_data["id"])

    auth_role = (
        user_data.get("auth_role")
        or user_data.get("role")
        or user_data.get("aud")
        or "authenticated"
    )

    supabase_client = get_supabase_service_client(settings)

    app_role = get_profile_app_role(
        supabase_client,
        user_id,
        client_factory=lambda: get_supabase_service_client(settings),
    )

    current_user = CurrentUser(
        id=user_id,
        email=user_data.get("email"),
        auth_role=auth_role,
        app_role=app_role,
        full_name=(user_data.get("user_metadata") or {}).get("full_name"),
        name=(user_data.get("user_metadata") or {}).get("name"),
        can_use_student=app_role in ("student", "admin"),
        can_use_teacher=app_role in ("teacher", "admin"),
    )
    current_user._access_token = token
    return current_user


def require_roles(roles: Sequence[str]) -> Callable[[CurrentUser], CurrentUser]:
    allowed_roles = set(roles)
    if allowed_roles.intersection({"student", "teacher"}):
        allowed_roles.add("admin")

    def role_dependency(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.app_role not in allowed_roles:
            logger.warning(
                "Role denied user_id=%s app_role=%r allowed_roles=%s",
                current_user.id,
                current_user.app_role,
                sorted(allowed_roles),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role: requires one of {sorted(allowed_roles)}",
            )

        return current_user

    return role_dependency
