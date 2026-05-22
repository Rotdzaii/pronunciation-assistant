from collections.abc import Callable, Sequence
import logging
from typing import Any, Literal

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel
from supabase import Client, create_client

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

AppRole = Literal["student", "teacher", "admin"]


class CurrentUser(BaseModel):
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
        raise _auth_error("Invalid or expired token")

    try:
        user_data = response.json()
    except ValueError as exc:
        raise _auth_error("Invalid or expired token") from exc

    if not isinstance(user_data, dict):
        raise _auth_error("Invalid or expired token")

    if not user_data.get("id"):
        raise _auth_error("Invalid or expired token")

    return user_data


def get_supabase_service_client(settings: Settings) -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise _settings_error()

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_profile_app_role(client: Client, user_id: str) -> AppRole | None:
    try:
        response = (
            client.table("profiles")
            .select("app_role")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load user profile",
        ) from exc

    data = response.data or []
    if not data:
        return None

    app_role = data[0].get("app_role")
    if app_role in ("student", "teacher", "admin"):
        return app_role
    return None


async def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    authorization = request.headers.get("authorization")
    logger.debug("Authorization header exists=%s", bool(authorization))

    if not authorization:
        raise _auth_error("Missing or invalid bearer token")

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        raise _auth_error("Missing or invalid bearer token")

    scheme, token = parts
    token = token.strip()

    logger.debug("Authorization scheme=%s", scheme)

    if scheme.lower() != "bearer" or not token:
        raise _auth_error("Missing or invalid bearer token")

    logger.debug("Bearer token prefix=%s...", token[:8])

    try:
        user_data = await _get_supabase_auth_user(token, settings)
    except httpx.HTTPError as exc:
        raise _auth_error("Unable to verify bearer token") from exc

    user_id = str(user_data["id"])

    auth_role = (
        user_data.get("auth_role")
        or user_data.get("role")
        or user_data.get("aud")
        or "authenticated"
    )

    supabase_client = get_supabase_service_client(settings)

    app_role = get_profile_app_role(supabase_client, user_id)

    return CurrentUser(
        id=user_id,
        email=user_data.get("email"),
        auth_role=auth_role,
        app_role=app_role,
        full_name=(user_data.get("user_metadata") or {}).get("full_name"),
        name=(user_data.get("user_metadata") or {}).get("name"),
        can_use_student=app_role in ("student", "admin"),
        can_use_teacher=app_role in ("teacher", "admin"),
    )


def require_roles(roles: Sequence[str]) -> Callable[[CurrentUser], CurrentUser]:
    allowed_roles = set(roles)
    if allowed_roles.intersection({"student", "teacher"}):
        allowed_roles.add("admin")

    def role_dependency(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.app_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )

        return current_user

    return role_dependency
