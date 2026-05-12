from collections.abc import Callable, Sequence
from typing import Any

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from supabase import Client, create_client

from app.core.config import Settings, get_settings


bearer_scheme = HTTPBearer(auto_error=False)


class CurrentUser(BaseModel):
    id: str
    email: str | None = None
    auth_role: str
    app_role: str | None = None


def _auth_error(detail: str = "Missing or invalid bearer token") -> HTTPException:
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


def _decode_supabase_jwt(token: str, settings: Settings) -> dict[str, Any]:
    if not settings.supabase_jwt_secret:
        return jwt.decode(token, options={"verify_signature": False})

    return jwt.decode(
        token,
        settings.supabase_jwt_secret,
        algorithms=["HS256"],
        audience="authenticated",
    )


async def _verify_token_with_supabase(token: str, settings: Settings) -> None:
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise _settings_error()

    auth_url = settings.supabase_url.rstrip("/") + "/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": settings.supabase_anon_key,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(auth_url, headers=headers)

    if response.status_code != status.HTTP_200_OK:
        raise _auth_error()


def get_supabase_service_client(settings: Settings) -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise _settings_error()

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def get_profile_app_role(client: Client, user_id: str) -> str | None:
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

    return data[0].get("app_role")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _auth_error("Missing bearer token")

    token = credentials.credentials

    try:
        claims = _decode_supabase_jwt(token, settings)
        if not settings.supabase_jwt_secret:
            await _verify_token_with_supabase(token, settings)
    except jwt.PyJWTError as exc:
        raise _auth_error() from exc
    except httpx.HTTPError as exc:
        raise _auth_error("Unable to verify bearer token") from exc

    user_id = claims.get("sub")
    if not user_id:
        raise _auth_error()

    supabase_client = get_supabase_service_client(settings)

    return CurrentUser(
        id=user_id,
        email=claims.get("email"),
        auth_role=claims.get("role", ""),
        app_role=get_profile_app_role(supabase_client, user_id),
    )


def require_roles(roles: Sequence[str]) -> Callable[[CurrentUser], CurrentUser]:
    allowed_roles = set(roles)

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
