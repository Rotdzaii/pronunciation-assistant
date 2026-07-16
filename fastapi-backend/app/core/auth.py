from collections.abc import Callable, Sequence
import logging
from time import sleep
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, PrivateAttr
from supabase import Client, create_client

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

AppRole = Literal["student", "teacher", "admin"]
PROFILE_LOOKUP_RETRY_DELAYS = (0.15, 0.3, 0.6)
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


def _signup_app_role(user_metadata: dict[str, Any] | None) -> Literal["student", "teacher"]:
    """Return the only roles public signup is allowed to request.

    Admin is assigned out-of-band. Treating missing, admin, or malformed
    metadata as student makes profile repair safe for existing auth users too.
    """
    if not isinstance(user_metadata, dict):
        return "student"
    requested_role = user_metadata.get("app_role")
    return requested_role if requested_role in ("student", "teacher") else "student"


def _is_network_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, NETWORK_EXCEPTIONS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_profile_app_role(client: Client, user_id: str) -> AppRole | None:
    response = (
        client.table("profiles")
        .select("app_role")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    data = response.data or []
    if not data:
        return None

    app_role = data[0].get("app_role")
    if app_role in ("student", "teacher", "admin"):
        return app_role

    logger.warning("Authenticated user has missing or invalid app_role user_id=%s app_role=%r", user_id, app_role)
    raise _permission_error("User role is not allowed")


def _create_missing_profile(
    client: Client,
    user_id: str,
    email: str | None,
    user_metadata: dict[str, Any] | None,
) -> None:
    """Best-effort insert for a valid auth user without overwriting a profile.

    This complements the database trigger for the short period after signup or
    for legacy users created while the trigger was unavailable. A duplicate
    insert is benign: the caller reads the profile again before returning.
    """
    client.table("profiles").insert({
        "id": user_id,
        "email": email,
        "app_role": _signup_app_role(user_metadata),
    }).execute()


def get_profile_app_role(
    client: Client,
    user_id: str,
    *,
    email: str | None = None,
    user_metadata: dict[str, Any] | None = None,
    client_factory: Callable[[], Client] | None = None,
) -> AppRole:
    last_network_error: BaseException | None = None

    for attempt, delay in enumerate(PROFILE_LOOKUP_RETRY_DELAYS, start=1):
        active_client = client
        if attempt == len(PROFILE_LOOKUP_RETRY_DELAYS) and client_factory is not None:
            active_client = client_factory()

        try:
            logger.debug(
                "DEBUG_PROFILE_QUERY attempt=%s user_id=%r",
                attempt,
                user_id,
            )
            app_role = _load_profile_app_role(active_client, user_id)
            last_network_error = None
            logger.debug(
                "DEBUG_PROFILE_RESPONSE user_id=%r data=%r",
                user_id,
                app_role,
            )
            if app_role is not None:
                return app_role
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
    if last_network_error is not None:
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

    # A valid JWT must not become a 401 solely because the profile trigger is
    # delayed or a legacy profile is missing. Insert-only repair protects any
    # existing admin role from being overwritten.
    repair_client = client_factory() if client_factory is not None else client
    try:
        _create_missing_profile(repair_client, user_id, email, user_metadata)
    except Exception as exc:
        if _is_network_exception(exc):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication profile service temporarily unavailable",
            ) from exc
        logger.info("Profile insert raced or was rejected user_id=%s; rechecking", user_id)

    try:
        repaired_role = _load_profile_app_role(repair_client, user_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unable to verify repaired profile user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load user profile",
        ) from exc

    if repaired_role is not None:
        return repaired_role

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="User profile is being prepared. Please retry shortly.",
    )


def _try_verify_jwt_local(token: str, settings: Settings) -> dict[str, Any] | None:
    """Verify the Supabase JWT locally using SUPABASE_JWT_SECRET (HS256 only).

    Returns a user-data dict on success, or None to signal the caller should
    fall back to the Supabase Auth API. Raises _auth_error only for tokens that
    are structurally valid HS256 but expired or have an invalid signature.
    Non-HS256 tokens (e.g. ES256) return None so the API fallback handles them.
    """
    if not settings.supabase_jwt_secret:
        return None

    try:
        import jwt as pyjwt
    except ImportError:
        return None

    try:
        token_header = pyjwt.get_unverified_header(token)
    except Exception:
        return None

    if token_header.get("alg") != "HS256":
        return None

    try:
        payload = pyjwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except pyjwt.ExpiredSignatureError:
        raise _auth_error("Invalid or expired token")
    except pyjwt.InvalidTokenError:
        raise _auth_error("Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise _auth_error("Invalid or expired token")

    return {
        "id": str(user_id),
        "email": payload.get("email"),
        "role": payload.get("role", "authenticated"),
        "aud": payload.get("aud", "authenticated"),
        "user_metadata": payload.get("user_metadata") or {},
    }


async def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    authorization = request.headers.get("authorization")
    logger.debug(
        "Auth: has_authorization_header=%s",
        bool(authorization),
    )

    if not authorization:
        raise _auth_error("Missing Authorization bearer token")

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        raise _auth_error("Missing Authorization bearer token")

    scheme, token = parts
    token = token.strip()

    logger.debug(
        "Auth: scheme=%s token_length=%s",
        scheme,
        len(token),
    )

    if scheme.lower() != "bearer" or not token:
        raise _auth_error("Missing Authorization bearer token")

    # Local JWT verification (no network call). Falls back to API when
    # SUPABASE_JWT_SECRET is not set or PyJWT is unavailable.
    user_data: dict[str, Any] | None = _try_verify_jwt_local(token, settings)

    if user_data is None:
        try:
            user_data = await _get_supabase_auth_user(token, settings)
        except httpx.HTTPError as exc:
            supabase_host = None
            if settings.supabase_url:
                try:
                    supabase_host = urlparse(settings.supabase_url).hostname
                except Exception:
                    pass
            logger.warning(
                "Unable to verify bearer token: Supabase Auth API unreachable "
                "supabase_host=%s error_type=%s",
                supabase_host,
                type(exc).__name__,
                exc_info=True,
            )
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
        email=user_data.get("email"),
        user_metadata=user_data.get("user_metadata") or {},
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
