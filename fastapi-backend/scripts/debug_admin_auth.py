import os
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parents[1]
TOKEN_ENV_PATH = BASE_DIR / ".tokens.local.env"


def load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")
    load_local_env()


def load_local_env(path: Path = TOKEN_ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise SystemExit(f"Missing env: {' or '.join(names)}")


def get_access_token(response: Any) -> str | None:
    session = getattr(response, "session", None)
    if session is not None:
        return getattr(session, "access_token", None)

    if isinstance(response, dict):
        session_dict = response.get("session") or {}
        token = session_dict.get("access_token")
        if isinstance(token, str):
            return token

    return None


def get_user_id(response: Any) -> str | None:
    user = getattr(response, "user", None)
    if user is not None:
        user_id = getattr(user, "id", None)
        return str(user_id) if user_id else None

    if isinstance(response, dict):
        user_dict = response.get("user") or {}
        user_id = user_dict.get("id")
        return str(user_id) if user_id else None

    return None


def get_user_email(response: Any) -> str | None:
    user = getattr(response, "user", None)
    if user is not None:
        return getattr(user, "email", None)

    if isinstance(response, dict):
        user_dict = response.get("user") or {}
        email = user_dict.get("email")
        if isinstance(email, str):
            return email

    return None


def print_profile_sync_sql(user_id: str, admin_email: str) -> None:
    print("Suggested SQL after verifying this is the intended admin account:")
    print(
        "insert into public.profiles (id, email, app_role, display_name) "
        f"values ('{user_id}', '{admin_email}', 'admin', 'Admin') "
        "on conflict (id) do update set app_role = 'admin', email = excluded.email;"
    )


def print_profile_by_email(supabase_url: str, service_role_key: str, admin_email: str) -> None:
    service_client = create_client(supabase_url, service_role_key)
    try:
        profile_rows = (
            service_client.table("profiles")
            .select("id,email,app_role,display_name")
            .eq("email", admin_email)
            .limit(5)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[FAIL] Unable to query public.profiles by email -> {exc}")
        return

    if not profile_rows:
        print(f"[FAIL] profile_exists_for_email=False email={admin_email}")
        return

    print(f"profile_rows_for_email={len(profile_rows)}")
    for profile in profile_rows:
        print(
            "profile_by_email="
            f"id={profile.get('id')} email={profile.get('email')} app_role={profile.get('app_role')}"
        )


def get_user_from_token(supabase_url: str, supabase_anon_key: str, token: str) -> dict[str, Any]:
    response = httpx.get(
        supabase_url.rstrip("/") + "/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": supabase_anon_key,
        },
        timeout=10.0,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Supabase Auth rejected ADMIN_TOKEN with status {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("id"):
        raise RuntimeError("Supabase Auth returned a user payload without id")
    return payload


def describe_token_expiry(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return "JWT payload is not readable."
        payload_segment = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_segment.encode("ascii")))
        exp = payload.get("exp")
        if not isinstance(exp, int):
            return "JWT exp is missing."
        expires_at = datetime.fromtimestamp(exp, tz=UTC)
        now = datetime.now(tz=UTC)
        state = "expired" if expires_at <= now else "expires"
        return f"JWT {state}_at={expires_at.isoformat()}."
    except Exception:
        return "JWT payload could not be decoded."


def main() -> None:
    load_env_file()

    supabase_url = get_required_env("SUPABASE_URL")
    supabase_anon_key = get_required_env(
        "SUPABASE_ANON_KEY",
        "SUPABASE_KEY",
        "SUPABASE_PUBLIC_ANON_KEY",
    )
    service_role_key = get_required_env("SUPABASE_SERVICE_ROLE_KEY")
    admin_email = os.getenv("DEMO_ADMIN_EMAIL", "admin@gmail.com")
    admin_password = os.getenv("DEMO_ADMIN_PASSWORD")
    admin_token = os.getenv("ADMIN_TOKEN")

    print(f"[DEBUG] Supabase URL: {supabase_url}")

    if admin_password:
        print(
            "[DEBUG] Admin login target: "
            f"email='{admin_email}', password_set=True, password_len={len(admin_password)}"
        )
        anon_client = create_client(supabase_url, supabase_anon_key)

        try:
            response = anon_client.auth.sign_in_with_password(
                {
                    "email": admin_email,
                    "password": admin_password,
                }
            )
        except Exception as exc:
            print(f"[FAIL] Admin login failed: {admin_email} -> {exc}")
            print("auth_user_exists=unknown")
            raise SystemExit(1)

        token = get_access_token(response)
        user_id = get_user_id(response)
        user_email = get_user_email(response)
        print(f"[PASS] Admin login succeeded: {admin_email}")
        print(f"auth_user_exists={bool(user_id)}")
        print(f"user_id={user_id or 'missing'}")
        print(f"user_email={user_email}")
        print(f"has_session={bool(token)}")
        print(f"access_token_len={len(token) if token else 0}")
    elif admin_token:
        print("[DEBUG] DEMO_ADMIN_PASSWORD is missing; using ADMIN_TOKEN from environment/local token file.")
        print(f"admin_token_len={len(admin_token)}")
        print(describe_token_expiry(admin_token))
        try:
            user_data = get_user_from_token(supabase_url, supabase_anon_key, admin_token)
        except Exception as exc:
            print(f"[FAIL] Unable to verify ADMIN_TOKEN -> {exc}")
            print_profile_by_email(supabase_url, service_role_key, admin_email)
            raise SystemExit(1)
        user_id = str(user_data.get("id"))
        user_email = user_data.get("email")
        print("[PASS] ADMIN_TOKEN verified by Supabase Auth")
        print(f"auth_user_exists={bool(user_id)}")
        print(f"user_id={user_id or 'missing'}")
        print(f"user_email={user_email}")
    else:
        print("[FAIL] Missing DEMO_ADMIN_PASSWORD and ADMIN_TOKEN.")
        print("Run: python scripts/login_demo_users.py --write-local-env")
        raise SystemExit(2)

    if not user_id:
        print("[FAIL] Login response did not include an auth user id.")
        raise SystemExit(1)

    service_client = create_client(supabase_url, service_role_key)
    try:
        profile_rows = (
            service_client.table("profiles")
            .select("id,email,app_role,display_name")
            .eq("id", user_id)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[FAIL] Unable to query public.profiles for admin user -> {exc}")
        raise SystemExit(1)

    if not profile_rows:
        print("[FAIL] profile_exists=False")
        print_profile_sync_sql(user_id, admin_email)
        raise SystemExit(1)

    profile = profile_rows[0]
    app_role = profile.get("app_role")
    print("[PASS] profile_exists=True")
    print(f"profile_email={profile.get('email')}")
    print(f"profile_app_role={app_role}")

    if app_role != "admin":
        print("[FAIL] profile_app_role is not admin")
        print_profile_sync_sql(user_id, admin_email)
        raise SystemExit(1)

    print("[PASS] admin profile is ready: app_role=admin")


if __name__ == "__main__":
    main()
