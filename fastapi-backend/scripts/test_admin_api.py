import os
import sys
import base64
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


BASE_DIR = Path(__file__).resolve().parents[1]
TOKEN_ENV_PATH = BASE_DIR / ".tokens.local.env"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


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


def token_diagnostics(token: str | None) -> str:
    if token is None:
        return "ADMIN_TOKEN is missing."
    if not token.strip():
        return "ADMIN_TOKEN is empty."
    return (
        "ADMIN_TOKEN is present, "
        f"length={len(token.strip())}, "
        f"{describe_token_expiry(token.strip())} "
        "Authorization header format=Bearer <token>. "
        "If this still returns 401, the token may be expired; run "
        "python scripts/login_demo_users.py --write-local-env again."
    )


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


def print_response_failure(path: str, response: httpx.Response, token: str | None) -> None:
    print(f"[FAIL] GET {path} -> {response.status_code}")
    print(f"body={response.text[:500]}")
    if response.status_code == 401:
        print("[DIAG] 401 Unauthorized diagnostics:")
        print(f"- {token_diagnostics(token)}")


def get_json(client: httpx.Client, base_url: str, path: str, token: str) -> Any | None:
    try:
        response = client.get(
            f"{base_url}{path}",
            headers={"Authorization": f"Bearer {token.strip()}"},
        )
    except httpx.ConnectError as exc:
        print(f"[FAIL] GET {path} could not connect to {base_url}: {exc}")
        print("Start the backend in another terminal:")
        print("uvicorn app.main:app --reload")
        return None
    except httpx.HTTPError as exc:
        print(f"[FAIL] GET {path} request failed: {exc}")
        return None

    if response.status_code != 200:
        print_response_failure(path, response, token)
        return None

    print(f"[PASS] GET {path} -> 200")
    try:
        return response.json()
    except ValueError:
        print(f"[FAIL] GET {path} returned non-JSON body")
        return None


def summarize_users(payload: Any) -> None:
    if not isinstance(payload, list):
        print("[FAIL] /admin/users did not return a JSON list")
        return

    role_counts = Counter(
        item.get("app_role") or "missing"
        for item in payload
        if isinstance(item, dict)
    )
    admin_profiles = [
        item
        for item in payload
        if isinstance(item, dict) and str(item.get("email") or "").lower() == "admin@gmail.com"
    ]

    print(f"users_count={len(payload)}")
    print(f"role_counts={dict(sorted(role_counts.items()))}")
    if not admin_profiles:
        print("[WARN] admin@gmail.com profile not found in /admin/users response")
        print("Suggested SQL if the auth user id is known:")
        print(
            "insert into public.profiles (id, email, app_role, display_name) "
            "values ('<auth-user-id>', 'admin@gmail.com', 'admin', 'Admin') "
            "on conflict (id) do update set app_role = 'admin', email = excluded.email;"
        )
        return

    admin_role = admin_profiles[0].get("app_role")
    print(f"admin_profile_app_role={admin_role}")
    if admin_role != "admin":
        print("[WARN] admin@gmail.com profile exists but app_role is not admin")
        print("Suggested SQL after verifying the target user:")
        print("update public.profiles set app_role = 'admin' where email = 'admin@gmail.com';")


def summarize_classes(payload: Any) -> None:
    if not isinstance(payload, list):
        print("[FAIL] /admin/classes did not return a JSON list")
        return

    codes = {
        str(item.get("code"))
        for item in payload
        if isinstance(item, dict) and item.get("code") is not None
    }
    expected_codes = {"PHOENIX-A", "PHOENIX-B", "PHOENIX-C"}
    missing = sorted(expected_codes - codes)

    print(f"classes_count={len(payload)}")
    print(f"has_phoenix_a_b_c={not missing}")
    if missing:
        print(f"missing_demo_classes={missing}")


def summarize_readiness(payload: Any) -> None:
    if not isinstance(payload, dict):
        print("[FAIL] /admin/demo/readiness did not return a JSON object")
        return

    checks = payload.get("checks")
    print(f"demo_readiness_passed={payload.get('passed')}")
    print(f"demo_readiness_checks={len(checks) if isinstance(checks, list) else 'n/a'}")


def main() -> int:
    load_local_env()

    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    admin_token = os.getenv("ADMIN_TOKEN")

    if not admin_token or not admin_token.strip():
        print("[FAIL] ADMIN_TOKEN is missing or empty.")
        print("Run:")
        print("python scripts/login_demo_users.py --write-local-env")
        return 2

    print(f"BASE_URL={base_url}")
    print(f"ADMIN_TOKEN length={len(admin_token.strip())}")
    print(describe_token_expiry(admin_token.strip()))

    with httpx.Client(timeout=20.0) as client:
        users = get_json(client, base_url, "/admin/users", admin_token)
        if users is not None:
            summarize_users(users)

        classes = get_json(client, base_url, "/admin/classes", admin_token)
        if classes is not None:
            summarize_classes(classes)

        readiness = get_json(client, base_url, "/admin/demo/readiness", admin_token)
        if readiness is not None:
            summarize_readiness(readiness)

    if users is None or classes is None or readiness is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
