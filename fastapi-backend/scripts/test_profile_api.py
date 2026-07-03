import os
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


def request_profile(client: httpx.Client, base_url: str, label: str, token: str) -> dict[str, Any] | None:
    try:
        response = client.get(
            f"{base_url}/profile/me",
            headers={"Authorization": f"Bearer {token.strip()}"},
        )
    except httpx.ConnectError as exc:
        print(f"[FAIL] {label} GET /profile/me could not connect to {base_url}: {exc}")
        print("Start the backend in another terminal:")
        print("uvicorn app.main:app --reload")
        return None
    except httpx.HTTPError as exc:
        print(f"[FAIL] {label} GET /profile/me request failed: {exc}")
        return None

    print(f"{label} GET /profile/me -> {response.status_code}")
    if response.status_code != 200:
        print(f"body={response.text[:800]}")
        return None

    try:
        payload = response.json()
    except ValueError:
        print(f"[FAIL] {label} GET /profile/me returned non-JSON body")
        return None

    if not isinstance(payload, dict):
        print(f"[FAIL] {label} GET /profile/me did not return a JSON object")
        return None

    print(f"{label} email={payload.get('email')}")
    print(f"{label} display_name={payload.get('display_name')}")
    print(f"{label} app_role={payload.get('app_role')}")
    print(f"{label} has_avatar_url={bool(payload.get('avatar_url'))}")
    print(f"{label} has_avatar_path={bool(payload.get('avatar_path'))}")
    return payload


def main() -> int:
    load_local_env()

    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    token_specs = (
        ("student", os.getenv("STUDENT_TOKEN")),
        ("teacher", os.getenv("TEACHER_TOKEN")),
        ("admin", os.getenv("ADMIN_TOKEN")),
    )

    available_tokens = [(label, token) for label, token in token_specs if token and token.strip()]
    if not available_tokens:
        print("[FAIL] No profile API tokens found.")
        print("Run:")
        print("python scripts/login_demo_users.py --write-local-env")
        return 2

    print(f"BASE_URL={base_url}")
    failed = False
    with httpx.Client(timeout=20.0) as client:
        for label, token in token_specs:
            if not token or not token.strip():
                print(f"[SKIP] {label} token is missing.")
                continue
            if request_profile(client, base_url, label, token) is None:
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
