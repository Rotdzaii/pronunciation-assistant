import os
import sys
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

    print(f"GET {path} -> {response.status_code}")
    if response.status_code != 200:
        print(f"body={response.text[:800]}")
        return None

    try:
        return response.json()
    except ValueError:
        print(f"[FAIL] GET {path} returned non-JSON body")
        return None


def summarize_classes(payload: Any) -> str | None:
    if not isinstance(payload, list):
        print("[FAIL] /teacher/classes did not return a JSON list")
        return None

    codes = [
        str(item.get("code"))
        for item in payload
        if isinstance(item, dict) and item.get("code") is not None
    ]
    print(f"classes_count={len(payload)}")
    print(f"class_codes={codes}")

    first = next((item for item in payload if isinstance(item, dict) and item.get("id")), None)
    return str(first["id"]) if first else None


def summarize_list(label: str, payload: Any) -> None:
    if isinstance(payload, list):
        print(f"{label}_count={len(payload)}")
        return
    print(f"[FAIL] {label} did not return a JSON list")


def summarize_detail(payload: Any) -> None:
    if not isinstance(payload, dict):
        print("[FAIL] class detail did not return a JSON object")
        return
    print(f"class_detail_code={payload.get('code')}")
    print(f"class_detail_students={len(payload.get('students') or [])}")
    print(f"class_detail_teachers={len(payload.get('teachers') or [])}")


def main() -> int:
    load_local_env()

    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    teacher_token = os.getenv("TEACHER_TOKEN")

    if not teacher_token or not teacher_token.strip():
        print("[FAIL] TEACHER_TOKEN is missing or empty.")
        print("Run:")
        print("python scripts/login_demo_users.py --write-local-env")
        return 2

    print(f"BASE_URL={base_url}")
    print(f"TEACHER_TOKEN length={len(teacher_token.strip())}")

    failed = False
    with httpx.Client(timeout=20.0) as client:
        classes = get_json(client, base_url, "/teacher/classes", teacher_token)
        if classes is None:
            return 1

        class_id = summarize_classes(classes)
        if not class_id:
            print("[WARN] Teacher has no classes; skipping detail endpoints.")
            return 0

        detail = get_json(client, base_url, f"/teacher/classes/{class_id}", teacher_token)
        if detail is None:
            failed = True
        else:
            summarize_detail(detail)

        students = get_json(client, base_url, f"/teacher/classes/{class_id}/students", teacher_token)
        if students is None:
            failed = True
        else:
            summarize_list("students", students)

        teachers = get_json(client, base_url, f"/teacher/classes/{class_id}/teachers", teacher_token)
        if teachers is None:
            failed = True
        else:
            summarize_list("teachers", teachers)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
