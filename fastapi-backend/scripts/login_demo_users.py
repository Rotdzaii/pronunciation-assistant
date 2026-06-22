import argparse
import os
from getpass import getpass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from supabase import create_client


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_USERS = {
    "student": "hocsinh07@phoenix.edu.vn",
    "teacher": "giangvien03@phoenix.edu.vn",
    "admin": "admin@gmail.com",
}


def load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")


def get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = " or ".join(names)
    raise SystemExit(f"Missing env: {joined}")


def extract_access_token(response: Any) -> str:
    session = getattr(response, "session", None)
    if session is not None:
        token = getattr(session, "access_token", None)
        if token:
            return token

    if isinstance(response, dict):
        session_dict = response.get("session") or {}
        token = session_dict.get("access_token")
        if token:
            return token

    raise RuntimeError("Login succeeded but access_token was not found in response.")


def login_token(supabase, email: str, password: str) -> str:
    response = supabase.auth.sign_in_with_password(
        {
            "email": email,
            "password": password,
        }
    )
    return extract_access_token(response)


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Login demo users and optionally write local API smoke-test tokens."
    )
    parser.add_argument("--student-email", default=os.getenv("DEMO_STUDENT_EMAIL", DEFAULT_USERS["student"]))
    parser.add_argument("--teacher-email", default=os.getenv("DEMO_TEACHER_EMAIL", DEFAULT_USERS["teacher"]))
    parser.add_argument("--admin-email", default=os.getenv("DEMO_ADMIN_EMAIL", DEFAULT_USERS["admin"]))
    parser.add_argument("--password", default=os.getenv("DEMO_PASSWORD"))
    parser.add_argument("--student-password", default=os.getenv("DEMO_STUDENT_PASSWORD"))
    parser.add_argument("--teacher-password", default=os.getenv("DEMO_TEACHER_PASSWORD"))
    parser.add_argument("--admin-password", default=os.getenv("DEMO_ADMIN_PASSWORD"))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--write-local-env",
        action="store_true",
        help="Write tokens to .tokens.local.env. Do not commit this file.",
    )

    args = parser.parse_args()

    supabase_url = get_required_env("SUPABASE_URL")
    supabase_anon_key = get_required_env(
        "SUPABASE_ANON_KEY",
        "SUPABASE_KEY",
        "SUPABASE_PUBLIC_ANON_KEY",
    )

    common_password = args.password
    student_password = args.student_password or common_password
    teacher_password = args.teacher_password or common_password
    admin_password = args.admin_password or common_password

    if not student_password:
        student_password = getpass("Student password: ")

    if not teacher_password:
        teacher_password = getpass("Teacher password: ")

    if not admin_password:
        admin_password = getpass("Admin password: ")

    supabase = create_client(supabase_url, supabase_anon_key)

    accounts = {
        "STUDENT_TOKEN": {
            "email": args.student_email,
            "password": student_password,
        },
        "TEACHER_TOKEN": {
            "email": args.teacher_email,
            "password": teacher_password,
        },
        "ADMIN_TOKEN": {
            "email": args.admin_email,
            "password": admin_password,
        },
    }

    print(f"[DEBUG] Supabase URL: {supabase_url}")
    print("[DEBUG] Login targets:")
    for env_name, account in accounts.items():
        account_password = account["password"]
        print(
            f"- {env_name}: email='{account['email']}', "
            f"password_set={bool(account_password)}, password_len={len(account_password)}"
        )

    tokens: dict[str, str] = {}

    for env_name, account in accounts.items():
        email = account["email"]
        password = account["password"]

        try:
            token = login_token(supabase, email, password)
            tokens[env_name] = token
            print(f"[PASS] Logged in: {email}")
        except Exception as exc:
            print(f"[FAIL] Login failed: {email} -> {exc}")
            raise SystemExit(1)

    if args.write_local_env:
        output_path = BASE_DIR / ".tokens.local.env"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"BASE_URL={args.base_url}\n")
            for env_name, token in tokens.items():
                f.write(f"{env_name}={token}\n")
        print(f"\n[PASS] Tokens written to: {output_path}")
        print("[WARN] Do not commit this file.")
    else:
        print("\n# PowerShell commands:")
        print(f'$env:BASE_URL="{args.base_url}"')
        for env_name, token in tokens.items():
            print(f'$env:{env_name}="{token}"')

    print("\n# Then run:")
    print(r"python scripts\test_admin_api.py")
    print(r"python scripts\smoke_test_class_api.py")


if __name__ == "__main__":
    main()
