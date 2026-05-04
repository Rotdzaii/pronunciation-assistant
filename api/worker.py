import os

import httpx
from dotenv import load_dotenv
from supabase import create_client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
NODE_WEBHOOK_URL = os.getenv("NODE_WEBHOOK_URL")
AI_WEBHOOK_SECRET = os.getenv("AI_WEBHOOK_SECRET")


if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL in .env")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY in .env")

if not NODE_WEBHOOK_URL:
    raise RuntimeError("Missing NODE_WEBHOOK_URL in .env")

if not AI_WEBHOOK_SECRET:
    raise RuntimeError("Missing AI_WEBHOOK_SECRET in .env")


supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def read_one_job():
    response = supabase.rpc("read_practice_job").execute()

    if not response.data:
        return None

    return response.data[0]


def send_mock_result_to_node(job_payload: dict):
    job_id = job_payload["job_id"]

    # TODO: Replace this mock payload with real AI inference result.
    # Current values are fixed only for testing queue -> webhook flow.
    webhook_payload = {
        "job_id": job_id,
        "status": "completed",
        "score": 85,
        "problem_phonemes": ["/k/", "/ju:/"],
    }

    headers = {
        "x-ai-webhook-secret": AI_WEBHOOK_SECRET,
    }

    with httpx.Client(timeout=10) as client:
        response = client.post(
            NODE_WEBHOOK_URL,
            json=webhook_payload,
            headers=headers,
        )
        response.raise_for_status()

    return response.json()


def archive_job_message(msg_id: int):
    response = supabase.rpc(
        "archive_practice_job",
        {"p_msg_id": msg_id},
    ).execute()

    return response.data


def run_once():
    job = read_one_job()

    if not job:
        print("No job found in practice_jobs queue.")
        return

    msg_id = job["msg_id"]
    payload = job["message"]

    print(f"Processing msg_id={msg_id}, job_id={payload.get('job_id')}")

    send_mock_result_to_node(payload)

    archived = archive_job_message(msg_id)

    print(f"Done. Archived msg_id={msg_id}, result={archived}")


if __name__ == "__main__":
    run_once()
