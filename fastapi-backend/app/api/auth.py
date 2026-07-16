import logging
import re

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.auth import CurrentUser, _get_supabase_auth_user, get_current_user, get_supabase_service_client
from app.core.config import Settings, get_settings


router = APIRouter(prefix="/auth", tags=["Auth"])
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", re.IGNORECASE)
_MIN_PASSWORD_LEN = 6


class ForgotPasswordPayload(BaseModel):
    email: str


class ResetPasswordPayload(BaseModel):
    token: str
    new_password: str


@router.get("/me", response_model=CurrentUser)
def read_current_user(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    return current_user


@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    payload: ForgotPasswordPayload,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Send a password reset email. Always returns 200 to prevent email enumeration."""
    email = payload.email.strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Địa chỉ email không hợp lệ.",
        )

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dịch vụ xác thực chưa được cấu hình.",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/recover",
                headers={
                    "apikey": settings.supabase_anon_key,
                    "Content-Type": "application/json",
                },
                json={"email": email},
            )

        if response.status_code not in (200, 204):
            logger.warning(
                "Supabase password reset request failed status=%s body=%s",
                response.status_code,
                response.text[:300],
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Không thể gửi email đặt lại mật khẩu. Vui lòng thử lại sau.",
            )

    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.warning("Network error sending password reset email", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dịch vụ xác thực tạm thời không khả dụng. Vui lòng thử lại sau.",
        ) from exc

    return {
        "message": "Nếu email tồn tại trong hệ thống, bạn sẽ nhận được link đặt lại mật khẩu trong vài phút."
    }


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    payload: ResetPasswordPayload,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Reset password using the recovery token from the email link."""
    if len(payload.new_password) < _MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mật khẩu phải có ít nhất {_MIN_PASSWORD_LEN} ký tự.",
        )

    token = payload.token.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token đặt lại mật khẩu không được để trống.",
        )

    try:
        user_data = await _get_supabase_auth_user(token, settings)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Link đặt lại mật khẩu không hợp lệ hoặc đã hết hạn. Vui lòng yêu cầu link mới.",
            ) from exc
        raise

    user_id = str(user_data["id"])

    supabase_client = get_supabase_service_client(settings)
    try:
        supabase_client.auth.admin.update_user_by_id(
            user_id,
            {"password": payload.new_password},
        )
    except Exception as exc:
        logger.exception("Unable to reset password for user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể cập nhật mật khẩu. Vui lòng thử lại.",
        ) from exc

    logger.info("Password reset completed for user_id=%s", user_id)
    return {"message": "Mật khẩu đã được cập nhật thành công. Vui lòng đăng nhập lại."}
