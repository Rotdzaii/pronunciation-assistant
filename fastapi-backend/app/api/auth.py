from fastapi import APIRouter, Depends

from app.core.auth import CurrentUser, get_current_user


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/me", response_model=CurrentUser)
def read_current_user(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    return current_user
