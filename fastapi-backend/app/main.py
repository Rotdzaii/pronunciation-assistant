from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_feature_router
from app.api.assignments import router as assignments_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.practice import router as practice_router
from app.api.profile import router as profile_router
from app.api.student import router as student_feature_router
from app.api.student_review import router as student_review_router
from app.api.teacher import router as teacher_router
from app.api.vocabulary import router as vocabulary_router
from app.api.class_management import (
    admin_router,
    student_router,
    teacher_router as teacher_classes_router,
)
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(profile_router)
app.include_router(practice_router)

# Teacher dashboard / analytics endpoints
app.include_router(teacher_router)

# Student review endpoints
app.include_router(student_review_router)

# Class management endpoints for student, teacher, and admin
app.include_router(student_router)
app.include_router(teacher_classes_router)
app.include_router(admin_router)

app.include_router(vocabulary_router)
app.include_router(assignments_router)

# Extended student feature endpoints
app.include_router(student_feature_router)

# Extended admin feature endpoints
app.include_router(admin_feature_router)