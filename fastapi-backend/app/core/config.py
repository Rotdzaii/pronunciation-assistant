from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Pronunciation Assistant API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    backend_cors_origins: str = (
        "http://localhost:8081,"
        "http://localhost:19006,"
        "http://localhost:3000"
    )
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    supabase_service_role_key: str | None = None
    practice_audio_bucket: str = "practice-audios"
    ai_webhook_secret: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.backend_cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
