import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ESTAT_API_KEY: str
    ESTAT_API_VERSION: str = "3.0"
    ESTAT_BASE_URL: str = "https://api.e-stat.go.jp/rest/3.0/app/json"

    model_config = SettingsConfigDict(
        env_file="/workspace/.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()