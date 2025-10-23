import json

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Optional
from pathlib import Path


def get_project_root() -> Path:
    """
    Gets the project root directory
    """
    return Path(__file__).parent.parent

PROJECT_ROOT = get_project_root()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    environment: str
    debug: bool
    log_level: str
    deepl_api_key: str
    exchange_rate_api_key: str
    google_cloud_project: str
    google_application_credentials: Optional[str] = None
    google_application_credentials_json: Optional[str] = None
    gemini_api_key: str
    google_custom_search_api_key: str
    google_custom_search_engine_id: str
    database_url: str
    cors_origins: str
    max_image_size_mb: int

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env") ,
        case_sensitive=False
    )

    @property
    def credentials_path(self):
        """
        Get absolute path to Google Cloud credentials file
        """
        cred_path = Path(self.google_application_credentials)

        if cred_path.is_absolute():
            return cred_path

        return PROJECT_ROOT / cred_path

    @property
    def database_path(self):
        """
        Get absolute path to database file
        """
        db_file = self.database_url.replace("sqlite:///", "")
        if db_file.startswith("./"):
            db_file = db_file[2:]

        db_path = Path(db_file)

        if not db_path.is_absolute():
            return PROJECT_ROOT / db_path

        return db_path

    @property
    def cors_origins_list(self) -> List[str]:
        """Parses CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()