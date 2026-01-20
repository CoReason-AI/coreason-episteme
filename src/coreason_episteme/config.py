from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed by pydantic-settings.
    Reads from environment variables (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Logging
    LOG_LEVEL: str = "INFO"

    # Engine
    MAX_RETRIES: int = 3
    GAP_SCANNER_SIMILARITY_THRESHOLD: float = 0.75
    DRUGGABILITY_THRESHOLD: float = 0.5


settings = Settings()
