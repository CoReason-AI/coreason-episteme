# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

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

    # External Services
    GRAPH_NEXUS_URL: str = "http://coreason-graph-nexus:8000"
    CODEX_URL: str = "http://coreason-codex:8000"
    SEARCH_URL: str = "http://coreason-search:8000"
    PRISM_URL: str = "http://coreason-prism:8000"
    INFERENCE_URL: str = "http://coreason-inference:8000"
    VERITAS_URL: str = "http://coreason-veritas:8000"


settings = Settings()
