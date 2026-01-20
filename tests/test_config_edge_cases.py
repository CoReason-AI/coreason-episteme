# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from coreason_episteme.config import Settings


def test_settings_defaults() -> None:
    """Test that settings load with correct defaults when no env vars are present."""
    # Ensure no env vars interfere
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.LOG_LEVEL == "INFO"
        assert settings.MAX_RETRIES == 3
        assert settings.GAP_SCANNER_SIMILARITY_THRESHOLD == 0.75
        assert settings.DRUGGABILITY_THRESHOLD == 0.5


def test_settings_override_via_env() -> None:
    """Test that environment variables correctly override defaults."""
    env_vars = {
        "LOG_LEVEL": "DEBUG",
        "MAX_RETRIES": "5",
        "GAP_SCANNER_SIMILARITY_THRESHOLD": "0.8",
        "DRUGGABILITY_THRESHOLD": "0.1",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.MAX_RETRIES == 5
        assert settings.GAP_SCANNER_SIMILARITY_THRESHOLD == 0.8
        assert settings.DRUGGABILITY_THRESHOLD == 0.1


def test_settings_invalid_types() -> None:
    """Test that validation fails for invalid types."""
    env_vars = {
        "MAX_RETRIES": "not_an_int",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValidationError):
            Settings()


def test_settings_case_insensitivity() -> None:
    """Test that environment variables are case-insensitive."""
    env_vars = {
        "max_retries": "10",
        "log_level": "WARNING",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        settings = Settings()
        assert settings.MAX_RETRIES == 10
        assert settings.LOG_LEVEL == "WARNING"
