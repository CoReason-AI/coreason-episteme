# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from coreason_episteme.utils import logger as logger_module
from coreason_episteme.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    # Note: running this test might actually create the directory in the test environment
    # if it doesn't exist.

    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()

    # Verify app.log creation if it was logged to (it might be empty or not created until log)
    # logger.info("Test log")
    # assert (log_path / "app.log").exists()


def test_logger_creates_directory_if_not_exists() -> None:
    """Test that the logger creates the log directory if it doesn't exist."""
    # We patch pathlib.Path because the module imports it directly
    with patch("pathlib.Path") as mock_path:
        # Mock the Path object instance
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        # Simulate directory does not exist
        mock_path_instance.exists.return_value = False

        # Reload the module to trigger the top-level code
        importlib.reload(logger_module)

        # Verify mkdir was called
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)

    # Restore original module to avoid side effects
    importlib.reload(logger_module)


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
