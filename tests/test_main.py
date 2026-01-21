from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_episteme.main import Episteme, EpistemeAsync, generate_hypothesis, hello_world


@pytest.fixture
def mock_clients() -> Dict[str, MagicMock]:
    return {
        "graph_client": MagicMock(),
        "codex_client": MagicMock(),
        "search_client": MagicMock(),
        "prism_client": MagicMock(),
        "inference_client": MagicMock(),
        "veritas_client": MagicMock(),
    }


def test_hello_world() -> None:
    assert hello_world() == "Hello World!"


@pytest.mark.asyncio
async def test_episteme_async_context_manager(mock_clients: Dict[str, Any]) -> None:
    # Mock httpx.AsyncClient to verify aclose is called
    with patch("httpx.AsyncClient") as MockClient:
        mock_http_client = AsyncMock()
        MockClient.return_value = mock_http_client

        async with EpistemeAsync(**mock_clients) as svc:
            assert svc is not None

        # Verify aclose was called since we didn't provide a client
        mock_http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_episteme_async_external_client(mock_clients: Dict[str, Any]) -> None:
    external_client = AsyncMock()
    async with EpistemeAsync(**mock_clients, client=external_client) as svc:
        assert svc is not None

    # Verify aclose was NOT called on external client
    external_client.aclose.assert_not_awaited()


def test_episteme_sync_context_manager(mock_clients: Dict[str, Any]) -> None:
    with patch("httpx.AsyncClient") as MockClient:
        mock_http_client = AsyncMock()
        MockClient.return_value = mock_http_client

        with Episteme(**mock_clients) as svc:
            assert svc is not None

        # Episteme.__exit__ runs async __aexit__ which calls aclose
        mock_http_client.aclose.assert_awaited_once()


def test_generate_hypothesis_sync_facade(mock_clients: Dict[str, Any]) -> None:
    with patch("coreason_episteme.main.Episteme") as MockEpisteme:
        mock_instance = MagicMock()
        MockEpisteme.return_value.__enter__.return_value = mock_instance

        generate_hypothesis("disease_123", **mock_clients)

        mock_instance.run.assert_called_once_with("disease_123")


def test_generate_hypothesis_missing_deps() -> None:
    with pytest.raises(RuntimeError, match="Missing required external client"):
        generate_hypothesis("disease_123")
