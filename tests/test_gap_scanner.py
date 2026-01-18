# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import MagicMock

import pytest

from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.interfaces import GapScanner
from coreason_episteme.models import KnowledgeGap, KnowledgeGapType
from tests.mocks import MockGapScanner


def test_interface_definition() -> None:
    """Test that GapScanner interface is importable."""
    assert GapScanner is not None


def test_mock_gap_scanner_scan_found() -> None:
    """Test that the MockGapScanner returns a gap when one is expected."""
    scanner = MockGapScanner()
    gaps = scanner.scan("DiseaseX")

    assert len(gaps) == 1
    assert isinstance(gaps[0], KnowledgeGap)
    assert "DiseaseX" in gaps[0].description
    assert gaps[0].source_nodes == ["PMID:123456", "PMID:789012"]


def test_mock_gap_scanner_scan_not_found() -> None:
    """Test that the MockGapScanner returns an empty list for 'CleanTarget'."""
    scanner = MockGapScanner()
    gaps = scanner.scan("CleanTarget")

    assert len(gaps) == 0


# --- GapScannerImpl Tests ---


@pytest.fixture  # type: ignore[misc]
def mock_graph_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_codex_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_search_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def gap_scanner_impl(
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
) -> GapScannerImpl:
    return GapScannerImpl(
        graph_client=mock_graph_client,
        codex_client=mock_codex_client,
        search_client=mock_search_client,
    )


def test_scan_cluster_gap_found(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
) -> None:
    """Test that a gap is created when clusters are found with high similarity."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = [
        {
            "cluster_a_id": "ID_A",
            "cluster_b_id": "ID_B",
            "cluster_a_name": "Cluster A",
            "cluster_b_name": "Cluster B",
        }
    ]
    mock_codex_client.get_semantic_similarity.return_value = 0.8  # > 0.75
    mock_search_client.find_literature_inconsistency.return_value = []

    # Execute
    gaps = gap_scanner_impl.scan("TargetX")

    # Verify
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.type == KnowledgeGapType.CLUSTER_DISCONNECT
    assert "Cluster A" in gap.description
    assert "Cluster B" in gap.description
    assert gap.source_nodes == ["ID_A", "ID_B"]

    mock_graph_client.find_disconnected_clusters.assert_called_with({"target": "TargetX"})
    mock_codex_client.get_semantic_similarity.assert_called_with("ID_A", "ID_B")


def test_scan_cluster_gap_filtered_low_similarity(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
) -> None:
    """Test that no gap is created when similarity is low."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = [
        {
            "cluster_a_id": "ID_A",
            "cluster_b_id": "ID_B",
            "cluster_a_name": "Cluster A",
            "cluster_b_name": "Cluster B",
        }
    ]
    mock_codex_client.get_semantic_similarity.return_value = 0.5  # < 0.75
    mock_search_client.find_literature_inconsistency.return_value = []

    # Execute
    gaps = gap_scanner_impl.scan("TargetX")

    # Verify
    assert len(gaps) == 0


def test_scan_literature_gap_found(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
) -> None:
    """Test that literature inconsistencies are returned."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = []
    lit_gap = KnowledgeGap(
        description="Lit Discrepancy",
        type=KnowledgeGapType.LITERATURE_INCONSISTENCY,
        source_nodes=["PMID:1"],
    )
    mock_search_client.find_literature_inconsistency.return_value = [lit_gap]

    # Execute
    gaps = gap_scanner_impl.scan("TargetX")

    # Verify
    assert len(gaps) == 1
    assert gaps[0] == lit_gap
    mock_search_client.find_literature_inconsistency.assert_called_with("TargetX")


def test_scan_combined_results(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
) -> None:
    """Test that results from both sources are combined."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = [{"cluster_a_id": "A", "cluster_b_id": "B"}]
    mock_codex_client.get_semantic_similarity.return_value = 0.9

    lit_gap = KnowledgeGap(
        description="Lit Discrepancy",
        type=KnowledgeGapType.LITERATURE_INCONSISTENCY,
    )
    mock_search_client.find_literature_inconsistency.return_value = [lit_gap]

    # Execute
    gaps = gap_scanner_impl.scan("TargetX")

    # Verify
    assert len(gaps) == 2
    types = {g.type for g in gaps}
    assert KnowledgeGapType.CLUSTER_DISCONNECT in types
    assert KnowledgeGapType.LITERATURE_INCONSISTENCY in types


def test_scan_malformed_cluster_data(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: MagicMock,
    mock_codex_client: MagicMock,
) -> None:
    """Test handling of clusters missing IDs."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"cluster_a_id": "A"}  # Missing B
    ]

    # Execute
    gaps = gap_scanner_impl.scan("TargetX")

    # Verify
    assert len(gaps) == 0
    mock_codex_client.get_semantic_similarity.assert_not_called()
