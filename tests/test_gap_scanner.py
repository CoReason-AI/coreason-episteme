# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import AsyncMock

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.interfaces import GapScanner
from coreason_episteme.models import KnowledgeGap, KnowledgeGapType
from tests.mocks import MockGapScanner


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        user_id="test-user",
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


def test_interface_definition() -> None:
    """Test that GapScanner interface is importable."""
    assert GapScanner is not None


@pytest.mark.asyncio
async def test_mock_gap_scanner_scan_found(user_context: UserContext) -> None:
    """Test that the MockGapScanner returns a gap when one is expected."""
    scanner = MockGapScanner()
    gaps = await scanner.scan("DiseaseX", context=user_context)

    assert len(gaps) == 1
    assert isinstance(gaps[0], KnowledgeGap)
    assert gaps[0].id is not None  # Verify ID is generated
    assert "DiseaseX" in gaps[0].description
    assert gaps[0].source_nodes == ["PMID:123456", "PMID:789012"]


@pytest.mark.asyncio
async def test_mock_gap_scanner_scan_not_found(user_context: UserContext) -> None:
    """Test that the MockGapScanner returns an empty list for 'CleanTarget'."""
    scanner = MockGapScanner()
    gaps = await scanner.scan("CleanTarget", context=user_context)

    assert len(gaps) == 0


# --- GapScannerImpl Tests ---


@pytest.fixture
def mock_graph_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_codex_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_search_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def gap_scanner_impl(
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
) -> GapScannerImpl:
    return GapScannerImpl(
        graph_client=mock_graph_client,
        codex_client=mock_codex_client,
        search_client=mock_search_client,
    )


@pytest.mark.asyncio
async def test_scan_cluster_gap_found(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
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
    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    # Verify
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.id is not None  # Verify ID is generated
    assert gap.type == KnowledgeGapType.CLUSTER_DISCONNECT
    assert "Cluster A" in gap.description
    assert "Cluster B" in gap.description
    assert gap.source_nodes == ["ID_A", "ID_B"]

    mock_graph_client.find_disconnected_clusters.assert_called_with({"target": "TargetX"})
    mock_codex_client.get_semantic_similarity.assert_called_with("ID_A", "ID_B")


@pytest.mark.asyncio
async def test_scan_cluster_gap_filtered_low_similarity(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
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
    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    # Verify
    assert len(gaps) == 0


@pytest.mark.asyncio
async def test_scan_literature_gap_found(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
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
    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    # Verify
    assert len(gaps) == 1
    assert gaps[0] == lit_gap
    mock_search_client.find_literature_inconsistency.assert_called_with("TargetX")


@pytest.mark.asyncio
async def test_scan_combined_results(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
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
    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    # Verify
    assert len(gaps) == 2
    types = {g.type for g in gaps}
    assert KnowledgeGapType.CLUSTER_DISCONNECT in types
    assert KnowledgeGapType.LITERATURE_INCONSISTENCY in types


@pytest.mark.asyncio
async def test_scan_malformed_cluster_data(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test handling of clusters missing IDs."""
    # Setup
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"cluster_a_id": "A"}  # Missing B
    ]

    # Execute
    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    # Verify
    assert len(gaps) == 0
    mock_codex_client.get_semantic_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_scan_boundary_condition(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test boundary condition where similarity is exactly 0.75."""
    mock_graph_client.find_disconnected_clusters.return_value = [{"cluster_a_id": "A", "cluster_b_id": "B"}]
    mock_codex_client.get_semantic_similarity.return_value = 0.75
    mock_search_client.find_literature_inconsistency.return_value = []

    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    assert len(gaps) == 1
    assert gaps[0].type == KnowledgeGapType.CLUSTER_DISCONNECT


@pytest.mark.asyncio
async def test_scan_robustness_malformed_data(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test robustness against malformed data (None, empty strings)."""
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"cluster_a_id": None, "cluster_b_id": "B"},
        {"cluster_a_id": "A", "cluster_b_id": None},
        {"cluster_a_id": "", "cluster_b_id": "B"},  # Empty string treated as falsey
        {"cluster_a_id": "A", "cluster_b_id": ""},
    ]
    mock_search_client.find_literature_inconsistency.return_value = []

    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    assert len(gaps) == 0
    mock_codex_client.get_semantic_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_scan_complex_scenario(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test a complex scenario with mixed valid/invalid clusters and search results."""
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"cluster_a_id": "A1", "cluster_b_id": "B1"},  # Valid (Sim 0.8)
        {"cluster_a_id": "A2", "cluster_b_id": "B2"},  # Invalid (Sim 0.4)
        {"cluster_a_id": "A3"},  # Malformed
    ]

    def similarity_side_effect(a: str, b: str) -> float:
        if a == "A1" and b == "B1":
            return 0.8
        if a == "A2" and b == "B2":
            return 0.4
        return 0.0

    mock_codex_client.get_semantic_similarity.side_effect = similarity_side_effect

    mock_search_client.find_literature_inconsistency.return_value = [
        KnowledgeGap(description="Lit Gap", type=KnowledgeGapType.LITERATURE_INCONSISTENCY, source_nodes=["PMID:1"])
    ]

    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    assert len(gaps) == 2
    assert any(g.type == KnowledgeGapType.CLUSTER_DISCONNECT for g in gaps)
    assert any(g.type == KnowledgeGapType.LITERATURE_INCONSISTENCY for g in gaps)


@pytest.mark.asyncio
async def test_scan_duplicate_symmetry(
    gap_scanner_impl: GapScannerImpl,
    mock_graph_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """
    Test behavior when symmetric pairs (A-B and B-A) are returned.
    Current behavior: Both are reported.
    """
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"cluster_a_id": "A", "cluster_b_id": "B", "cluster_a_name": "Cluster A", "cluster_b_name": "Cluster B"},
        {"cluster_a_id": "B", "cluster_b_id": "A", "cluster_a_name": "Cluster B", "cluster_b_name": "Cluster A"},
    ]
    mock_codex_client.get_semantic_similarity.return_value = 0.9
    mock_search_client.find_literature_inconsistency.return_value = []

    gaps = await gap_scanner_impl.scan("TargetX", context=user_context)

    assert len(gaps) == 2
    assert gaps[0].source_nodes == ["A", "B"]
    assert gaps[1].source_nodes == ["B", "A"]
