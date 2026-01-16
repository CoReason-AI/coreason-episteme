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

import networkx as nx
import pytest

from coreason_episteme.adapters.local_clients import (
    LocalCodexClient,
    LocalGraphNexusClient,
    LocalPrismClient,
    LocalSearchClient,
    LocalVeritasClient,
)
from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.models import (
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
    KnowledgeGap,
)


@pytest.fixture  # type: ignore[misc]
def mock_graph_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_prism_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_codex_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_search_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_veritas_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def bridge_builder(
    mock_graph_client: MagicMock,
    mock_prism_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
    mock_veritas_client: MagicMock,
) -> BridgeBuilderImpl:
    return BridgeBuilderImpl(
        graph_client=mock_graph_client,
        prism_client=mock_prism_client,
        codex_client=mock_codex_client,
        search_client=mock_search_client,
        veritas_client=mock_veritas_client,
    )


def test_generate_hypothesis_success(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: MagicMock,
    mock_prism_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
    mock_veritas_client: MagicMock,
) -> None:
    """Test successful hypothesis generation with verification and logging."""
    # Setup inputs
    gap = KnowledgeGap(description="Gap between A and B", source_nodes=["NodeA", "NodeB"])

    # Setup mock returns
    bridge_target = GeneticTarget(
        symbol="GeneX",
        ensembl_id="ENSG000001",
        druggability_score=0.0,  # Placeholder, will be checked by prism
        novelty_score=0.8,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]

    mock_prism_client.check_druggability.return_value = 0.9
    mock_codex_client.validate_target.return_value = bridge_target.model_copy()

    # Search verification passes
    mock_search_client.verify_citation.return_value = True

    # Execute
    hypothesis = bridge_builder.generate_hypothesis(gap)

    # Verify
    assert hypothesis is not None
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.confidence == ConfidenceLevel.SPECULATIVE
    assert hypothesis.target_candidate.symbol == "GeneX"

    # Verify interactions
    mock_graph_client.find_latent_bridges.assert_called_with("NodeA", "NodeB")
    mock_prism_client.check_druggability.assert_called_with("ENSG000001")
    mock_codex_client.validate_target.assert_called_with("GeneX")

    # Verify hallucination check
    expected_claim = "NodeA interacts with GeneX and GeneX affects NodeB"
    mock_search_client.verify_citation.assert_called_with(expected_claim)

    # Verify logging
    mock_veritas_client.log_trace.assert_called_once()
    args, _ = mock_veritas_client.log_trace.call_args
    assert args[0] == hypothesis.id
    assert args[1]["gap"] == "Gap between A and B"
    assert args[1]["selected_target"] == "GeneX"


def test_generate_hypothesis_citation_verification_fail(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: MagicMock,
    mock_prism_client: MagicMock,
    mock_codex_client: MagicMock,
    mock_search_client: MagicMock,
    mock_veritas_client: MagicMock,
) -> None:
    """Test when citation verification fails."""
    gap = KnowledgeGap(description="Gap between A and B", source_nodes=["NodeA", "NodeB"])

    bridge_target = GeneticTarget(
        symbol="GeneX",
        ensembl_id="ENSG000001",
        druggability_score=0.0,
        novelty_score=0.8,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]
    mock_prism_client.check_druggability.return_value = 0.9
    mock_codex_client.validate_target.return_value = bridge_target.model_copy()

    # Search verification FAILS
    mock_search_client.verify_citation.return_value = False

    # Execute
    hypothesis = bridge_builder.generate_hypothesis(gap)

    # Verify
    assert hypothesis is None
    mock_search_client.verify_citation.assert_called()
    mock_veritas_client.log_trace.assert_not_called()


def test_generate_hypothesis_no_bridges(bridge_builder: BridgeBuilderImpl, mock_graph_client: MagicMock) -> None:
    """Test when no latent bridges are found."""
    gap = KnowledgeGap(description="Gap between A and B", source_nodes=["NodeA", "NodeB"])
    mock_graph_client.find_latent_bridges.return_value = []

    hypothesis = bridge_builder.generate_hypothesis(gap)

    assert hypothesis is None


def test_generate_hypothesis_no_druggable_bridges(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: MagicMock,
    mock_prism_client: MagicMock,
) -> None:
    """Test when bridges exist but are not druggable."""
    gap = KnowledgeGap(description="Gap between A and B", source_nodes=["NodeA", "NodeB"])

    bridge_target = GeneticTarget(
        symbol="GeneY",
        ensembl_id="ENSG000002",
        druggability_score=0.0,
        novelty_score=0.5,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]
    mock_prism_client.check_druggability.return_value = 0.3

    hypothesis = bridge_builder.generate_hypothesis(gap)

    assert hypothesis is None


def test_generate_hypothesis_codex_validation_fail(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: MagicMock,
    mock_prism_client: MagicMock,
    mock_codex_client: MagicMock,
) -> None:
    """Test when Codex fails to validate the target."""
    gap = KnowledgeGap(description="Gap between A and B", source_nodes=["NodeA", "NodeB"])

    bridge_target = GeneticTarget(
        symbol="GeneZ",
        ensembl_id="ENSG000003",
        druggability_score=0.0,
        novelty_score=0.5,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]
    mock_prism_client.check_druggability.return_value = 0.8
    mock_codex_client.validate_target.return_value = None

    hypothesis = bridge_builder.generate_hypothesis(gap)

    assert hypothesis is None


def test_generate_hypothesis_insufficient_nodes(
    bridge_builder: BridgeBuilderImpl,
) -> None:
    """Test when gap has fewer than 2 source nodes."""
    gap_one_node = KnowledgeGap(description="Gap with one node", source_nodes=["NodeA"])
    gap_no_node = KnowledgeGap(description="Gap with no nodes", source_nodes=[])
    gap_none = KnowledgeGap(description="Gap with None nodes", source_nodes=None)

    assert bridge_builder.generate_hypothesis(gap_one_node) is None
    assert bridge_builder.generate_hypothesis(gap_no_node) is None
    assert bridge_builder.generate_hypothesis(gap_none) is None


def test_bridge_builder_with_local_clients() -> None:
    """Test BridgeBuilder using purely Local clients to verify wiring."""
    # Setup
    g = nx.Graph()
    # Source and Target
    g.add_node("DiseaseX", type="disease")
    g.add_node("GeneA", type="gene")
    # Bridge
    g.add_node("BridgeGene", is_bridge=True, ensembl_id="ENSG_BRIDGE", druggability=0.8)

    graph_client = LocalGraphNexusClient(g)

    # Needs to match what LocalGraphNexusClient.find_latent_bridges returns
    prism_client = LocalPrismClient({"ENSG_BRIDGE": 0.8})
    codex_client = LocalCodexClient()  # validates all
    search_client = LocalSearchClient(
        {"valid_claims": ["DiseaseX interacts with BridgeGene and BridgeGene affects GeneA"]}
    )
    veritas_client = LocalVeritasClient()

    builder = BridgeBuilderImpl(graph_client, prism_client, codex_client, search_client, veritas_client)

    gap = KnowledgeGap(description="Gap", source_nodes=["DiseaseX", "GeneA"])

    # Execute
    hypothesis = builder.generate_hypothesis(gap)

    # Verify
    assert hypothesis is not None
    assert hypothesis.target_candidate.symbol == "BridgeGene"
    assert len(veritas_client.logs) == 1
