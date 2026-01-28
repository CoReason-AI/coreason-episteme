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

from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.models import (
    BridgeResult,
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
    KnowledgeGap,
    KnowledgeGapType,
)


@pytest.fixture
def mock_graph_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_prism_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_codex_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_search_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


@pytest.fixture
def bridge_builder(
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
) -> BridgeBuilderImpl:
    return BridgeBuilderImpl(
        graph_client=mock_graph_client,
        prism_client=mock_prism_client,
        codex_client=mock_codex_client,
        search_client=mock_search_client,
    )


@pytest.mark.asyncio
async def test_generate_hypothesis_success(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test successful hypothesis generation with metadata return."""
    # Setup inputs
    gap = KnowledgeGap(
        description="Gap between A and B",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

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
    result = await bridge_builder.generate_hypothesis(gap, context=user_context)

    # Verify Result Structure
    assert isinstance(result, BridgeResult)
    assert result.hypothesis is not None
    assert isinstance(result.hypothesis, Hypothesis)
    assert result.hypothesis.confidence == ConfidenceLevel.SPECULATIVE
    assert result.hypothesis.target_candidate.symbol == "GeneX"

    # Verify Metadata
    assert result.bridges_found_count == 1
    assert result.considered_candidates == ["GeneX"]

    # Verify interactions
    mock_graph_client.find_latent_bridges.assert_called_with("NodeA", "NodeB")
    mock_prism_client.check_druggability.assert_called_with("ENSG000001")
    mock_codex_client.validate_target.assert_called_with("GeneX")

    # Verify hallucination check
    expected_claim = "NodeA interacts with GeneX and GeneX affects NodeB"
    mock_search_client.verify_citation.assert_called_with(expected_claim)


@pytest.mark.asyncio
async def test_generate_hypothesis_citation_verification_fail(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test when citation verification fails."""
    gap = KnowledgeGap(
        description="Gap between A and B",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

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
    result = await bridge_builder.generate_hypothesis(gap, context=user_context)

    # Verify
    assert result.hypothesis is None
    assert result.bridges_found_count == 1
    assert "GeneX" in result.considered_candidates
    mock_search_client.verify_citation.assert_called()


@pytest.mark.asyncio
async def test_generate_hypothesis_no_bridges(
    bridge_builder: BridgeBuilderImpl, mock_graph_client: AsyncMock, user_context: UserContext
) -> None:
    """Test when no latent bridges are found."""
    gap = KnowledgeGap(
        description="Gap between A and B",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )
    mock_graph_client.find_latent_bridges.return_value = []

    result = await bridge_builder.generate_hypothesis(gap, context=user_context)

    assert result.hypothesis is None
    assert result.bridges_found_count == 0
    assert result.considered_candidates == []


@pytest.mark.asyncio
async def test_generate_hypothesis_no_druggable_bridges(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test when bridges exist but are not druggable."""
    gap = KnowledgeGap(
        description="Gap between A and B",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

    bridge_target = GeneticTarget(
        symbol="GeneY",
        ensembl_id="ENSG000002",
        druggability_score=0.0,
        novelty_score=0.5,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]
    mock_prism_client.check_druggability.return_value = 0.3

    result = await bridge_builder.generate_hypothesis(gap, context=user_context)

    assert result.hypothesis is None
    assert result.bridges_found_count == 1
    assert "GeneY" in result.considered_candidates


@pytest.mark.asyncio
async def test_generate_hypothesis_codex_validation_fail(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    mock_codex_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test when Codex fails to validate the target."""
    gap = KnowledgeGap(
        description="Gap between A and B",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

    bridge_target = GeneticTarget(
        symbol="GeneZ",
        ensembl_id="ENSG000003",
        druggability_score=0.0,
        novelty_score=0.5,
    )
    mock_graph_client.find_latent_bridges.return_value = [bridge_target]
    mock_prism_client.check_druggability.return_value = 0.8
    mock_codex_client.validate_target.return_value = None

    result = await bridge_builder.generate_hypothesis(gap, context=user_context)

    assert result.hypothesis is None
    assert result.bridges_found_count == 1
    assert "GeneZ" in result.considered_candidates


@pytest.mark.asyncio
async def test_generate_hypothesis_insufficient_nodes(
    bridge_builder: BridgeBuilderImpl, user_context: UserContext
) -> None:
    """Test when gap has fewer than 2 source nodes."""
    gap_one_node = KnowledgeGap(
        description="Gap with one node",
        source_nodes=["NodeA"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

    result = await bridge_builder.generate_hypothesis(gap_one_node, context=user_context)
    assert result.hypothesis is None
    assert result.bridges_found_count == 0


@pytest.mark.asyncio
async def test_generate_hypothesis_excluded_targets(
    bridge_builder: BridgeBuilderImpl,
    mock_graph_client: AsyncMock,
    mock_prism_client: AsyncMock,
    mock_codex_client: AsyncMock,
    mock_search_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test filtering of excluded targets."""
    gap = KnowledgeGap(
        description="Gap",
        source_nodes=["NodeA", "NodeB"],
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )

    # Candidate 1: Toxic (excluded)
    target_toxic = GeneticTarget(
        symbol="ToxicGene",
        ensembl_id="ENSG000001",
        druggability_score=0.9,
        novelty_score=0.8,
    )
    # Candidate 2: Safe
    target_safe = GeneticTarget(
        symbol="SafeGene",
        ensembl_id="ENSG000002",
        druggability_score=0.8,
        novelty_score=0.8,
    )

    mock_graph_client.find_latent_bridges.return_value = [target_toxic, target_safe]
    mock_prism_client.check_druggability.return_value = 0.9
    mock_codex_client.validate_target.side_effect = lambda s: target_toxic if s == "ToxicGene" else target_safe
    mock_search_client.verify_citation.return_value = True

    # Exclude ToxicGene
    result = await bridge_builder.generate_hypothesis(
        gap, context=user_context, excluded_targets=["ToxicGene"]
    )

    assert result.hypothesis is not None
    assert result.hypothesis.target_candidate.symbol == "SafeGene"

    # Metadata check
    # bridges_found_count counts ALL returned by graph client (2)
    assert result.bridges_found_count == 2
    assert "ToxicGene" in result.considered_candidates
    assert "SafeGene" in result.considered_candidates
