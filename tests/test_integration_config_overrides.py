# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import AsyncMock, Mock

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    PrismClient,
    SearchClient,
)
from coreason_episteme.models import GeneticTarget, KnowledgeGap, KnowledgeGapType


class TestIntegrationConfigOverrides:
    """
    Integration tests verifying that configuration changes dynamically affect component behavior.
    """

    @pytest.fixture
    def mock_graph_client(self) -> Mock:
        return AsyncMock(spec=GraphNexusClient)

    @pytest.fixture
    def mock_codex_client(self) -> Mock:
        return AsyncMock(spec=CodexClient)

    @pytest.fixture
    def mock_search_client(self) -> Mock:
        return AsyncMock(spec=SearchClient)

    @pytest.fixture
    def mock_prism_client(self) -> Mock:
        return AsyncMock(spec=PrismClient)

    @pytest.fixture
    def user_context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    @pytest.mark.asyncio
    async def test_gap_scanner_similarity_threshold_override(
        self,
        mock_graph_client: Mock,
        mock_codex_client: Mock,
        mock_search_client: Mock,
        user_context: UserContext,
    ) -> None:
        """
        Verify that changing GAP_SCANNER_SIMILARITY_THRESHOLD affects which clusters are considered connected.
        """
        # Setup: A pair with 0.6 similarity
        mock_graph_client.find_disconnected_clusters.return_value = [
            {
                "cluster_a_id": "C1",
                "cluster_b_id": "C2",
                "cluster_a_name": "Alpha",
                "cluster_b_name": "Beta",
            }
        ]
        mock_codex_client.get_semantic_similarity.return_value = 0.6
        mock_search_client.find_literature_inconsistency.return_value = []

        # Case 1: Default Threshold (0.75) -> Should find NO gaps (0.6 < 0.75)
        # Note: We manually inject the value to simulate the effect of the settings override
        # because the component reads the default from settings at instantiation or uses the passed value.
        # Since we refactored to allow injection, we test the injection mechanism.
        scanner_strict = GapScannerImpl(
            graph_client=mock_graph_client,
            codex_client=mock_codex_client,
            search_client=mock_search_client,
            similarity_threshold=0.75,
        )
        gaps_strict = await scanner_strict.scan("DiseaseX", context=user_context)
        assert len(gaps_strict) == 0

        # Case 2: Lowered Threshold (0.5) -> Should find 1 gap (0.6 >= 0.5)
        scanner_lax = GapScannerImpl(
            graph_client=mock_graph_client,
            codex_client=mock_codex_client,
            search_client=mock_search_client,
            similarity_threshold=0.5,
        )
        gaps_lax = await scanner_lax.scan("DiseaseX", context=user_context)
        assert len(gaps_lax) == 1
        assert gaps_lax[0].type == KnowledgeGapType.CLUSTER_DISCONNECT

    @pytest.mark.asyncio
    async def test_bridge_builder_druggability_threshold_override(
        self,
        mock_graph_client: Mock,
        mock_prism_client: Mock,
        mock_codex_client: Mock,
        mock_search_client: Mock,
        user_context: UserContext,
    ) -> None:
        """
        Verify that changing DRUGGABILITY_THRESHOLD affects which candidates are selected.
        """
        # Setup
        gap = KnowledgeGap(
            description="Gap", type=KnowledgeGapType.CLUSTER_DISCONNECT, source_nodes=["Source", "Target"]
        )
        candidate_symbol = "GENE_A"

        # Graph returns a potential bridge
        bridge_candidate = GeneticTarget(
            symbol=candidate_symbol, ensembl_id="ENSG001", druggability_score=0.4, novelty_score=0.9
        )
        mock_graph_client.find_latent_bridges.return_value = [bridge_candidate]

        # Prism returns 0.4 score
        mock_prism_client.check_druggability.return_value = 0.4

        # Codex validates OK
        mock_codex_client.validate_target.return_value = bridge_candidate

        # Search validates citation OK
        mock_search_client.verify_citation.return_value = True

        # Case 1: Default Threshold (0.5) -> Should reject (0.4 < 0.5)
        builder_strict = BridgeBuilderImpl(
            graph_client=mock_graph_client,
            prism_client=mock_prism_client,
            codex_client=mock_codex_client,
            search_client=mock_search_client,
            druggability_threshold=0.5,
        )
        result_strict = await builder_strict.generate_hypothesis(gap, context=user_context)
        assert result_strict.hypothesis is None
        assert result_strict.bridges_found_count == 1  # It found it, but filtered it

        # Case 2: Lowered Threshold (0.3) -> Should accept (0.4 >= 0.3)
        builder_lax = BridgeBuilderImpl(
            graph_client=mock_graph_client,
            prism_client=mock_prism_client,
            codex_client=mock_codex_client,
            search_client=mock_search_client,
            druggability_threshold=0.3,
        )
        result_lax = await builder_lax.generate_hypothesis(gap, context=user_context)
        assert result_lax.hypothesis is not None
        assert result_lax.hypothesis.target_candidate.symbol == candidate_symbol
