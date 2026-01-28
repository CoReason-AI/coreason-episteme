# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

"""
Bridge Builder component implementation.

This module implements the `BridgeBuilderImpl`, responsible for generating
hypotheses by finding latent bridges in the knowledge graph.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

from coreason_identity.models import UserContext

from coreason_episteme.config import settings
from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    PrismClient,
    SearchClient,
)
from coreason_episteme.models import (
    PICO,
    BridgeResult,
    ConfidenceLevel,
    Hypothesis,
    KnowledgeGap,
)
from coreason_episteme.utils.logger import logger


@dataclass
class BridgeBuilderImpl:
    """
    Implementation of the Bridge Builder (Hypothesis Formulator).

    Generates hypotheses by finding "Latent Bridges" between disconnected concepts
    in the Knowledge Graph. It ensures targets are druggable and citation-backed.

    Attributes:
        graph_client: Client for GraphNexus (Traversals).
        prism_client: Client for Prism (Druggability).
        codex_client: Client for Codex (Ontology).
        search_client: Client for Search (Hallucination Check).
        druggability_threshold: Minimum score to consider a target druggable.
    """

    graph_client: GraphNexusClient
    prism_client: PrismClient
    codex_client: CodexClient
    search_client: SearchClient
    druggability_threshold: float = field(default_factory=lambda: settings.DRUGGABILITY_THRESHOLD)

    async def generate_hypothesis(
        self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
    ) -> BridgeResult:
        """
        Generates a hypothesis bridging the knowledge gap.

        Process:
        1. Queries GraphNexus for latent bridges between source nodes.
        2. Filters out targets listed in `excluded_targets`.
        3. Checks druggability using `prism_client`.
        4. Validates target details using `codex_client`.
        5. Performs a "Hallucination Check" using `search_client` to verify citations.
        6. Selects the best candidate (highest druggability) and constructs a Hypothesis.

        Args:
            gap: The KnowledgeGap to bridge.
            context: The user context triggering the generation.
            excluded_targets: Optional list of target symbols to exclude from consideration.

        Returns:
            BridgeResult: A BridgeResult containing the hypothesis (if found) and metadata about the process.
        """
        logger.info(
            f"Attempting to build bridge for gap: {gap.description}",
            user_id=context.sub,
        )
        if excluded_targets:
            logger.info(f"Excluding targets: {excluded_targets}")

        # Default result (failure)
        result_metadata: Dict[str, Any] = {"bridges_found_count": 0, "considered_candidates": []}

        if not gap.source_nodes or len(gap.source_nodes) < 2:
            logger.warning("Gap does not have enough source nodes to find bridges.")
            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

        source_id = gap.source_nodes[0]
        target_id = gap.source_nodes[1]

        potential_bridges = await self.graph_client.find_latent_bridges(source_id, target_id)

        # Populate metadata
        result_metadata["bridges_found_count"] = len(potential_bridges)
        result_metadata["considered_candidates"] = [b.symbol for b in potential_bridges]

        if not potential_bridges:
            logger.info("No latent bridges found.")
            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

        best_candidate = None
        best_druggability = -1.0

        for bridge in potential_bridges:
            # Check exclusions
            if excluded_targets and bridge.symbol in excluded_targets:
                logger.debug(f"Skipping excluded target: {bridge.symbol}")
                continue

            # Check druggability
            druggability = await self.prism_client.check_druggability(bridge.ensembl_id)
            if druggability > self.druggability_threshold:  # Threshold for "druggable"
                # Validate details with Codex
                validated_target = await self.codex_client.validate_target(bridge.symbol)
                if validated_target:
                    # Hallucination Check: Verify citation
                    # We construct a claim to verify.
                    interaction_claim = (
                        f"{source_id} interacts with {validated_target.symbol} "
                        f"and {validated_target.symbol} affects {target_id}"
                    )
                    is_verified = await self.search_client.verify_citation(interaction_claim)

                    if is_verified:
                        # Update with fresh data from Codex (and keep the druggability score)
                        validated_target.druggability_score = druggability

                        if druggability > best_druggability:
                            best_druggability = druggability
                            best_candidate = validated_target
                    else:
                        logger.info(f"Discarding candidate {bridge.symbol} due to failed citation verification.")

        if not best_candidate:
            logger.info("No valid targets found among bridges (druggable & verified).")
            # Explicitly cast or assure types to satisfy mypy strictness
            return BridgeResult(
                hypothesis=None,
                bridges_found_count=int(result_metadata["bridges_found_count"]),
                considered_candidates=cast(List[str], result_metadata["considered_candidates"]),
            )

        # Construct Hypothesis
        hypothesis_id = str(uuid.uuid4())
        mechanism = f"Regulation of {target_id} via {best_candidate.symbol} (bridging from {source_id})."

        hypothesis = Hypothesis(
            id=hypothesis_id,
            title=f"Proposed Link: {source_id} -> {best_candidate.symbol} -> {target_id}",
            knowledge_gap=gap.description,
            proposed_mechanism=mechanism,
            target_candidate=best_candidate,
            causal_validation_score=0.0,
            key_counterfactual="",
            killer_experiment_pico=PICO(population="TBD", intervention="TBD", comparator="TBD", outcome="TBD"),
            evidence_chain=gap.source_nodes + [best_candidate.ensembl_id],
            confidence=ConfidenceLevel.SPECULATIVE,
        )

        logger.info(f"Generated hypothesis: {hypothesis.id}")
        return BridgeResult(
            hypothesis=hypothesis,
            bridges_found_count=int(result_metadata["bridges_found_count"]),
            considered_candidates=cast(List[str], result_metadata["considered_candidates"]),
        )
