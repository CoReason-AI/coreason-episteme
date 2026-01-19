# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import uuid
from typing import List, Optional

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


class BridgeBuilderImpl:
    """Implementation of the Bridge Builder (Hypothesis Formulator)."""

    def __init__(
        self,
        graph_client: GraphNexusClient,
        prism_client: PrismClient,
        codex_client: CodexClient,
        search_client: SearchClient,
    ):
        self.graph_client = graph_client
        self.prism_client = prism_client
        self.codex_client = codex_client
        self.search_client = search_client

    def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
        """
        Generates a hypothesis bridging the knowledge gap.

        1. Queries GraphNexus for latent bridges between source nodes.
        2. Filters out excluded_targets.
        3. Filters bridges for druggability via Prism.
        4. Validates target via Codex.
        5. Verifies citations via Search (Hallucination Check).
        6. Constructs a Hypothesis and returns BridgeResult.
        """
        logger.info(f"Attempting to build bridge for gap: {gap.description}")
        if excluded_targets:
            logger.info(f"Excluding targets: {excluded_targets}")

        # Default result (failure)
        result_metadata = {"bridges_found_count": 0, "considered_candidates": []}

        if not gap.source_nodes or len(gap.source_nodes) < 2:
            logger.warning("Gap does not have enough source nodes to find bridges.")
            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

        source_id = gap.source_nodes[0]
        target_id = gap.source_nodes[1]

        potential_bridges = self.graph_client.find_latent_bridges(source_id, target_id)

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
            druggability = self.prism_client.check_druggability(bridge.ensembl_id)
            if druggability > 0.5:  # Threshold for "druggable"
                # Validate details with Codex
                validated_target = self.codex_client.validate_target(bridge.symbol)
                if validated_target:
                    # Hallucination Check: Verify citation
                    # We construct a claim to verify.
                    interaction_claim = (
                        f"{source_id} interacts with {validated_target.symbol} "
                        f"and {validated_target.symbol} affects {target_id}"
                    )
                    is_verified = self.search_client.verify_citation(interaction_claim)

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
            return BridgeResult(
                hypothesis=None,
                bridges_found_count=result_metadata["bridges_found_count"],
                considered_candidates=result_metadata["considered_candidates"],
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
            bridges_found_count=result_metadata["bridges_found_count"],
            considered_candidates=result_metadata["considered_candidates"],
        )
