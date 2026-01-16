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
from typing import Optional

from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    PrismClient,
    SearchClient,
    VeritasClient,
)
from coreason_episteme.models import ConfidenceLevel, Hypothesis, KnowledgeGap
from coreason_episteme.utils.logger import logger


class BridgeBuilderImpl:
    """Implementation of the Bridge Builder (Hypothesis Formulator)."""

    def __init__(
        self,
        graph_client: GraphNexusClient,
        prism_client: PrismClient,
        codex_client: CodexClient,
        search_client: SearchClient,
        veritas_client: VeritasClient,
    ):
        self.graph_client = graph_client
        self.prism_client = prism_client
        self.codex_client = codex_client
        self.search_client = search_client
        self.veritas_client = veritas_client

    def generate_hypothesis(self, gap: KnowledgeGap) -> Optional[Hypothesis]:
        """
        Generates a hypothesis bridging the knowledge gap.

        1. Queries GraphNexus for latent bridges between source nodes.
        2. Filters bridges for druggability via Prism.
        3. Validates target via Codex.
        4. Verifies citations via Search (Hallucination Check).
        5. Logs trace via Veritas.
        6. Constructs a Hypothesis.
        """
        logger.info(f"Attempting to build bridge for gap: {gap.description}")

        if not gap.source_nodes or len(gap.source_nodes) < 2:
            logger.warning("Gap does not have enough source nodes to find bridges.")
            return None

        # We start with the order provided, but we might swap them if the evidence flow suggests otherwise.
        node_a = gap.source_nodes[0]
        node_b = gap.source_nodes[1]

        potential_bridges = self.graph_client.find_latent_bridges(node_a, node_b)
        if not potential_bridges:
            logger.info("No latent bridges found.")
            return None

        best_candidate = None
        best_druggability = -1.0
        # Keep track of the resolved direction for the best candidate
        # (source, target) tuple
        best_direction = (node_a, node_b)

        for bridge in potential_bridges:
            # Check druggability
            druggability = self.prism_client.check_druggability(bridge.ensembl_id)
            if druggability > 0.5:  # Threshold for "druggable"
                # Validate details with Codex
                validated_target = self.codex_client.validate_target(bridge.symbol)
                if validated_target:
                    # Hallucination Check: Verify citation in both directions

                    # Direction 1: A -> Bridge -> B
                    claim_1 = (
                        f"{node_a} interacts with {validated_target.symbol} "
                        f"and {validated_target.symbol} affects {node_b}"
                    )
                    is_verified = False

                    if self.search_client.verify_citation(claim_1):
                        is_verified = True
                        current_direction = (node_a, node_b)
                    else:
                        # Direction 2: B -> Bridge -> A
                        claim_2 = (
                            f"{node_b} interacts with {validated_target.symbol} "
                            f"and {validated_target.symbol} affects {node_a}"
                        )
                        if self.search_client.verify_citation(claim_2):
                            is_verified = True
                            current_direction = (node_b, node_a)

                    if is_verified:
                        # Update with fresh data from Codex (and keep the druggability score)
                        validated_target.druggability_score = druggability

                        if druggability > best_druggability:
                            best_druggability = druggability
                            best_candidate = validated_target
                            best_direction = current_direction
                    else:
                        logger.info(f"Discarding candidate {bridge.symbol} due to failed citation verification.")

        if not best_candidate:
            logger.info("No valid targets found among bridges (druggable & verified).")
            return None

        final_source_id, final_target_id = best_direction

        # Construct Hypothesis
        hypothesis_id = str(uuid.uuid4())
        mechanism = f"Regulation of {final_target_id} via {best_candidate.symbol} (bridging from {final_source_id})."

        hypothesis = Hypothesis(
            id=hypothesis_id,
            title=f"Proposed Link: {final_source_id} -> {best_candidate.symbol} -> {final_target_id}",
            knowledge_gap=gap.description,
            proposed_mechanism=mechanism,
            target_candidate=best_candidate,
            causal_validation_score=0.0,
            key_counterfactual="",
            killer_experiment_pico={},
            evidence_chain=gap.source_nodes + [best_candidate.ensembl_id],
            confidence=ConfidenceLevel.SPECULATIVE,
        )

        # Log trace
        trace_data = {
            "gap": gap.description,
            "source_nodes": gap.source_nodes,
            "bridges_found": len(potential_bridges),
            "selected_target": best_candidate.symbol,
            "mechanism": mechanism,
        }
        self.veritas_client.log_trace(hypothesis_id, trace_data)

        logger.info(f"Generated hypothesis: {hypothesis.id}")
        return hypothesis
