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

from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
    KnowledgeGap,
    KnowledgeGapType,
)
from coreason_episteme.utils.logger import logger


class MockGapScanner:
    """A mock implementation of the GapScanner for testing and development."""

    def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Simulates scanning for knowledge gaps.

        Args:
            target: The disease or entity ID.

        Returns:
            A list containing a mock KnowledgeGap, or empty list if target is 'CleanTarget'.
        """
        logger.info(f"Scanning for gaps related to {target} (MOCK)")

        if target == "CleanTarget":
            logger.info("No gaps found.")
            return []

        # Return a simulated gap
        gap = KnowledgeGap(
            description=(
                f"Simulated gap: Unexplained inhibition of {target} pathway despite high expression of Regulator X."
            ),
            type=KnowledgeGapType.CLUSTER_DISCONNECT,
            source_nodes=["PMID:123456", "PMID:789012"],
        )
        return [gap]


class MockBridgeBuilder:
    def generate_hypothesis(self, gap: KnowledgeGap) -> Optional[Hypothesis]:
        if "Unbridgeable" in gap.description:
            return None

        return Hypothesis(
            id=str(uuid.uuid4()),
            title="Mock Hypothesis",
            knowledge_gap=gap.description,
            proposed_mechanism="Mock Mechanism",
            target_candidate=GeneticTarget(
                symbol="MOCK1", ensembl_id="ENSG000001", druggability_score=0.9, novelty_score=0.8
            ),
            causal_validation_score=0.0,
            key_counterfactual="",
            killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
            evidence_chain=[],
            confidence=ConfidenceLevel.SPECULATIVE,
        )


class MockCausalValidator:
    def validate(self, hypothesis: Hypothesis) -> Hypothesis:
        # Simulate validation score
        if "BadTarget" in hypothesis.target_candidate.symbol:
            hypothesis.causal_validation_score = 0.1
        else:
            hypothesis.causal_validation_score = 0.85
            hypothesis.key_counterfactual = "If Mock1 is inhibited, disease Y is reduced."
        return hypothesis


class MockAdversarialReviewer:
    def review(self, hypothesis: Hypothesis) -> Hypothesis:
        if "Risky" in hypothesis.target_candidate.symbol:
            hypothesis.critiques.append("Toxicology risk detected.")
        return hypothesis


class MockProtocolDesigner:
    def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        hypothesis.killer_experiment_pico = PICO(
            population="Mice",
            intervention="Drug X",
            comparator="Placebo",
            outcome="Survival",
        )
        return hypothesis
