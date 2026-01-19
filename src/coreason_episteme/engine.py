# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List

from coreason_episteme.interfaces import (
    AdversarialReviewer,
    BridgeBuilder,
    CausalValidator,
    GapScanner,
    ProtocolDesigner,
)
from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


class EpistemeEngine:
    """
    The Hypothesis Engine (Theorist).
    Orchestrates the Scan-Bridge-Simulate-Critique loop.
    """

    def __init__(
        self,
        gap_scanner: GapScanner,
        bridge_builder: BridgeBuilder,
        causal_validator: CausalValidator,
        adversarial_reviewer: AdversarialReviewer,
        protocol_designer: ProtocolDesigner,
    ):
        self.gap_scanner = gap_scanner
        self.bridge_builder = bridge_builder
        self.causal_validator = causal_validator
        self.adversarial_reviewer = adversarial_reviewer
        self.protocol_designer = protocol_designer

    def run(self, disease_id: str) -> List[Hypothesis]:
        """
        Executes the hypothesis generation pipeline for a given disease ID.

        1. Scan for Knowledge Gaps.
        2. Generate Hypotheses (Bridge the Gaps).
        3. Validate (Causal Simulation).
        4. Adversarial Review (Critique).
        5. Protocol Design (Experiment).
        """
        logger.info(f"Starting Episteme Engine for: {disease_id}")
        results: List[Hypothesis] = []

        # 1. Gap Scanning
        gaps = self.gap_scanner.scan(disease_id)
        if not gaps:
            logger.info("No knowledge gaps found. Exiting.")
            return []

        for gap in gaps:
            # 2. Latent Bridging
            hypothesis = self.bridge_builder.generate_hypothesis(gap)
            if not hypothesis:
                continue

            # 3. Causal Simulation
            hypothesis = self.causal_validator.validate(hypothesis)

            # Filtering Policy: Discard if causal plausibility is too low.
            # Assuming a threshold of 0.5 for "Plausible" enough to proceed.
            if hypothesis.causal_validation_score < 0.5:
                logger.info(
                    f"Hypothesis {hypothesis.id} discarded due to low causal score "
                    f"({hypothesis.causal_validation_score})."
                )
                continue

            # 4. Adversarial Review
            hypothesis = self.adversarial_reviewer.review(hypothesis)

            # 5. Protocol Design
            # Only design experiments for surviving hypotheses
            hypothesis = self.protocol_designer.design_experiment(hypothesis)

            results.append(hypothesis)

        logger.info(f"Engine finished. Generated {len(results)} hypotheses.")
        return results
