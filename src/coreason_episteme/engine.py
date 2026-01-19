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
from coreason_episteme.models import CritiqueSeverity, Hypothesis
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
        5. Refinement Loop: If FATAL critiques exist, exclude target and retry.
        6. Protocol Design (Experiment).
        """
        logger.info(f"Starting Episteme Engine for: {disease_id}")
        results: List[Hypothesis] = []

        # 1. Gap Scanning
        gaps = self.gap_scanner.scan(disease_id)
        if not gaps:
            logger.info("No knowledge gaps found. Exiting.")
            return []

        for gap in gaps:
            excluded_targets: List[str] = []
            max_retries = 3
            attempts = 0

            while attempts < max_retries:
                attempts += 1
                logger.info(
                    f"Attempt {attempts}/{max_retries} for gap: {gap.description[:50]}... "
                    f"(Excluded: {len(excluded_targets)})"
                )

                # 2. Latent Bridging
                hypothesis = self.bridge_builder.generate_hypothesis(gap, excluded_targets=excluded_targets)
                if not hypothesis:
                    logger.info("No hypothesis generated for gap.")
                    break

                # 3. Causal Simulation
                hypothesis = self.causal_validator.validate(hypothesis)

                # Filtering Policy: Discard if causal plausibility is too low.
                if hypothesis.causal_validation_score < 0.5:
                    logger.info(
                        f"Hypothesis {hypothesis.id} discarded due to low causal score "
                        f"({hypothesis.causal_validation_score})."
                    )
                    break  # Assuming low score on best candidate means no good path, or we could continue filtering?
                    # For now, let's assume if the best candidate fails causal check, we stop or loop?
                    # The prompt implies refinement loop is for "Critique".
                    # But low score is also a failure.
                    # Let's sticking to refining on CRITIQUES for now as per PRD.

                # 4. Adversarial Review
                hypothesis = self.adversarial_reviewer.review(hypothesis)

                # 5. Refinement Check
                fatal_critiques = [c for c in hypothesis.critiques if c.severity == CritiqueSeverity.FATAL]
                if fatal_critiques:
                    target_symbol = hypothesis.target_candidate.symbol
                    logger.warning(
                        f"Hypothesis {hypothesis.id} rejected due to FATAL critiques ({len(fatal_critiques)}). "
                        f"Refining loop -> Excluding {target_symbol}"
                    )
                    excluded_targets.append(target_symbol)
                    continue  # Loop back to try next candidate
                else:
                    # Success!
                    # 6. Protocol Design
                    hypothesis = self.protocol_designer.design_experiment(hypothesis)
                    results.append(hypothesis)
                    break  # Move to next gap

        logger.info(f"Engine finished. Generated {len(results)} hypotheses.")
        return results
