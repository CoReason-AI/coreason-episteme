# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from dataclasses import dataclass, field
from typing import List

from coreason_episteme.config import settings
from coreason_episteme.interfaces import (
    AdversarialReviewer,
    BridgeBuilder,
    CausalValidator,
    GapScanner,
    ProtocolDesigner,
    VeritasClient,
)
from coreason_episteme.models import CritiqueSeverity, Hypothesis, HypothesisTrace
from coreason_episteme.utils.logger import logger


@dataclass
class EpistemeEngine:
    """
    The Hypothesis Engine (Theorist).
    Orchestrates the Scan-Bridge-Simulate-Critique loop.
    """

    gap_scanner: GapScanner
    bridge_builder: BridgeBuilder
    causal_validator: CausalValidator
    adversarial_reviewer: AdversarialReviewer
    protocol_designer: ProtocolDesigner
    veritas_client: VeritasClient
    max_retries: int = field(default_factory=lambda: settings.MAX_RETRIES)

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
            attempts = 0

            # Initialize Trace
            trace = HypothesisTrace(gap=gap, status="PENDING")

            while attempts < self.max_retries:
                attempts += 1
                trace.excluded_targets_history = list(excluded_targets)  # Update history
                trace.refinement_retries = attempts - 1

                logger.info(
                    f"Attempt {attempts}/{self.max_retries} for gap: {gap.description[:50]}... "
                    f"(Excluded: {len(excluded_targets)})"
                )

                # 2. Latent Bridging
                bridge_result = self.bridge_builder.generate_hypothesis(gap, excluded_targets=excluded_targets)

                # Accumulate bridge metadata
                trace.bridges_found_count = bridge_result.bridges_found_count
                trace.considered_candidates = bridge_result.considered_candidates

                hypothesis = bridge_result.hypothesis
                if not hypothesis:
                    logger.info("No hypothesis generated for gap.")
                    trace.status = "DISCARDED (No Bridge)"
                    # Log trace if failed completely to find a bridge?
                    # Yes, we should log the attempt even if it failed.
                    break

                # Link trace ID to hypothesis ID if available
                trace.hypothesis_id = hypothesis.id

                # 3. Causal Simulation
                hypothesis = self.causal_validator.validate(hypothesis)

                # Accumulate validation data
                trace.causal_validation_score = hypothesis.causal_validation_score
                trace.key_counterfactual = hypothesis.key_counterfactual

                # Filtering Policy: Discard if causal plausibility is too low.
                if hypothesis.causal_validation_score < 0.5:
                    logger.info(
                        f"Hypothesis {hypothesis.id} discarded due to low causal score "
                        f"({hypothesis.causal_validation_score})."
                    )
                    trace.status = "DISCARDED (Low Causal Score)"
                    # We break here, as per original logic, though in a real system we might loop.
                    # Current logic is: break loop, move to next gap.
                    break

                # 4. Adversarial Review
                hypothesis = self.adversarial_reviewer.review(hypothesis)

                # Accumulate critiques
                trace.critiques = hypothesis.critiques

                # 5. Refinement Check
                fatal_critiques = [c for c in hypothesis.critiques if c.severity == CritiqueSeverity.FATAL]
                if fatal_critiques:
                    target_symbol = hypothesis.target_candidate.symbol
                    logger.warning(
                        f"Hypothesis {hypothesis.id} rejected due to FATAL critiques ({len(fatal_critiques)}). "
                        f"Refining loop -> Excluding {target_symbol}"
                    )
                    excluded_targets.append(target_symbol)
                    # We loop back. The trace continues.
                    # We might want to persist the "history" of rejected targets in more detail,
                    # but `excluded_targets_history` captures the symbols.
                    continue
                else:
                    # Success!
                    # 6. Protocol Design
                    hypothesis = self.protocol_designer.design_experiment(hypothesis)

                    trace.result = hypothesis
                    trace.status = "ACCEPTED"

                    # Log the final trace
                    if trace.hypothesis_id:
                        self.veritas_client.log_trace(
                            trace.hypothesis_id,
                            trace.model_dump(),
                        )

                    results.append(hypothesis)
                    break  # Move to next gap

            # If loop exhausted without success or broke early
            if trace.status != "ACCEPTED":
                # Log trace for failed attempt if we have an ID
                # If we never got a hypothesis ID (e.g. no bridges), we generate a UUID or use None
                # The interface requires hypothesis_id: str.
                log_id = trace.hypothesis_id or f"failed-gap-{hash(gap.description)}"
                self.veritas_client.log_trace(
                    log_id,
                    trace.model_dump(),
                )

        logger.info(f"Engine finished. Generated {len(results)} hypotheses.")
        return results
