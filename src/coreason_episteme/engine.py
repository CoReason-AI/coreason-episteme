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
Core Engine module for coreason-episteme.

This module contains the `EpistemeEngineAsync` class, which orchestrates the
primary "Scan-Bridge-Simulate-Critique" workflow loop.
"""

from dataclasses import dataclass, field
from types import TracebackType
from typing import List, Optional, Type

from coreason_identity.models import UserContext

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
class EpistemeEngineAsync:
    """
    The Hypothesis Engine (Theorist).

    Orchestrates the Scan-Bridge-Simulate-Critique loop to generate and validate
    scientific hypotheses. It manages the lifecycle of hypothesis generation,
    from identifying gaps to refining candidates based on adversarial feedback.

    Attributes:
        gap_scanner: Component to identify knowledge gaps (Negative Space Analysis).
        bridge_builder: Component to propose hypotheses bridging gaps (Latent Bridging).
        causal_validator: Component to simulate and validate mechanisms (Causal Simulation).
        adversarial_reviewer: Component to critique hypotheses (Adversarial Review).
        protocol_designer: Component to design validation experiments (Protocol Design).
        veritas_client: Client for logging provenance traces.
        max_retries: Maximum number of refinement attempts for a single gap.
    """

    gap_scanner: GapScanner
    bridge_builder: BridgeBuilder
    causal_validator: CausalValidator
    adversarial_reviewer: AdversarialReviewer
    protocol_designer: ProtocolDesigner
    veritas_client: VeritasClient
    max_retries: int = field(default_factory=lambda: settings.MAX_RETRIES)

    async def __aenter__(self) -> "EpistemeEngineAsync":
        # Placeholder for resource initialization if needed
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # Placeholder for resource cleanup if needed
        pass

    async def run(self, disease_id: str, *, context: UserContext) -> List[Hypothesis]:
        """
        Executes the hypothesis generation pipeline for a given disease ID.

        Pipeline Steps:
        1. **Gap Scanning:** Identify disconnected clusters or literature inconsistencies.
        2. **Refinement Loop:** For each gap:
            a. **Latent Bridging:** Propose a target/mechanism to bridge the gap.
            b. **Causal Simulation:** Validate the mechanism via counterfactuals.
            c. **Adversarial Review:** Critique the hypothesis (Toxicology, IP, etc.).
            d. **Refinement:** If FATAL critiques occur, exclude the target and retry (up to max_retries).
            e. **Protocol Design:** Design a PICO experiment for successful hypotheses.
        3. **Provenance:** Log the full trace of the generation process via Veritas.

        Args:
            disease_id: The identifier of the disease or condition to investigate.
            context: The user context triggering the process.

        Returns:
            List[Hypothesis]: A list of validated and critiqued Hypothesis objects.
        """
        if context is None:
            raise ValueError("context is required")

        logger.info(
            f"Starting Episteme Engine for: {disease_id}",
            user_id=context.user_id,
            protocol_title=disease_id,
        )
        results: List[Hypothesis] = []

        # 1. Gap Scanning
        gaps = await self.gap_scanner.scan(disease_id, context=context)
        if not gaps:
            logger.info("No knowledge gaps found. Exiting.")
            return []

        for gap in gaps:
            excluded_targets: List[str] = []
            attempts = 0

            # Initialize Trace
            trace = HypothesisTrace(gap=gap, gap_id=gap.id, status="PENDING")

            try:
                while attempts < self.max_retries:
                    attempts += 1
                    trace.excluded_targets_history = list(excluded_targets)  # Update history
                    trace.refinement_retries = attempts - 1

                    logger.info(
                        f"Attempt {attempts}/{self.max_retries} for gap: {gap.description[:50]}... "
                        f"(Excluded: {len(excluded_targets)})"
                    )

                    # 2. Latent Bridging
                    bridge_result = await self.bridge_builder.generate_hypothesis(
                        gap, context=context, excluded_targets=excluded_targets
                    )

                    # Accumulate bridge metadata
                    trace.bridges_found_count = bridge_result.bridges_found_count
                    trace.considered_candidates = bridge_result.considered_candidates

                    hypothesis = bridge_result.hypothesis
                    if not hypothesis:
                        logger.info("No hypothesis generated for gap.")
                        trace.status = "DISCARDED (No Bridge)"
                        break

                    # Link trace ID to hypothesis ID if available
                    trace.hypothesis_id = hypothesis.id
                    # Use the ensembl_id of the target as the bridge_id
                    trace.bridge_id = hypothesis.target_candidate.ensembl_id

                    # 3. Causal Simulation
                    hypothesis = await self.causal_validator.validate(hypothesis, context=context)

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
                        break

                    # 4. Adversarial Review
                    hypothesis = await self.adversarial_reviewer.review(hypothesis, context=context)

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
                        continue
                    else:
                        # Success!
                        # 6. Protocol Design
                        hypothesis = await self.protocol_designer.design_experiment(hypothesis)

                        trace.result = hypothesis
                        trace.status = "ACCEPTED"

                        # Log the final trace
                        if trace.hypothesis_id:
                            await self.veritas_client.log_trace(
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
                    log_id = trace.hypothesis_id or f"failed-gap-{gap.id}"
                    await self.veritas_client.log_trace(
                        log_id,
                        trace.model_dump(),
                    )

            except Exception as e:
                logger.exception(f"Error processing gap {gap.description}: {e}")
                trace.status = f"ERROR: {str(e)}"
                log_id = trace.hypothesis_id or f"error-gap-{gap.id}"
                await self.veritas_client.log_trace(
                    log_id,
                    trace.model_dump(),
                )
                continue

        logger.info(f"Engine finished. Generated {len(results)} hypotheses.")
        return results
