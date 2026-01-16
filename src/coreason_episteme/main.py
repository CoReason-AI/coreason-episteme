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

from coreason_episteme.adapters.local_clients import (
    LocalCodexClient,
    LocalGraphNexusClient,
    LocalInferenceClient,
    LocalPrismClient,
    LocalSearchClient,
    LocalVeritasClient,
)
from coreason_episteme.components.bridge_builder import BridgeBuilderImpl
from coreason_episteme.components.causal_validator import CausalValidatorImpl
from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.components.protocol_designer import ProtocolDesignerImpl
from coreason_episteme.models import ConfidenceLevel, Hypothesis
from coreason_episteme.utils.logger import logger


class EpistemeEngine:
    """
    The Orchestrator for Scientific Intuition.
    Implements the Scan-Bridge-Simulate-Critique Loop.
    """

    def __init__(self, use_local_defaults: bool = True):
        # In a real app, these would be injected or configured via env vars.
        # For this package, we default to the local adapters we built.
        if use_local_defaults:
            self.graph_client = LocalGraphNexusClient()
            self.search_client = LocalSearchClient()
            self.codex_client = LocalCodexClient()
            self.prism_client = LocalPrismClient()
            self.inference_client = LocalInferenceClient()
            self.veritas_client = LocalVeritasClient()
        else:
            # Placeholder for real configuration
            raise NotImplementedError("Only local defaults are currently supported.")

        # Initialize Components
        self.gap_scanner = GapScannerImpl(self.graph_client, self.search_client, self.codex_client)
        self.bridge_builder = BridgeBuilderImpl(
            self.graph_client, self.prism_client, self.codex_client, self.search_client, self.veritas_client
        )
        self.causal_validator = CausalValidatorImpl(self.inference_client)
        self.protocol_designer = ProtocolDesignerImpl()

    def generate_hypothesis(self, disease_id: str) -> List[Hypothesis]:
        """
        Generates novel hypotheses for the given disease/target.

        Process:
        1. Scan for Gaps.
        2. Bridge Gaps to form Hypotheses.
        3. Validate Hypotheses via Causal Simulation.
        4. Design Experiments for valid Hypotheses.
        """
        logger.info(f"Starting hypothesis generation for {disease_id}")

        generated_hypotheses: List[Hypothesis] = []

        # 1. Scan
        gaps = self.gap_scanner.scan(disease_id)
        if not gaps:
            logger.info("No knowledge gaps found.")
            return []

        for gap in gaps:
            # 2. Bridge
            hypothesis = self.bridge_builder.generate_hypothesis(gap)
            if not hypothesis:
                continue

            # 3. Validate (Simulate)
            hypothesis = self.causal_validator.validate(hypothesis)

            # Filter low scores?
            # PRD: "Hypotheses scoring below threshold are discarded before human review."
            # Let's say threshold is 0.5
            if hypothesis.causal_validation_score < 0.5:
                logger.info(
                    f"Hypothesis {hypothesis.id} discarded due to low validation score "
                    f"({hypothesis.causal_validation_score})"
                )
                continue

            # 4. Design Protocol
            hypothesis = self.protocol_designer.design_experiment(hypothesis)

            # Upgrade confidence if high score
            if hypothesis.causal_validation_score > 0.8:
                hypothesis.confidence = ConfidenceLevel.PROBABLE
            elif hypothesis.causal_validation_score > 0.5:
                hypothesis.confidence = ConfidenceLevel.PLAUSIBLE

            generated_hypotheses.append(hypothesis)

        logger.info(f"Generated {len(generated_hypotheses)} hypotheses.")
        return generated_hypotheses
