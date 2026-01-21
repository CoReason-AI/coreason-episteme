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
Causal Validator component implementation.

This module implements the `CausalValidatorImpl`, responsible for running
counterfactual simulations to validate proposed hypotheses.
"""

from dataclasses import dataclass

from coreason_episteme.interfaces import InferenceClient
from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


@dataclass
class CausalValidatorImpl:
    """
    Implementation of the Causal Validator (The Simulator).

    Uses `coreason-inference` to run counterfactual simulations on the proposed hypothesis,
    testing if the mechanism holds up under in-silico stress testing.

    Attributes:
        inference_client: Client for Inference service.
    """

    inference_client: InferenceClient

    async def validate(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Validates the hypothesis using causal simulation.

        Runs a counterfactual simulation: "If we inhibit Target Gene [A], does it causally
        interrupt the disease pathway [C]?".

        Args:
            hypothesis: The hypothesis to validate.

        Returns:
            Hypothesis: The hypothesis object updated with the causal validation score and
            description of the key counterfactual tested.
        """
        logger.info(f"Validating hypothesis: {hypothesis.id}")

        mechanism = hypothesis.proposed_mechanism
        intervention_target = hypothesis.target_candidate.symbol

        # Run counterfactual simulation
        score = await self.inference_client.run_counterfactual_simulation(
            mechanism=mechanism,
            intervention_target=intervention_target,
        )

        logger.info(f"Hypothesis {hypothesis.id} validation score: {score}")

        # Update hypothesis
        hypothesis.causal_validation_score = score
        hypothesis.key_counterfactual = f"Simulated inhibition of {intervention_target} in context of {mechanism}"

        return hypothesis
