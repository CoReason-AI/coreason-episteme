# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from coreason_episteme.interfaces import InferenceClient
from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


class CausalValidatorImpl:
    """Implementation of the Causal Validator (The Simulator)."""

    def __init__(self, inference_client: InferenceClient):
        self.inference_client = inference_client

    def validate(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Validates the hypothesis using causal simulation.
        Updates the hypothesis with validation score.
        """
        logger.info(f"Validating hypothesis: {hypothesis.id}")

        mechanism = hypothesis.proposed_mechanism
        intervention_target = hypothesis.target_candidate.symbol

        # Run counterfactual simulation
        score = self.inference_client.run_counterfactual_simulation(
            mechanism=mechanism,
            intervention_target=intervention_target,
        )

        logger.info(f"Hypothesis {hypothesis.id} validation score: {score}")

        # Update hypothesis
        hypothesis.causal_validation_score = score
        hypothesis.key_counterfactual = f"Simulated inhibition of {intervention_target} in context of {mechanism}"

        return hypothesis
