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
Adversarial Reviewer component implementation.

This module implements the `AdversarialReviewerImpl`, which aggregates multiple
review strategies to critique generated hypotheses.
"""

from dataclasses import dataclass, field
from typing import List

from coreason_identity.models import UserContext

from coreason_episteme.components.strategies import ReviewStrategy
from coreason_episteme.models import Critique, Hypothesis
from coreason_episteme.utils.logger import logger


@dataclass
class AdversarialReviewerImpl:
    """
    Implementation of the Adversarial Reviewer (The Council).

    Orchestrates reviews from various strategies to critique the hypothesis
    from multiple perspectives (Toxicology, Clinical, IP, etc.).

    Attributes:
        strategies: A list of ReviewStrategy implementations to apply.
    """

    strategies: List[ReviewStrategy] = field(default_factory=list)

    async def review(self, hypothesis: Hypothesis, context: UserContext) -> Hypothesis:
        """
        Conducts an adversarial review of the hypothesis.

        Iterates through all configured strategies, aggregates their critiques,
        and appends them to the hypothesis.

        Args:
            hypothesis: The hypothesis to review.
            context: The user context triggering the review.

        Returns:
            Hypothesis: The hypothesis object with appended critiques.
        """
        logger.info(
            f"Convening Review Board for hypothesis: {hypothesis.id}",
            user_id=context.user_id,
        )

        critiques: list[Critique] = []

        for strategy in self.strategies:
            critiques.extend(await strategy.review(hypothesis))

        # Append to hypothesis
        hypothesis.critiques.extend(critiques)

        if not critiques:
            logger.info(f"Hypothesis {hypothesis.id} survived Adversarial Review with no critiques.")
        else:
            logger.info(f"Hypothesis {hypothesis.id} received {len(critiques)} critiques.")

        return hypothesis
