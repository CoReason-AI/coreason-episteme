# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List, Protocol

from coreason_episteme.models import Critique, Hypothesis


class ReviewStrategy(Protocol):
    """
    Interface for an adversarial review strategy (The Council Member).
    """

    def review(self, hypothesis: Hypothesis) -> List[Critique]:
        """
        Conducts a review of the hypothesis and returns a list of critiques.
        """
        ...
