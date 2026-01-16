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

from coreason_episteme.models import KnowledgeGap
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
            source_nodes=["PMID:123456", "PMID:789012"],
        )
        return [gap]
