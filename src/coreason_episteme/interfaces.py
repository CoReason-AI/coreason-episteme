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

from coreason_episteme.models import KnowledgeGap


class GapScanner(Protocol):
    """Interface for components that identify knowledge gaps."""

    def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Scans for knowledge gaps related to the target.

        Args:
            target: The disease or biological entity to scan for.

        Returns:
            A list of KnowledgeGap objects.
        """
        ...
