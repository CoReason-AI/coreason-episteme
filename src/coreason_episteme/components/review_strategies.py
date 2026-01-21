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
Strategies implementation for the Adversarial Reviewer.

This module contains concrete implementations of the `ReviewStrategy` protocol,
each representing a different perspective (persona) in the adversarial review process.
"""

from dataclasses import dataclass
from typing import List

from coreason_episteme.interfaces import InferenceClient, SearchClient
from coreason_episteme.models import Critique, CritiqueSeverity, Hypothesis
from coreason_episteme.utils.logger import logger


def _format_critiques(items: List[str], source: str, severity: CritiqueSeverity) -> List[Critique]:
    """
    Helper to format list of strings into Critique objects.

    Args:
        items: List of critique content strings.
        source: The source of the critique (e.g., "The Toxicologist").
        severity: The severity of the critiques.

    Returns:
        List[Critique]: A list of Critique objects.
    """
    return [Critique(source=source, content=item, severity=severity) for item in items]


@dataclass
class ToxicologyStrategy:
    """
    The Toxicologist Strategy.

    Checks for safety risks associated with the proposed target.
    Critiques are typically fatal if toxicity is high.

    Attributes:
        inference_client: Client for running toxicology screens.
    """

    inference_client: InferenceClient

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        """
        Runs the toxicology screen and returns critiques.

        Args:
            hypothesis: The hypothesis to review.

        Returns:
            List[Critique]: A list of critiques related to toxicology.
        """
        logger.debug(f"Running Toxicology Screen for {hypothesis.target_candidate.symbol}...")
        tox_risks = await self.inference_client.run_toxicology_screen(hypothesis.target_candidate)
        if tox_risks:
            logger.info(f"Toxicology risks found: {len(tox_risks)}")
            return _format_critiques(tox_risks, "Toxicologist", CritiqueSeverity.FATAL)
        return []


@dataclass
class ClinicalRedundancyStrategy:
    """
    The Clinician Strategy.

    Checks if the proposed hypothesis is redundant with existing interventions
    or established clinical knowledge.

    Attributes:
        inference_client: Client for checking clinical redundancy.
    """

    inference_client: InferenceClient

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        """
        Checks for clinical redundancy and returns critiques.

        Args:
            hypothesis: The hypothesis to review.

        Returns:
            List[Critique]: A list of critiques related to clinical redundancy.
        """
        logger.debug("Checking Clinical Redundancy...")
        redundancies = await self.inference_client.check_clinical_redundancy(
            hypothesis.proposed_mechanism, hypothesis.target_candidate
        )
        if redundancies:
            logger.info(f"Clinical redundancies found: {len(redundancies)}")
            return _format_critiques(redundancies, "Clinician", CritiqueSeverity.MEDIUM)
        return []


@dataclass
class PatentStrategy:
    """
    The IP Strategist.

    Checks for patent infringement or "Freedom to Operate" issues.

    Attributes:
        search_client: Client for searching patent databases.
    """

    search_client: SearchClient

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        """
        Checks for patent conflicts and returns critiques.

        Args:
            hypothesis: The hypothesis to review.

        Returns:
            List[Critique]: A list of critiques related to patent issues.
        """
        logger.debug("Checking Patent Infringement...")
        patents = await self.search_client.check_patent_infringement(
            hypothesis.target_candidate, hypothesis.proposed_mechanism
        )
        if patents:
            logger.info(f"Patent conflicts found: {len(patents)}")
            return _format_critiques(patents, "IP Strategist", CritiqueSeverity.HIGH)
        return []


@dataclass
class ScientificSkepticStrategy:
    """
    The Scientific Skeptic.

    Checks for evidence that explicitly disconfirms the hypothesis (Null Hypothesis check).

    Attributes:
        search_client: Client for finding disconfirming evidence in literature.
    """

    search_client: SearchClient

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        """
        Searches for disconfirming evidence and returns critiques.

        Args:
            hypothesis: The hypothesis to review.

        Returns:
            List[Critique]: A list of critiques if disconfirming evidence is found.
        """
        logger.debug("Searching for Disconfirming Evidence (Null Hypothesis Check)...")
        # Attempt to parse subject/object from hypothesis or use best-effort mapping
        subject = hypothesis.target_candidate.symbol
        object_context = hypothesis.knowledge_gap
        action = "affect"

        disconfirming_evidence = await self.search_client.find_disconfirming_evidence(
            subject=subject, object=object_context, action=action
        )

        if disconfirming_evidence:
            # We use a slightly different content format for evidence
            formatted_evidence = [f"Disconfirming evidence found: {e}" for e in disconfirming_evidence]
            logger.info(f"Disconfirming evidence found: {len(disconfirming_evidence)}")
            return _format_critiques(formatted_evidence, "Scientific Skeptic", CritiqueSeverity.FATAL)
        return []
