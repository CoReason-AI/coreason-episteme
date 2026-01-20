# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from dataclasses import dataclass
from typing import List

from coreason_episteme.interfaces import InferenceClient, SearchClient
from coreason_episteme.models import Critique, CritiqueSeverity, Hypothesis
from coreason_episteme.utils.logger import logger


@dataclass
class AdversarialReviewerImpl:
    """
    Implementation of the Adversarial Reviewer (The Council).

    Orchestrates reviews from:
    1. The Toxicologist (Safety risks)
    2. The Clinician (Redundancy)
    3. The IP Strategist (Patent infringement)
    """

    inference_client: InferenceClient
    search_client: SearchClient

    def _format_critiques(self, items: List[str], source: str, severity: CritiqueSeverity) -> List[Critique]:
        """Helper to format list of strings into Critique objects."""
        return [Critique(source=source, content=item, severity=severity) for item in items]

    def review(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Conducts an adversarial review of the hypothesis.
        Aggregates critiques and appends them to the hypothesis.
        """
        logger.info(f"Convening Review Board for hypothesis: {hypothesis.id}")

        critiques: list[Critique] = []

        # 1. The Toxicologist (Inference)
        logger.debug(f"Running Toxicology Screen for {hypothesis.target_candidate.symbol}...")
        tox_risks = self.inference_client.run_toxicology_screen(hypothesis.target_candidate)
        if tox_risks:
            critiques.extend(self._format_critiques(tox_risks, "Toxicologist", CritiqueSeverity.FATAL))
            logger.info(f"Toxicology risks found: {len(tox_risks)}")

        # 2. The Clinician (Inference)
        logger.debug("Checking Clinical Redundancy...")
        redundancies = self.inference_client.check_clinical_redundancy(
            hypothesis.proposed_mechanism, hypothesis.target_candidate
        )
        if redundancies:
            critiques.extend(self._format_critiques(redundancies, "Clinician", CritiqueSeverity.MEDIUM))
            logger.info(f"Clinical redundancies found: {len(redundancies)}")

        # 3. The IP Strategist (Search)
        logger.debug("Checking Patent Infringement...")
        patents = self.search_client.check_patent_infringement(
            hypothesis.target_candidate, hypothesis.proposed_mechanism
        )
        if patents:
            critiques.extend(self._format_critiques(patents, "IP Strategist", CritiqueSeverity.HIGH))
            logger.info(f"Patent conflicts found: {len(patents)}")

        # 4. The Scientific Skeptic (Strict Null Hypothesis)
        logger.debug("Searching for Disconfirming Evidence (Null Hypothesis Check)...")
        # Attempt to parse subject/object from hypothesis or use best-effort mapping
        subject = hypothesis.target_candidate.symbol
        object_context = hypothesis.knowledge_gap
        action = "affect"

        disconfirming_evidence = self.search_client.find_disconfirming_evidence(
            subject=subject, object=object_context, action=action
        )

        if disconfirming_evidence:
            # We use a slightly different content format for evidence
            formatted_evidence = [f"Disconfirming evidence found: {e}" for e in disconfirming_evidence]
            critiques.extend(
                self._format_critiques(formatted_evidence, "Scientific Skeptic", CritiqueSeverity.FATAL)
            )
            logger.info(f"Disconfirming evidence found: {len(disconfirming_evidence)}")

        # Append to hypothesis
        hypothesis.critiques.extend(critiques)

        if not critiques:
            logger.info(f"Hypothesis {hypothesis.id} survived Adversarial Review with no critiques.")
        else:
            logger.info(f"Hypothesis {hypothesis.id} received {len(critiques)} critiques.")

        return hypothesis
