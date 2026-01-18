# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from coreason_episteme.interfaces import InferenceClient, SearchClient
from coreason_episteme.models import Hypothesis
from coreason_episteme.utils.logger import logger


class AdversarialReviewerImpl:
    """
    Implementation of the Adversarial Reviewer (The Council).

    Orchestrates reviews from:
    1. The Toxicologist (Safety risks)
    2. The Clinician (Redundancy)
    3. The IP Strategist (Patent infringement)
    """

    def __init__(self, inference_client: InferenceClient, search_client: SearchClient):
        self.inference_client = inference_client
        self.search_client = search_client

    def review(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Conducts an adversarial review of the hypothesis.
        Aggregates critiques and appends them to the hypothesis.
        """
        logger.info(f"Convening Review Board for hypothesis: {hypothesis.id}")

        critiques = []

        # 1. The Toxicologist (Inference)
        logger.debug(f"Running Toxicology Screen for {hypothesis.target_candidate.symbol}...")
        tox_risks = self.inference_client.run_toxicology_screen(hypothesis.target_candidate)
        if tox_risks:
            formatted_risks = [f"[Toxicologist] {risk}" for risk in tox_risks]
            critiques.extend(formatted_risks)
            logger.info(f"Toxicology risks found: {len(tox_risks)}")

        # 2. The Clinician (Inference)
        logger.debug("Checking Clinical Redundancy...")
        redundancies = self.inference_client.check_clinical_redundancy(
            hypothesis.proposed_mechanism, hypothesis.target_candidate
        )
        if redundancies:
            formatted_redundancies = [f"[Clinician] {item}" for item in redundancies]
            critiques.extend(formatted_redundancies)
            logger.info(f"Clinical redundancies found: {len(redundancies)}")

        # 3. The IP Strategist (Search)
        logger.debug("Checking Patent Infringement...")
        patents = self.search_client.check_patent_infringement(
            hypothesis.target_candidate, hypothesis.proposed_mechanism
        )
        if patents:
            formatted_patents = [f"[IP Strategist] Potential conflict: {patent}" for patent in patents]
            critiques.extend(formatted_patents)
            logger.info(f"Patent conflicts found: {len(patents)}")

        # Append to hypothesis
        hypothesis.critiques.extend(critiques)

        if not critiques:
            logger.info(f"Hypothesis {hypothesis.id} survived Adversarial Review with no critiques.")
        else:
            logger.info(f"Hypothesis {hypothesis.id} received {len(critiques)} critiques.")

        return hypothesis
