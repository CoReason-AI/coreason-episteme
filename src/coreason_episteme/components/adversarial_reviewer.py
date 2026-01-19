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
from coreason_episteme.models import Critique, CritiqueSeverity, Hypothesis
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

        critiques: list[Critique] = []

        # 1. The Toxicologist (Inference)
        logger.debug(f"Running Toxicology Screen for {hypothesis.target_candidate.symbol}...")
        tox_risks = self.inference_client.run_toxicology_screen(hypothesis.target_candidate)
        if tox_risks:
            formatted_risks = [
                Critique(source="Toxicologist", content=risk, severity=CritiqueSeverity.FATAL) for risk in tox_risks
            ]
            critiques.extend(formatted_risks)
            logger.info(f"Toxicology risks found: {len(tox_risks)}")

        # 2. The Clinician (Inference)
        logger.debug("Checking Clinical Redundancy...")
        redundancies = self.inference_client.check_clinical_redundancy(
            hypothesis.proposed_mechanism, hypothesis.target_candidate
        )
        if redundancies:
            formatted_redundancies = [
                Critique(source="Clinician", content=item, severity=CritiqueSeverity.MEDIUM) for item in redundancies
            ]
            critiques.extend(formatted_redundancies)
            logger.info(f"Clinical redundancies found: {len(redundancies)}")

        # 3. The IP Strategist (Search)
        logger.debug("Checking Patent Infringement...")
        patents = self.search_client.check_patent_infringement(
            hypothesis.target_candidate, hypothesis.proposed_mechanism
        )
        if patents:
            formatted_patents = [
                Critique(source="IP Strategist", content=patent, severity=CritiqueSeverity.HIGH) for patent in patents
            ]
            critiques.extend(formatted_patents)
            logger.info(f"Patent conflicts found: {len(patents)}")

        # 4. The Scientific Skeptic (Strict Null Hypothesis)
        logger.debug("Searching for Disconfirming Evidence (Null Hypothesis Check)...")
        # Attempt to parse subject/object from hypothesis or use best-effort mapping
        # Subject: Intervention Target
        # Object: Disease/Outcome (from knowledge gap or mechanism context)
        # Action: "regulate" or "affect"
        subject = hypothesis.target_candidate.symbol
        # We try to extract the object from the mechanism or gap description.
        # For robustness, we'll use the mechanism summary as the context.
        object_context = hypothesis.knowledge_gap
        action = "affect"

        disconfirming_evidence = self.search_client.find_disconfirming_evidence(
            subject=subject, object=object_context, action=action
        )

        if disconfirming_evidence:
            formatted_skepticism = [
                Critique(
                    source="Scientific Skeptic",
                    content=f"Disconfirming evidence found: {evidence}",
                    severity=CritiqueSeverity.FATAL,
                )
                for evidence in disconfirming_evidence
            ]
            critiques.extend(formatted_skepticism)
            logger.info(f"Disconfirming evidence found: {len(disconfirming_evidence)}")

        # Append to hypothesis
        hypothesis.critiques.extend(critiques)

        if not critiques:
            logger.info(f"Hypothesis {hypothesis.id} survived Adversarial Review with no critiques.")
        else:
            logger.info(f"Hypothesis {hypothesis.id} received {len(critiques)} critiques.")

        return hypothesis
