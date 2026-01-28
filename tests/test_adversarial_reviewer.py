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
from unittest.mock import AsyncMock

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.components.adversarial_reviewer import AdversarialReviewerImpl
from coreason_episteme.components.review_strategies import (
    ClinicalRedundancyStrategy,
    PatentStrategy,
    ScientificSkepticStrategy,
    ToxicologyStrategy,
)
from coreason_episteme.components.strategies import ReviewStrategy
from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    CritiqueSeverity,
    GeneticTarget,
    Hypothesis,
)


@pytest.fixture
def mock_inference_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_search_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


@pytest.fixture
def adversarial_reviewer(mock_inference_client: AsyncMock, mock_search_client: AsyncMock) -> AdversarialReviewerImpl:
    strategies: List[ReviewStrategy] = [
        ToxicologyStrategy(inference_client=mock_inference_client),
        ClinicalRedundancyStrategy(inference_client=mock_inference_client),
        PatentStrategy(search_client=mock_search_client),
        ScientificSkepticStrategy(search_client=mock_search_client),
    ]
    return AdversarialReviewerImpl(strategies=strategies)


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    target = GeneticTarget(
        symbol="GeneA",
        ensembl_id="ENSG001",
        druggability_score=0.9,
        novelty_score=0.8,
    )
    pico = PICO(
        population="Pop",
        intervention="Int",
        comparator="Comp",
        outcome="Out",
    )
    return Hypothesis(
        id="hypo-1",
        title="Test Hypothesis",
        knowledge_gap="Gap",
        proposed_mechanism="Mech",
        target_candidate=target,
        causal_validation_score=0.8,
        key_counterfactual="CF",
        killer_experiment_pico=pico,
        evidence_chain=[],
        confidence=ConfidenceLevel.PLAUSIBLE,
    )


@pytest.mark.asyncio
async def test_review_clean_pass(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: AsyncMock,
    mock_search_client: AsyncMock,
    sample_hypothesis: Hypothesis,
    user_context: UserContext,
) -> None:
    """Test a review where no issues are found (Clean Pass)."""
    # Setup mocks to return empty lists (no issues)
    mock_inference_client.run_toxicology_screen.return_value = []
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []
    mock_search_client.find_disconfirming_evidence.return_value = []

    # Execute
    reviewed_hypothesis = await adversarial_reviewer.review(sample_hypothesis, context=user_context)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 0
    mock_inference_client.run_toxicology_screen.assert_called_once_with(sample_hypothesis.target_candidate)
    mock_inference_client.check_clinical_redundancy.assert_called_once()
    mock_search_client.check_patent_infringement.assert_called_once()
    mock_search_client.find_disconfirming_evidence.assert_called_once()


@pytest.mark.asyncio
async def test_review_toxicology_fail(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: AsyncMock,
    mock_search_client: AsyncMock,
    sample_hypothesis: Hypothesis,
    user_context: UserContext,
) -> None:
    """Test a review where Toxicology finds risks."""
    # Setup mocks
    mock_inference_client.run_toxicology_screen.return_value = ["Liver Toxicity Risk"]
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []
    mock_search_client.find_disconfirming_evidence.return_value = []

    # Execute
    reviewed_hypothesis = await adversarial_reviewer.review(sample_hypothesis, context=user_context)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 1
    critique = reviewed_hypothesis.critiques[0]
    assert critique.source == "Toxicologist"
    assert critique.content == "Liver Toxicity Risk"
    assert critique.severity == CritiqueSeverity.FATAL


@pytest.mark.asyncio
async def test_review_skeptic_fail(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: AsyncMock,
    mock_search_client: AsyncMock,
    sample_hypothesis: Hypothesis,
    user_context: UserContext,
) -> None:
    """Test a review where the Scientific Skeptic finds disconfirming evidence."""
    # Setup mocks
    mock_inference_client.run_toxicology_screen.return_value = []
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []
    mock_search_client.find_disconfirming_evidence.return_value = [
        "Paper X (2020) explicitly states GeneA has no effect on Pathway Y."
    ]

    # Execute
    reviewed_hypothesis = await adversarial_reviewer.review(sample_hypothesis, context=user_context)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 1
    critique = reviewed_hypothesis.critiques[0]
    assert critique.source == "Scientific Skeptic"
    assert "Paper X (2020)" in critique.content
    assert critique.severity == CritiqueSeverity.FATAL

    # Check arguments
    mock_search_client.find_disconfirming_evidence.assert_called_once_with(
        subject="GeneA", object="Gap", action="affect"
    )


@pytest.mark.asyncio
async def test_review_multiple_critiques(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: AsyncMock,
    mock_search_client: AsyncMock,
    sample_hypothesis: Hypothesis,
    user_context: UserContext,
) -> None:
    """Test a review where multiple reviewers find issues."""
    # Setup mocks
    mock_inference_client.run_toxicology_screen.return_value = ["Cardio Risk"]
    mock_inference_client.check_clinical_redundancy.return_value = ["Similar drug in Phase 2"]
    mock_search_client.check_patent_infringement.return_value = ["US Patent 12345"]
    mock_search_client.find_disconfirming_evidence.return_value = ["Contradictory Study Z"]

    # Execute
    reviewed_hypothesis = await adversarial_reviewer.review(sample_hypothesis, context=user_context)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 4

    # Check severities
    critique_map = {c.source: c.severity for c in reviewed_hypothesis.critiques}
    assert critique_map["Toxicologist"] == CritiqueSeverity.FATAL
    assert critique_map["Clinician"] == CritiqueSeverity.MEDIUM
    assert critique_map["IP Strategist"] == CritiqueSeverity.HIGH
    assert critique_map["Scientific Skeptic"] == CritiqueSeverity.FATAL
