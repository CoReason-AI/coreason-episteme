# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import MagicMock

import pytest

from coreason_episteme.components.adversarial_reviewer import AdversarialReviewerImpl
from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
)


@pytest.fixture  # type: ignore[misc]
def mock_inference_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def mock_search_client() -> MagicMock:
    return MagicMock()


@pytest.fixture  # type: ignore[misc]
def adversarial_reviewer(mock_inference_client: MagicMock, mock_search_client: MagicMock) -> AdversarialReviewerImpl:
    return AdversarialReviewerImpl(
        inference_client=mock_inference_client,
        search_client=mock_search_client,
    )


@pytest.fixture  # type: ignore[misc]
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


def test_review_clean_pass(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """Test a review where no issues are found (Clean Pass)."""
    # Setup mocks to return empty lists (no issues)
    mock_inference_client.run_toxicology_screen.return_value = []
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []

    # Execute
    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 0
    mock_inference_client.run_toxicology_screen.assert_called_once_with(sample_hypothesis.target_candidate)
    mock_inference_client.check_clinical_redundancy.assert_called_once()
    mock_search_client.check_patent_infringement.assert_called_once()


def test_review_toxicology_fail(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """Test a review where Toxicology finds risks."""
    # Setup mocks
    mock_inference_client.run_toxicology_screen.return_value = ["Liver Toxicity Risk"]
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []

    # Execute
    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 1
    assert "[Toxicologist] Liver Toxicity Risk" in reviewed_hypothesis.critiques[0]


def test_review_multiple_critiques(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """Test a review where multiple reviewers find issues."""
    # Setup mocks
    mock_inference_client.run_toxicology_screen.return_value = ["Cardio Risk"]
    mock_inference_client.check_clinical_redundancy.return_value = ["Similar drug in Phase 2"]
    mock_search_client.check_patent_infringement.return_value = ["US Patent 12345"]

    # Execute
    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Verify
    assert len(reviewed_hypothesis.critiques) == 3
    critique_texts = " ".join(reviewed_hypothesis.critiques)
    assert "[Toxicologist]" in critique_texts
    assert "[Clinician]" in critique_texts
    assert "[IP Strategist]" in critique_texts
