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
        critiques=[],
    )


def test_review_service_returns_none(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Edge Case: Service returns None instead of empty list.
    The system should handle this gracefully without crashing.
    """
    # Simulate a service (Inference Client) behaving badly and returning None
    mock_inference_client.run_toxicology_screen.return_value = None
    mock_inference_client.check_clinical_redundancy.return_value = None
    mock_search_client.check_patent_infringement.return_value = None

    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Should have 0 critiques, not crash
    assert len(reviewed_hypothesis.critiques) == 0


def test_review_high_volume_critiques(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Complex Scenario: Large number of critiques.
    Verifies that the system can handle a high volume of feedback.
    """
    # Simulate massive failure
    mock_inference_client.run_toxicology_screen.return_value = [f"Risk {i}" for i in range(50)]
    mock_inference_client.check_clinical_redundancy.return_value = [f"Redundancy {i}" for i in range(50)]
    mock_search_client.check_patent_infringement.return_value = [f"Patent {i}" for i in range(50)]

    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Should have 150 critiques
    assert len(reviewed_hypothesis.critiques) == 150
    # Updated assertions to check Critique object content
    assert any("Risk 49" in c.content and c.source == "Toxicologist" for c in reviewed_hypothesis.critiques)
    assert any("Redundancy 0" in c.content and c.source == "Clinician" for c in reviewed_hypothesis.critiques)


def test_review_exception_propagation(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Edge Case: Service raises an exception.
    The current behavior is to propagate the exception (Fail Fast).
    """
    mock_inference_client.run_toxicology_screen.side_effect = RuntimeError("Service Down")

    with pytest.raises(RuntimeError, match="Service Down"):
        adversarial_reviewer.review(sample_hypothesis)


def test_review_cumulative_critiques(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test State: Calling review multiple times appends critiques.
    This verifies that previous critiques are not overwritten.
    """
    # First run
    mock_inference_client.run_toxicology_screen.return_value = ["Risk A"]
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []

    hypo_v1 = adversarial_reviewer.review(sample_hypothesis)
    assert len(hypo_v1.critiques) == 1

    # Second run (maybe different results?)
    mock_inference_client.run_toxicology_screen.return_value = ["Risk B"]

    hypo_v2 = adversarial_reviewer.review(hypo_v1)

    # Should have 2 critiques now (Risk A + Risk B)
    assert len(hypo_v2.critiques) == 2
    assert any("Risk A" in c.content for c in hypo_v2.critiques)
    assert any("Risk B" in c.content for c in hypo_v2.critiques)


def test_review_scientific_skeptic_failure_handling(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Edge Case: Scientific Skeptic returns None or invalid data.
    The system should treat None as 'no evidence found' (similar to empty list).
    """
    mock_inference_client.run_toxicology_screen.return_value = []
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []

    # Simulate None return from search client
    mock_search_client.find_disconfirming_evidence.return_value = None

    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Should have 0 critiques
    assert len(reviewed_hypothesis.critiques) == 0


def test_review_all_reviewers_trigger(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Complex Scenario: Every single reviewer finds an issue.
    Ensures that the hypothesis accumulates all types of critiques correctly.
    """
    mock_inference_client.run_toxicology_screen.return_value = ["Toxic"]
    mock_inference_client.check_clinical_redundancy.return_value = ["Redundant"]
    mock_search_client.check_patent_infringement.return_value = ["Patent Breach"]
    mock_search_client.find_disconfirming_evidence.return_value = ["Paper says NO"]

    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    assert len(reviewed_hypothesis.critiques) == 4

    critique_sources = {c.source for c in reviewed_hypothesis.critiques}
    assert "Toxicologist" in critique_sources
    assert "Clinician" in critique_sources
    assert "IP Strategist" in critique_sources
    assert "Scientific Skeptic" in critique_sources


def test_review_empty_strings_robustness(
    adversarial_reviewer: AdversarialReviewerImpl,
    mock_inference_client: MagicMock,
    mock_search_client: MagicMock,
    sample_hypothesis: Hypothesis,
) -> None:
    """
    Test Edge Case: Empty strings in hypothesis fields used for search.
    """
    mock_inference_client.run_toxicology_screen.return_value = []
    mock_inference_client.check_clinical_redundancy.return_value = []
    mock_search_client.check_patent_infringement.return_value = []
    mock_search_client.find_disconfirming_evidence.return_value = []

    # Set fields to empty strings
    sample_hypothesis.target_candidate.symbol = ""
    sample_hypothesis.knowledge_gap = ""

    reviewed_hypothesis = adversarial_reviewer.review(sample_hypothesis)

    # Verify call was still made with empty strings
    mock_search_client.find_disconfirming_evidence.assert_called_once_with(subject="", object="", action="affect")
    assert len(reviewed_hypothesis.critiques) == 0
