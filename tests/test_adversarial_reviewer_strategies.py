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

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.components.adversarial_reviewer import AdversarialReviewerImpl
from coreason_episteme.components.strategies import ReviewStrategy
from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    Critique,
    CritiqueSeverity,
    GeneticTarget,
    Hypothesis,
)

# --- Custom Mock Strategies ---


@dataclass
class EmptyStrategy:
    """Strategy that does nothing."""

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        return []


@dataclass
class MalfunctioningStrategy:
    """Strategy that raises an exception."""

    error_message: str = "Strategy Failed"

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        raise ValueError(self.error_message)


@dataclass
class NoneReturningStrategy:
    """Strategy that violates protocol by returning None."""

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        return None  # type: ignore


@dataclass
class CustomCritiqueStrategy:
    """Strategy that returns specific critiques for testing."""

    critiques_to_return: List[Critique]

    async def review(self, hypothesis: Hypothesis) -> List[Critique]:
        return self.critiques_to_return


# --- Fixtures ---


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="test-hypo-strategy",
        title="Strategy Test",
        knowledge_gap="Gap",
        proposed_mechanism="Mech",
        target_candidate=GeneticTarget(symbol="GeneX", ensembl_id="ENSG00X", druggability_score=0.5, novelty_score=0.5),
        causal_validation_score=0.5,
        key_counterfactual="CF",
        killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
        evidence_chain=[],
        confidence=ConfidenceLevel.SPECULATIVE,
        critiques=[],
    )


# --- Tests ---


@pytest.mark.asyncio
async def test_reviewer_with_no_strategies(sample_hypothesis: Hypothesis, user_context: UserContext) -> None:
    """
    Edge Case: Reviewer initialized with empty list of strategies.
    Should run without error and produce 0 critiques.
    """
    reviewer = AdversarialReviewerImpl(strategies=[])
    result = await reviewer.review(sample_hypothesis, context=user_context)
    assert len(result.critiques) == 0


@pytest.mark.asyncio
async def test_reviewer_with_malfunctioning_strategy(sample_hypothesis: Hypothesis, user_context: UserContext) -> None:
    """
    Edge Case: One strategy raises an exception.
    The exception should propagate (Fail Fast).
    """
    strategies: List[ReviewStrategy] = [
        EmptyStrategy(),
        MalfunctioningStrategy(error_message="Boom!"),
        EmptyStrategy(),
    ]
    reviewer = AdversarialReviewerImpl(strategies=strategies)

    with pytest.raises(ValueError, match="Boom!"):
        await reviewer.review(sample_hypothesis, context=user_context)


@pytest.mark.asyncio
async def test_reviewer_with_none_returning_strategy(sample_hypothesis: Hypothesis, user_context: UserContext) -> None:
    """
    Edge Case: Strategy violates protocol and returns None.
    Reviewer expects iterable, so this will likely raise TypeError when extending.
    This test confirms the failure mode (we want it to fail or we need to patch implementation).
    Current implementation uses `extend`, so `None` will cause TypeError.
    """
    strategies: List[ReviewStrategy] = [NoneReturningStrategy()]
    reviewer = AdversarialReviewerImpl(strategies=strategies)

    with pytest.raises(TypeError):
        await reviewer.review(sample_hypothesis, context=user_context)


@pytest.mark.asyncio
async def test_reviewer_complex_mix(sample_hypothesis: Hypothesis, user_context: UserContext) -> None:
    """
    Complex Scenario: Multiple strategies returning mixed critiques.
    Verifies aggregation and order.
    """
    critique_fatal = Critique(source="Strat1", content="Fatal Error", severity=CritiqueSeverity.FATAL)
    critique_warn = Critique(source="Strat2", content="Warning", severity=CritiqueSeverity.MEDIUM)

    strategies: List[ReviewStrategy] = [
        CustomCritiqueStrategy(critiques_to_return=[critique_fatal]),
        EmptyStrategy(),
        CustomCritiqueStrategy(critiques_to_return=[critique_warn]),
    ]

    reviewer = AdversarialReviewerImpl(strategies=strategies)
    result = await reviewer.review(sample_hypothesis, context=user_context)

    assert len(result.critiques) == 2
    assert result.critiques[0] == critique_fatal
    assert result.critiques[1] == critique_warn
    # Verify order is preserved
    assert result.critiques[0].severity == CritiqueSeverity.FATAL
    assert result.critiques[1].severity == CritiqueSeverity.MEDIUM
