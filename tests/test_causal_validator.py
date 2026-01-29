# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import AsyncMock

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.components.causal_validator import CausalValidatorImpl
from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
)


@pytest.fixture
def mock_inference_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        user_id="test-user",
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


@pytest.fixture
def causal_validator(mock_inference_client: AsyncMock) -> CausalValidatorImpl:
    return CausalValidatorImpl(inference_client=mock_inference_client)


@pytest.mark.asyncio
async def test_validate_success(
    causal_validator: CausalValidatorImpl,
    mock_inference_client: AsyncMock,
    user_context: UserContext,
) -> None:
    """Test successful hypothesis validation."""
    # Setup hypothesis
    target = GeneticTarget(
        symbol="GeneX",
        ensembl_id="ENSG000001",
        druggability_score=0.9,
        novelty_score=0.8,
    )
    hypothesis = Hypothesis(
        id="hypo-123",
        title="Test Hypothesis",
        knowledge_gap="Gap Description",
        proposed_mechanism="Mechanism A -> B -> C",
        target_candidate=target,
        causal_validation_score=0.0,
        key_counterfactual="",
        killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
        evidence_chain=[],
        confidence=ConfidenceLevel.SPECULATIVE,
    )

    # Setup mock return
    mock_inference_client.run_counterfactual_simulation.return_value = 0.85

    # Execute
    validated_hypothesis = await causal_validator.validate(hypothesis, context=user_context)

    # Verify
    assert validated_hypothesis.causal_validation_score == 0.85
    assert "GeneX" in validated_hypothesis.key_counterfactual
    assert "Mechanism A -> B -> C" in validated_hypothesis.key_counterfactual

    # Verify call
    mock_inference_client.run_counterfactual_simulation.assert_called_with(
        mechanism="Mechanism A -> B -> C",
        intervention_target="GeneX",
    )
