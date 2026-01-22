# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import pytest

from coreason_episteme.components.protocol_designer import ProtocolDesignerImpl
from coreason_episteme.models import PICO, ConfidenceLevel, GeneticTarget, Hypothesis


@pytest.fixture
def protocol_designer() -> ProtocolDesignerImpl:
    return ProtocolDesignerImpl()


@pytest.fixture
def sample_hypothesis() -> Hypothesis:
    target = GeneticTarget(
        symbol="TargetX",
        ensembl_id="ENSG00000X",
        druggability_score=0.9,
        novelty_score=0.8,
    )
    return Hypothesis(
        id="test-hyp-1",
        title="Test Hypothesis",
        knowledge_gap="Gap description",
        proposed_mechanism="Mechanism Y",
        target_candidate=target,
        causal_validation_score=0.8,
        key_counterfactual="Counterfactual Z",
        killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
        evidence_chain=[],
        confidence=ConfidenceLevel.PLAUSIBLE,
    )


@pytest.mark.asyncio
async def test_design_experiment_populates_pico(
    protocol_designer: ProtocolDesignerImpl, sample_hypothesis: Hypothesis
) -> None:
    """Test that the experiment design populates the PICO fields correctly."""
    result = await protocol_designer.design_experiment(sample_hypothesis)

    assert result.killer_experiment_pico is not None
    pico = result.killer_experiment_pico

    # Assert fields are populated (not empty strings)
    assert pico.population
    assert pico.intervention
    assert pico.comparator
    assert pico.outcome

    assert "Mechanism Y" in pico.population
    assert "TargetX" in pico.intervention
    assert "Vehicle control" in pico.comparator
    assert "Mechanism Y" in pico.outcome


@pytest.mark.asyncio
async def test_design_experiment_returns_hypothesis(
    protocol_designer: ProtocolDesignerImpl, sample_hypothesis: Hypothesis
) -> None:
    """Test that the method returns the modified hypothesis object."""
    result = await protocol_designer.design_experiment(sample_hypothesis)
    assert result is sample_hypothesis
