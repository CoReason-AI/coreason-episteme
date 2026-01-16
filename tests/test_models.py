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
from pydantic import ValidationError

from coreason_episteme.models import ConfidenceLevel, GeneticTarget, Hypothesis, KnowledgeGap


def test_genetic_target_creation() -> None:
    target = GeneticTarget(
        symbol="TP53",
        ensembl_id="ENSG00000141510",
        druggability_score=0.95,
        novelty_score=0.1,
    )
    assert target.symbol == "TP53"
    assert target.ensembl_id == "ENSG00000141510"
    assert target.druggability_score == 0.95
    assert target.novelty_score == 0.1


def test_confidence_level_enum() -> None:
    assert ConfidenceLevel.SPECULATIVE == "SPECULATIVE"
    assert ConfidenceLevel.PLAUSIBLE == "PLAUSIBLE"
    assert ConfidenceLevel.PROBABLE == "PROBABLE"


def test_hypothesis_creation() -> None:
    target = GeneticTarget(
        symbol="TP53",
        ensembl_id="ENSG00000141510",
        druggability_score=0.95,
        novelty_score=0.1,
    )

    hypothesis = Hypothesis(
        id="hyp-123",
        title="Test Hypothesis",
        knowledge_gap="Gap description",
        proposed_mechanism="Mechanism description",
        target_candidate=target,
        causal_validation_score=0.8,
        key_counterfactual="If X then Y",
        killer_experiment_pico={"P": "Population", "I": "Intervention"},
        evidence_chain=["Paper 1", "Paper 2"],
        confidence=ConfidenceLevel.PLAUSIBLE,
    )

    assert hypothesis.id == "hyp-123"
    assert hypothesis.target_candidate.symbol == "TP53"
    assert hypothesis.confidence == ConfidenceLevel.PLAUSIBLE
    assert hypothesis.killer_experiment_pico == {"P": "Population", "I": "Intervention"}


def test_hypothesis_validation_error() -> None:
    # Missing required fields
    with pytest.raises(ValidationError):
        Hypothesis(  # type: ignore
            id="hyp-123",
            title="Incomplete",
            # Missing other fields
        )


def test_knowledge_gap_creation() -> None:
    gap = KnowledgeGap(description="Something missing", source_nodes=["NodeA", "NodeB"])
    assert gap.description == "Something missing"
    assert gap.source_nodes == ["NodeA", "NodeB"]

    gap_optional = KnowledgeGap(description="Only description")
    assert gap_optional.description == "Only description"
    assert gap_optional.source_nodes is None


def test_serialization() -> None:
    target = GeneticTarget(
        symbol="TP53",
        ensembl_id="ENSG00000141510",
        druggability_score=0.95,
        novelty_score=0.1,
    )
    json_str = target.model_dump_json()
    assert "TP53" in json_str

    loaded = GeneticTarget.model_validate_json(json_str)
    assert loaded.symbol == target.symbol
