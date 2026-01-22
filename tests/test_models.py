# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import json

import pytest
from pydantic import ValidationError

from coreason_episteme.models import (
    PICO,
    ConfidenceLevel,
    GeneticTarget,
    Hypothesis,
    KnowledgeGap,
    KnowledgeGapType,
)


def test_pico_model_valid() -> None:
    """Test creating a valid PICO object."""
    pico = PICO(
        population="Patients with Type 2 Diabetes",
        intervention="Inhibition of Gene X",
        comparator="Standard of Care (Metformin)",
        outcome="Reduction in HbA1c",
    )
    assert pico.population == "Patients with Type 2 Diabetes"
    assert pico.intervention == "Inhibition of Gene X"


def test_pico_model_invalid_missing_field() -> None:
    """Test PICO validation failure on missing field."""
    with pytest.raises(ValidationError):
        PICO(
            population="Patients",
            intervention="Drug",
            # Missing comparator
            outcome="Health",  # type: ignore[call-arg]
        )


def test_pico_equality() -> None:
    """Test that two identical PICO objects are equal."""
    pico1 = PICO(population="P", intervention="I", comparator="C", outcome="O")
    pico2 = PICO(population="P", intervention="I", comparator="C", outcome="O")
    pico3 = PICO(population="X", intervention="I", comparator="C", outcome="O")

    assert pico1 == pico2
    assert pico1 != pico3


def test_knowledge_gap_model_valid() -> None:
    """Test creating a valid KnowledgeGap object with type."""
    gap = KnowledgeGap(
        description="Disconnect between Protein A and Disease B",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
        source_nodes=["Protein A", "Disease B"],
    )
    assert gap.type == KnowledgeGapType.CLUSTER_DISCONNECT
    assert gap.description == "Disconnect between Protein A and Disease B"
    assert gap.id is not None  # ID should be auto-generated


def test_knowledge_gap_id_generation() -> None:
    """Test that KnowledgeGap generates unique IDs."""
    gap1 = KnowledgeGap(
        description="Desc1",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )
    gap2 = KnowledgeGap(
        description="Desc2",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )
    assert gap1.id is not None
    assert gap2.id is not None
    assert gap1.id != gap2.id


def test_knowledge_gap_model_invalid_type() -> None:
    """Test KnowledgeGap validation with invalid type."""
    with pytest.raises(ValidationError):
        KnowledgeGap(
            description="Bad Gap",
            type="UNKNOWN_TYPE",
        )


def test_knowledge_gap_edge_cases() -> None:
    """Test edge cases for KnowledgeGap."""
    # Test with empty source_nodes list
    gap_empty = KnowledgeGap(
        description="Desc",
        type=KnowledgeGapType.LITERATURE_INCONSISTENCY,
        source_nodes=[],
    )
    assert gap_empty.source_nodes == []

    # Test with None source_nodes
    gap_none = KnowledgeGap(
        description="Desc",
        type=KnowledgeGapType.LITERATURE_INCONSISTENCY,
        source_nodes=None,
    )
    assert gap_none.source_nodes is None

    # Test with empty description (valid string)
    gap_desc = KnowledgeGap(
        description="",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
    )
    assert gap_desc.description == ""


def test_hypothesis_model_valid() -> None:
    """Test creating a valid Hypothesis object with nested PICO."""
    target = GeneticTarget(
        symbol="GENE-X",
        ensembl_id="ENSG000001",
        druggability_score=0.85,
        novelty_score=0.9,
    )
    pico = PICO(
        population="Pop",
        intervention="Int",
        comparator="Comp",
        outcome="Out",
    )
    hypothesis = Hypothesis(
        id="HYP-123",
        title="Test Hypothesis",
        knowledge_gap="Gap Description",
        proposed_mechanism="Pathway Y",
        target_candidate=target,
        causal_validation_score=0.75,
        key_counterfactual="If not X, then Y",
        killer_experiment_pico=pico,
        evidence_chain=["Paper 1", "Node 2"],
        confidence=ConfidenceLevel.PLAUSIBLE,
    )

    assert hypothesis.killer_experiment_pico.population == "Pop"
    assert hypothesis.confidence == ConfidenceLevel.PLAUSIBLE


def test_hypothesis_model_invalid_nested_pico() -> None:
    """Test Hypothesis validation when PICO is invalid (e.g. dict instead of object)."""
    target = GeneticTarget(
        symbol="GENE-X",
        ensembl_id="ENSG000001",
        druggability_score=0.85,
        novelty_score=0.9,
    )

    # Passing a dict should actually work if Pydantic can coerce it,
    # but let's test a structurally invalid dict (missing field)
    with pytest.raises(ValidationError):
        Hypothesis(
            id="HYP-123",
            title="Test Hypothesis",
            knowledge_gap="Gap",
            proposed_mechanism="Mech",
            target_candidate=target,
            causal_validation_score=0.5,
            key_counterfactual="Counter",
            killer_experiment_pico={
                "population": "Pop",
                "intervention": "Int",
                # Missing comparator and outcome
            },
            evidence_chain=[],
            confidence=ConfidenceLevel.SPECULATIVE,
        )


def test_hypothesis_serialization() -> None:
    """Test complex object serialization and deserialization."""
    target = GeneticTarget(
        symbol="TargetZ",
        ensembl_id="ENSG00000Z",
        druggability_score=0.1,
        novelty_score=0.2,
    )
    pico = PICO(
        population="PopZ",
        intervention="IntZ",
        comparator="CompZ",
        outcome="OutZ",
    )
    original_hyp = Hypothesis(
        id="HYP-JSON",
        title="JSON Test",
        knowledge_gap="Gap JSON",
        proposed_mechanism="Mech JSON",
        target_candidate=target,
        causal_validation_score=0.99,
        key_counterfactual="Counter Z",
        killer_experiment_pico=pico,
        evidence_chain=["E1", "E2"],
        confidence=ConfidenceLevel.PROBABLE,
    )

    # Serialize to JSON
    json_str = original_hyp.model_dump_json()

    # Deserialize back
    loaded_hyp = Hypothesis.model_validate_json(json_str)

    assert loaded_hyp == original_hyp
    assert loaded_hyp.killer_experiment_pico.population == "PopZ"
    assert loaded_hyp.target_candidate.symbol == "TargetZ"

    # Verify JSON structure
    data = json.loads(json_str)
    assert data["killer_experiment_pico"]["population"] == "PopZ"
    assert data["confidence"] == "PROBABLE"
