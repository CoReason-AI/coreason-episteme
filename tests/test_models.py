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
            outcome="Health",  # type: ignore
        )


def test_knowledge_gap_model_valid() -> None:
    """Test creating a valid KnowledgeGap object with type."""
    gap = KnowledgeGap(
        description="Disconnect between Protein A and Disease B",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
        source_nodes=["Protein A", "Disease B"],
    )
    assert gap.type == KnowledgeGapType.CLUSTER_DISCONNECT
    assert gap.description == "Disconnect between Protein A and Disease B"


def test_knowledge_gap_model_invalid_type() -> None:
    """Test KnowledgeGap validation with invalid type."""
    with pytest.raises(ValidationError):
        KnowledgeGap(
            description="Bad Gap",
            type="UNKNOWN_TYPE",  # type: ignore
        )


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
            },  # type: ignore
            evidence_chain=[],
            confidence=ConfidenceLevel.SPECULATIVE,
        )
