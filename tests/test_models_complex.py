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
    Critique,
    CritiqueSeverity,
    GeneticTarget,
    Hypothesis,
    HypothesisTrace,
    KnowledgeGap,
    KnowledgeGapType,
)


def test_complex_hypothesis_trace_lifecycle() -> None:
    """
    Test the lifecycle of a HypothesisTrace object, simulating the Engine's state transitions.
    This covers complex nesting and state mutation.
    """
    # 1. Initial State (Gap Scanning)
    gap = KnowledgeGap(
        description="Gap A",
        type=KnowledgeGapType.CLUSTER_DISCONNECT,
        source_nodes=["A", "B"],
    )
    trace = HypothesisTrace(gap=gap, status="PENDING")

    assert trace.status == "PENDING"
    assert trace.bridges_found_count == 0
    assert trace.excluded_targets_history == []

    # 2. Bridge Phase (First Attempt - Failure/Refinement)
    trace.bridges_found_count = 5
    trace.considered_candidates = ["Gene1", "Gene2", "Gene3"]
    # Simulate adding an excluded target
    trace.excluded_targets_history.append("Gene1")
    trace.refinement_retries = 1

    assert "Gene1" in trace.excluded_targets_history
    assert trace.refinement_retries == 1

    # 3. Critique Phase (Accumulating Critiques)
    critique1 = Critique(source="Toxicologist", content="Liver toxicity", severity=CritiqueSeverity.FATAL)
    critique2 = Critique(source="Clinician", content="Redundant", severity=CritiqueSeverity.MEDIUM)

    trace.critiques.append(critique1)
    trace.critiques.append(critique2)

    assert len(trace.critiques) == 2
    assert trace.critiques[0].severity == CritiqueSeverity.FATAL

    # 4. Final Success State
    target = GeneticTarget(symbol="Gene2", ensembl_id="ENSG0002", druggability_score=0.9, novelty_score=0.8)
    hypothesis = Hypothesis(
        id="HYP-FINAL",
        title="Final Hyp",
        knowledge_gap="Gap A",
        proposed_mechanism="Mech A",
        target_candidate=target,
        causal_validation_score=0.8,
        key_counterfactual="If not Gene2...",
        killer_experiment_pico=PICO(population="Pop", intervention="Int", comparator="Comp", outcome="Out"),
        evidence_chain=["A", "B", "ENSG0002"],
        confidence=ConfidenceLevel.PROBABLE,
        critiques=[critique2],  # Maybe only non-fatal ones remain attached to the hypothesis
    )

    trace.result = hypothesis
    trace.status = "ACCEPTED"
    trace.hypothesis_id = hypothesis.id

    # Verify serialization of this complex object
    json_str = trace.model_dump_json()
    reloaded_trace = HypothesisTrace.model_validate_json(json_str)

    assert reloaded_trace.status == "ACCEPTED"
    assert reloaded_trace.result is not None
    assert reloaded_trace.result.id == "HYP-FINAL"
    assert len(reloaded_trace.critiques) == 2  # The trace keeps the history of critiques
    assert reloaded_trace.excluded_targets_history == ["Gene1"]


def test_enum_validation_edge_cases() -> None:
    """Test validation behavior for Enums with invalid values."""

    # Invalid Severity
    with pytest.raises(ValidationError) as excinfo:
        Critique(
            source="Test",
            content="Test",
            severity="CATASTROPHIC",  # Invalid value, valid ones are LOW, MEDIUM, HIGH, FATAL
        )
    assert "Input should be 'LOW', 'MEDIUM', 'HIGH' or 'FATAL'" in str(excinfo.value)

    # Invalid Confidence Level
    with pytest.raises(ValidationError):
        Hypothesis(
            id="H1",
            title="T",
            knowledge_gap="G",
            proposed_mechanism="M",
            target_candidate=GeneticTarget(symbol="S", ensembl_id="E", druggability_score=1.0, novelty_score=1.0),
            causal_validation_score=1.0,
            key_counterfactual="K",
            killer_experiment_pico=PICO(population="P", intervention="I", comparator="C", outcome="O"),
            evidence_chain=[],
            confidence="CERTAIN",  # Invalid
        )


def test_partial_hypothesis_trace() -> None:
    """Test HypothesisTrace with minimal fields (testing Optional handling)."""
    gap = KnowledgeGap(description="D", type=KnowledgeGapType.LITERATURE_INCONSISTENCY)
    trace = HypothesisTrace(gap=gap)

    # Check default values
    assert trace.bridges_found_count == 0
    assert trace.considered_candidates == []
    assert trace.excluded_targets_history == []
    assert trace.critiques == []
    assert trace.result is None
    assert trace.status == "PENDING"
    assert trace.causal_validation_score is None
    assert trace.key_counterfactual is None


def test_critique_list_operations() -> None:
    """Test operations on the critiques list within Hypothesis."""
    hyp = Hypothesis(
        id="H1",
        title="T",
        knowledge_gap="G",
        proposed_mechanism="M",
        target_candidate=GeneticTarget(symbol="S", ensembl_id="E", druggability_score=1.0, novelty_score=1.0),
        causal_validation_score=1.0,
        key_counterfactual="K",
        killer_experiment_pico=PICO(population="P", intervention="I", comparator="C", outcome="O"),
        evidence_chain=[],
        confidence=ConfidenceLevel.SPECULATIVE,
    )

    # Start empty
    assert hyp.critiques == []

    # Add valid critique
    c1 = Critique(source="S1", content="C1", severity=CritiqueSeverity.LOW)
    hyp.critiques.append(c1)
    assert len(hyp.critiques) == 1

    # Attempt to add invalid object (not caught by runtime list append in Python,
    # but caught if we try to re-validate or strict type check)
    # Pydantic models are not strict lists at runtime unless using a custom list type,
    # but let's see what happens if we dump and validate.

    hyp.critiques.append("Not a critique object")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Hypothesis.model_validate(hyp.model_dump())
