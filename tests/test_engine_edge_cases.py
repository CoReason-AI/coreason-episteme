# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List, Optional

import pytest

from coreason_episteme.engine import EpistemeEngine
from coreason_episteme.models import (
    PICO,
    BridgeResult,
    ConfidenceLevel,
    Critique,
    CritiqueSeverity,
    GeneticTarget,
    Hypothesis,
    KnowledgeGap,
    KnowledgeGapType,
)
from tests.mocks import (
    MockAdversarialReviewer,
    MockBridgeBuilder,
    MockCausalValidator,
    MockGapScanner,
    MockProtocolDesigner,
    MockVeritasClient,
)


@pytest.fixture  # type: ignore[misc]
def engine() -> EpistemeEngine:
    return EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=MockBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )


def test_multi_gap_mixed_outcomes() -> None:
    """
    Scenario: The GapScanner returns 3 gaps.
    1. Gap A -> Bridge Found -> Valid -> Accepted.
    2. Gap B -> No Bridges Found -> Discarded.
    3. Gap C -> Bridge Found -> Low Causal Score -> Discarded.

    Verify: 3 distinct traces are logged with correct statuses.
    """

    class MultiGapScanner(MockGapScanner):
        def scan(self, target: str) -> List[KnowledgeGap]:
            return [
                KnowledgeGap(description="Gap A", type=KnowledgeGapType.CLUSTER_DISCONNECT, source_nodes=["A1", "A2"]),
                KnowledgeGap(description="Gap B", type=KnowledgeGapType.CLUSTER_DISCONNECT, source_nodes=["B1", "B2"]),
                KnowledgeGap(description="Gap C", type=KnowledgeGapType.CLUSTER_DISCONNECT, source_nodes=["C1", "C2"]),
            ]

    class MultiScenarioBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            if gap.description == "Gap A":
                # Success case
                hyp = Hypothesis(
                    id="hyp-A",
                    title="Success A",
                    knowledge_gap=gap.description,
                    proposed_mechanism="Mech A",
                    target_candidate=GeneticTarget(
                        symbol="GeneA", ensembl_id="E1", druggability_score=0.9, novelty_score=0.9
                    ),
                    causal_validation_score=0.9,
                    key_counterfactual="",
                    killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                    evidence_chain=[],
                    confidence=ConfidenceLevel.SPECULATIVE,
                )
                return BridgeResult(hypothesis=hyp, bridges_found_count=1, considered_candidates=["GeneA"])

            if gap.description == "Gap B":
                # No bridges
                return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

            if gap.description == "Gap C":
                # Found bridge, but will fail validation
                hyp = Hypothesis(
                    id="hyp-C",
                    title="Fail C",
                    knowledge_gap=gap.description,
                    proposed_mechanism="Mech C",
                    target_candidate=GeneticTarget(
                        symbol="GeneC", ensembl_id="E3", druggability_score=0.9, novelty_score=0.9
                    ),
                    # Will be overwritten by validator mock, so we need a smart validator mock
                    causal_validation_score=0.1,
                    key_counterfactual="",
                    killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                    evidence_chain=[],
                    confidence=ConfidenceLevel.SPECULATIVE,
                )
                return BridgeResult(hypothesis=hyp, bridges_found_count=1, considered_candidates=["GeneC"])

            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

    class ContextAwareValidator(MockCausalValidator):
        def validate(self, hypothesis: Hypothesis) -> Hypothesis:
            if hypothesis.target_candidate.symbol == "GeneC":
                hypothesis.causal_validation_score = 0.2
            else:
                hypothesis.causal_validation_score = 0.9
            return hypothesis

    veritas = MockVeritasClient()

    engine = EpistemeEngine(
        gap_scanner=MultiGapScanner(),
        bridge_builder=MultiScenarioBridgeBuilder(),
        causal_validator=ContextAwareValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=veritas,
    )

    results = engine.run("DiseaseX")

    # Verify Results (Only 1 accepted)
    assert len(results) == 1
    assert results[0].id == "hyp-A"

    # Verify Traces
    assert len(veritas.traces) == 3

    # Sort by gap description to simplify assertions
    traces = sorted(veritas.traces, key=lambda t: t["data"]["gap"]["description"])

    # Trace A (Gap A)
    assert traces[0]["data"]["gap"]["description"] == "Gap A"
    assert traces[0]["data"]["status"] == "ACCEPTED"
    assert traces[0]["id"] == "hyp-A"

    # Trace B (Gap B)
    assert traces[1]["data"]["gap"]["description"] == "Gap B"
    assert traces[1]["data"]["status"] == "DISCARDED (No Bridge)"
    # ID might be auto-generated or failure ID
    assert "failed-gap" in traces[1]["id"] or traces[1]["id"] is None

    # Trace C (Gap C)
    assert traces[2]["data"]["gap"]["description"] == "Gap C"
    assert traces[2]["data"]["status"] == "DISCARDED (Low Causal Score)"
    assert traces[2]["id"] == "hyp-C"


def test_deep_refinement_history_logging() -> None:
    """
    Scenario:
    1. Candidate 1 (GeneX) -> FATAL Critique.
    2. Candidate 2 (GeneY) -> FATAL Critique.
    3. Candidate 3 (GeneZ) -> Success.

    Verify:
    - Trace captures all excluded targets in history.
    - Retry count is 2.
    - Final status is ACCEPTED.
    """

    class RetryingBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            excluded = excluded_targets or []
            candidates = ["GeneX", "GeneY", "GeneZ"]

            # Find first non-excluded candidate
            for c in candidates:
                if c not in excluded:
                    hyp = Hypothesis(
                        id=f"hyp-{c}",
                        title=f"Hypothesis {c}",
                        knowledge_gap=gap.description,
                        proposed_mechanism="Mech",
                        target_candidate=GeneticTarget(
                            symbol=c, ensembl_id="E", druggability_score=0.9, novelty_score=0.9
                        ),
                        causal_validation_score=0.9,
                        key_counterfactual="",
                        killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                        evidence_chain=[],
                        confidence=ConfidenceLevel.SPECULATIVE,
                    )
                    return BridgeResult(hypothesis=hyp, bridges_found_count=3, considered_candidates=candidates)

            return BridgeResult(hypothesis=None, bridges_found_count=3, considered_candidates=candidates)

    class StrictReviewer(MockAdversarialReviewer):
        def review(self, hypothesis: Hypothesis) -> Hypothesis:
            sym = hypothesis.target_candidate.symbol
            if sym in ["GeneX", "GeneY"]:
                hypothesis.critiques.append(
                    Critique(source="Toxicologist", content="Fatal", severity=CritiqueSeverity.FATAL)
                )
            return hypothesis

    veritas = MockVeritasClient()

    engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=RetryingBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=StrictReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=veritas,
    )

    results = engine.run("DiseaseY")

    assert len(results) == 1
    assert results[0].target_candidate.symbol == "GeneZ"

    # Verify Trace
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]

    assert trace["status"] == "ACCEPTED"
    assert trace["refinement_retries"] == 2
    assert "GeneX" in trace["excluded_targets_history"]
    assert "GeneY" in trace["excluded_targets_history"]
    assert len(trace["excluded_targets_history"]) == 2
