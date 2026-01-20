# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

import uuid
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
)
from tests.mocks import (
    MockAdversarialReviewer,
    MockBridgeBuilder,
    MockCausalValidator,
    MockGapScanner,
    MockProtocolDesigner,
    MockVeritasClient,
)


@pytest.fixture
def engine() -> EpistemeEngine:
    return EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=MockBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )


def test_refinement_max_retries_exceeded() -> None:
    """
    Edge Case: The system keeps finding toxic targets until max_retries is hit.
    Result: Should return empty list (failed to find safe hypothesis).
    """

    class InfiniteToxicBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            # Always return a new Toxic target
            unique_id = uuid.uuid4().hex[:6]
            hyp = Hypothesis(
                id=str(uuid.uuid4()),
                title="Toxic Hypothesis",
                knowledge_gap=gap.description,
                proposed_mechanism="Mechanism",
                target_candidate=GeneticTarget(
                    symbol=f"Risky_{unique_id}",
                    ensembl_id="ENSG000",
                    druggability_score=0.9,
                    novelty_score=0.9,
                ),
                causal_validation_score=0.9,
                key_counterfactual="",
                killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                evidence_chain=[],
                confidence=ConfidenceLevel.SPECULATIVE,
            )
            return BridgeResult(
                hypothesis=hyp, bridges_found_count=1, considered_candidates=[hyp.target_candidate.symbol]
            )

    engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=InfiniteToxicBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),  # Will flag "Risky_" as FATAL
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = engine.run("TargetX")
    assert len(results) == 0


def test_refinement_candidate_exhaustion() -> None:
    """
    Edge Case: BridgeBuilder runs out of candidates before max_retries.
    Result: Should return empty list.
    """

    class ExhaustibleBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            # If any target is excluded, assume we ran out of options
            if excluded_targets:
                return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

            # Return one toxic target initially
            hyp = Hypothesis(
                id=str(uuid.uuid4()),
                title="Toxic Hypothesis",
                knowledge_gap=gap.description,
                proposed_mechanism="Mechanism",
                target_candidate=GeneticTarget(
                    symbol="Risky_OneShot",
                    ensembl_id="ENSG000",
                    druggability_score=0.9,
                    novelty_score=0.9,
                ),
                causal_validation_score=0.9,
                key_counterfactual="",
                killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                evidence_chain=[],
                confidence=ConfidenceLevel.SPECULATIVE,
            )
            return BridgeResult(
                hypothesis=hyp, bridges_found_count=1, considered_candidates=[hyp.target_candidate.symbol]
            )

    engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=ExhaustibleBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = engine.run("TargetX")
    assert len(results) == 0


def test_complex_severity_threshold() -> None:
    """
    Complex Scenario: "The Unicorn Hunt"
    1. Candidate A -> Toxic (FATAL) -> Refine
    2. Candidate B -> Disproven (FATAL) -> Refine
    3. Candidate C -> IP Issue (HIGH) -> Accept (Warning)
    """

    class UnicornBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            excluded = excluded_targets or []

            # 1. Toxic
            if "ToxicGene" not in excluded:
                target = GeneticTarget(symbol="ToxicGene", ensembl_id="E1", druggability_score=0.9, novelty_score=0.9)
            # 2. Disproven
            elif "DisprovenGene" not in excluded:
                target = GeneticTarget(
                    symbol="DisprovenGene", ensembl_id="E2", druggability_score=0.9, novelty_score=0.9
                )
            # 3. IP Issue (Final)
            else:
                target = GeneticTarget(symbol="IPGene", ensembl_id="E3", druggability_score=0.9, novelty_score=0.9)

            hyp = Hypothesis(
                id=str(uuid.uuid4()),
                title="Hypothesis",
                knowledge_gap=gap.description,
                proposed_mechanism="Mech",
                target_candidate=target,
                causal_validation_score=0.9,
                key_counterfactual="",
                killer_experiment_pico=PICO(population="", intervention="", comparator="", outcome=""),
                evidence_chain=[],
                confidence=ConfidenceLevel.SPECULATIVE,
            )
            return BridgeResult(hypothesis=hyp, bridges_found_count=1, considered_candidates=[target.symbol])

    class ComplexReviewer(MockAdversarialReviewer):
        def review(self, hypothesis: Hypothesis) -> Hypothesis:
            symbol = hypothesis.target_candidate.symbol
            if symbol == "ToxicGene":
                hypothesis.critiques.append(
                    Critique(source="Toxicologist", content="Fatal Toxicity", severity=CritiqueSeverity.FATAL)
                )
            elif symbol == "DisprovenGene":
                hypothesis.critiques.append(
                    Critique(source="Scientific Skeptic", content="Fatal Disproof", severity=CritiqueSeverity.FATAL)
                )
            elif symbol == "IPGene":
                hypothesis.critiques.append(
                    Critique(source="IP Strategist", content="Patent Conflict", severity=CritiqueSeverity.HIGH)
                )
            return hypothesis

    engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=UnicornBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=ComplexReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = engine.run("TargetX")

    # Should accept the 3rd one
    assert len(results) == 1
    winner = results[0]
    assert winner.target_candidate.symbol == "IPGene"
    # Verify it has the HIGH critique attached
    assert any(c.severity == CritiqueSeverity.HIGH for c in winner.critiques)
    # Verify it does NOT have FATAL critiques
    assert not any(c.severity == CritiqueSeverity.FATAL for c in winner.critiques)
