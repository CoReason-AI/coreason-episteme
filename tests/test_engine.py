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
from coreason_episteme.models import BridgeResult, KnowledgeGap

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


def test_engine_run_happy_path(engine: EpistemeEngine) -> None:
    """Test the full engine loop with a valid target."""
    results = engine.run("TargetX")

    assert len(results) == 1
    hypothesis = results[0]

    # Verify pipeline stages
    assert hypothesis.knowledge_gap.startswith("Simulated gap")
    assert hypothesis.proposed_mechanism == "Mock Mechanism"
    # Valid score from MockCausalValidator (default is 0.85)
    assert hypothesis.causal_validation_score == 0.85
    # Protocol Design applied
    assert hypothesis.killer_experiment_pico.population == "Mice"
    # Adversarial review check (no critiques for "TargetX")
    assert len(hypothesis.critiques) == 0

    # Verify Trace Logging
    veritas = engine.veritas_client  # type: ignore
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert trace["status"] == "ACCEPTED"
    assert trace["bridges_found_count"] == 2


def test_engine_run_no_gaps(engine: EpistemeEngine) -> None:
    """Test engine when no gaps are found."""
    results = engine.run("CleanTarget")  # Mock returns [] for "CleanTarget"
    assert len(results) == 0


def test_engine_run_low_causal_score(engine: EpistemeEngine) -> None:
    """Test that hypotheses with low causal scores are filtered out."""

    class BadTargetBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            result = super().generate_hypothesis(gap, excluded_targets)
            if result.hypothesis:
                result.hypothesis.target_candidate.symbol = "BadTarget"
            return result

    bad_engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=BadTargetBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = bad_engine.run("TargetX")
    # Should be filtered out because score will be 0.1 < 0.5
    assert len(results) == 0

    # Verify Trace Logs "DISCARDED"
    veritas = bad_engine.veritas_client  # type: ignore
    assert len(veritas.traces) == 1
    assert veritas.traces[0]["data"]["status"] == "DISCARDED (Low Causal Score)"


def test_engine_run_refinement_loop(engine: EpistemeEngine) -> None:
    """
    Test the Refinement Loop:
    1. First candidate is 'RiskyTarget' -> Triggers FATAL critique.
    2. Engine excludes 'RiskyTarget' and retries.
    3. Second candidate is 'SafeTarget' -> Success.
    """

    class RefinementBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            result = super().generate_hypothesis(gap, excluded_targets)

            # If RiskyTarget is NOT excluded, return it first
            if not excluded_targets or "RiskyTarget" not in excluded_targets:
                if result.hypothesis:
                    result.hypothesis.target_candidate.symbol = "RiskyTarget"
                return result

            # If RiskyTarget IS excluded, return SafeTarget
            if result.hypothesis:
                result.hypothesis.target_candidate.symbol = "SafeTarget"
            return result

    refinement_engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=RefinementBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = refinement_engine.run("TargetX")

    # Should succeed eventually
    assert len(results) == 1
    hypothesis = results[0]

    # Verify we got the SafeTarget eventually
    assert hypothesis.target_candidate.symbol == "SafeTarget"
    # SafeTarget shouldn't have FATAL critiques
    assert len(hypothesis.critiques) == 0

    # Verify Trace shows retries
    veritas = refinement_engine.veritas_client  # type: ignore
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert trace["status"] == "ACCEPTED"
    assert "RiskyTarget" in trace["excluded_targets_history"]
    assert trace["refinement_retries"] > 0


def test_engine_run_bridge_failure(engine: EpistemeEngine) -> None:
    """Test engine when bridge builder fails to generate a hypothesis."""

    class BrokenBridgeBuilder(MockBridgeBuilder):
        def generate_hypothesis(self, gap: KnowledgeGap, excluded_targets: Optional[List[str]] = None) -> BridgeResult:
            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

    broken_engine = EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=BrokenBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    results = broken_engine.run("TargetX")
    assert len(results) == 0

    # Verify Trace
    veritas = broken_engine.veritas_client  # type: ignore
    assert len(veritas.traces) == 1
    assert veritas.traces[0]["data"]["status"] == "DISCARDED (No Bridge)"
