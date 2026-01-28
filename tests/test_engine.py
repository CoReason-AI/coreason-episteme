# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List, Optional, cast

import pytest
from coreason_identity.models import UserContext

from coreason_episteme.engine import EpistemeEngineAsync
from coreason_episteme.models import BridgeResult, KnowledgeGap
from tests.mocks import (
    MockAdversarialReviewer,
    MockBridgeBuilder,
    MockCausalValidator,
    MockGapScanner,
    MockProtocolDesigner,
    MockVeritasClient,
)


@pytest.fixture
def engine() -> EpistemeEngineAsync:
    return EpistemeEngineAsync(
        gap_scanner=MockGapScanner(),
        bridge_builder=MockBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )


@pytest.fixture
def user_context() -> UserContext:
    return UserContext(
        sub="test-user",
        email="test@coreason.ai",
        permissions=[],
        project_context="test",
    )


@pytest.mark.asyncio
async def test_engine_run_happy_path(engine: EpistemeEngineAsync, user_context: UserContext) -> None:
    """Test the full engine loop with a valid target."""
    async with engine:
        results = await engine.run("TargetX", context=user_context)

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
    veritas = cast(MockVeritasClient, engine.veritas_client)
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert trace["status"] == "ACCEPTED"
    assert trace["bridges_found_count"] == 2
    assert trace["gap_id"] is not None
    assert trace["bridge_id"] is not None


@pytest.mark.asyncio
async def test_engine_run_no_gaps(engine: EpistemeEngineAsync, user_context: UserContext) -> None:
    """Test engine when no gaps are found."""
    async with engine:
        results = await engine.run("CleanTarget", context=user_context)  # Mock returns [] for "CleanTarget"
    assert len(results) == 0


@pytest.mark.asyncio
async def test_engine_run_low_causal_score(engine: EpistemeEngineAsync, user_context: UserContext) -> None:
    """Test that hypotheses with low causal scores are filtered out."""

    class BadTargetBridgeBuilder(MockBridgeBuilder):
        async def generate_hypothesis(
            self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
        ) -> BridgeResult:
            result = await super().generate_hypothesis(gap, context, excluded_targets)
            if result.hypothesis:
                result.hypothesis.target_candidate.symbol = "BadTarget"
            return result

    bad_engine = EpistemeEngineAsync(
        gap_scanner=MockGapScanner(),
        bridge_builder=BadTargetBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    async with bad_engine:
        results = await bad_engine.run("TargetX", context=user_context)
    # Should be filtered out because score will be 0.1 < 0.5
    assert len(results) == 0

    # Verify Trace Logs "DISCARDED"
    veritas = cast(MockVeritasClient, bad_engine.veritas_client)
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert trace["status"] == "DISCARDED (Low Causal Score)"
    assert trace["gap_id"] is not None
    # bridge_id should be present because hypothesis was generated (but discarded)
    assert trace["bridge_id"] is not None


@pytest.mark.asyncio
async def test_engine_run_refinement_loop(engine: EpistemeEngineAsync, user_context: UserContext) -> None:
    """
    Test the Refinement Loop:
    1. First candidate is 'RiskyTarget' -> Triggers FATAL critique.
    2. Engine excludes 'RiskyTarget' and retries.
    3. Second candidate is 'SafeTarget' -> Success.
    """

    class RefinementBridgeBuilder(MockBridgeBuilder):
        async def generate_hypothesis(
            self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
        ) -> BridgeResult:
            result = await super().generate_hypothesis(gap, context, excluded_targets)

            # If RiskyTarget is NOT excluded, return it first
            if not excluded_targets or "RiskyTarget" not in excluded_targets:
                if result.hypothesis:
                    result.hypothesis.target_candidate.symbol = "RiskyTarget"
                return result

            # If RiskyTarget IS excluded, return SafeTarget
            if result.hypothesis:
                result.hypothesis.target_candidate.symbol = "SafeTarget"
            return result

    refinement_engine = EpistemeEngineAsync(
        gap_scanner=MockGapScanner(),
        bridge_builder=RefinementBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    async with refinement_engine:
        results = await refinement_engine.run("TargetX", context=user_context)

    # Should succeed eventually
    assert len(results) == 1
    hypothesis = results[0]

    # Verify we got the SafeTarget eventually
    assert hypothesis.target_candidate.symbol == "SafeTarget"
    # SafeTarget shouldn't have FATAL critiques
    assert len(hypothesis.critiques) == 0

    # Verify Trace shows retries
    veritas = cast(MockVeritasClient, refinement_engine.veritas_client)
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert trace["status"] == "ACCEPTED"
    assert "RiskyTarget" in trace["excluded_targets_history"]
    assert trace["refinement_retries"] > 0


@pytest.mark.asyncio
async def test_engine_run_bridge_failure(engine: EpistemeEngineAsync, user_context: UserContext) -> None:
    """Test engine when bridge builder fails to generate a hypothesis."""

    class BrokenBridgeBuilder(MockBridgeBuilder):
        async def generate_hypothesis(
            self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
        ) -> BridgeResult:
            return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])

    broken_engine = EpistemeEngineAsync(
        gap_scanner=MockGapScanner(),
        bridge_builder=BrokenBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=MockVeritasClient(),
    )

    async with broken_engine:
        results = await broken_engine.run("TargetX", context=user_context)
    assert len(results) == 0

    # Verify Trace


@pytest.mark.asyncio
async def test_engine_missing_context(engine: EpistemeEngineAsync) -> None:
    """Test that missing context raises ValueError."""
    with pytest.raises(ValueError, match="context is required"):
        await engine.run("TargetX", context=None)
