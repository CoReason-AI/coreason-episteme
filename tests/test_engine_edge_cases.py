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
async def test_engine_exception_handling(user_context: UserContext) -> None:
    """
    Edge Case: A component raises an unhandled exception.
    Result: Engine should catch it, log trace with status=ERROR, and continue to next gap (or exit).
    """

    class CrashingBridgeBuilder(MockBridgeBuilder):
        async def generate_hypothesis(
            self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
        ) -> BridgeResult:
            raise RuntimeError("Critical Failure in Bridge Builder")

    veritas = MockVeritasClient()
    engine = EpistemeEngineAsync(
        gap_scanner=MockGapScanner(),
        bridge_builder=CrashingBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=veritas,
    )

    # Run engine. It should NOT raise, because we wrapped in try/except.
    async with engine:
        results = await engine.run("TargetX", context=user_context)

    # Should return empty results (as the gap failed)
    assert len(results) == 0

    # Verify Trace
    assert len(veritas.traces) == 1
    trace = veritas.traces[0]["data"]
    assert "ERROR" in trace["status"]
    assert "Critical Failure" in trace["status"]
    assert trace["gap_id"] is not None
