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

from coreason_episteme.engine import EpistemeEngine
from tests.mocks import (
    MockAdversarialReviewer,
    MockBridgeBuilder,
    MockCausalValidator,
    MockGapScanner,
    MockProtocolDesigner,
    MockVeritasClient,
)


@pytest.fixture  # type: ignore[misc]
def mock_veritas() -> MockVeritasClient:
    return MockVeritasClient()


@pytest.fixture  # type: ignore[misc]
def engine(mock_veritas: MockVeritasClient) -> EpistemeEngine:
    # Attempt to inject veritas_client - this will fail until EpistemeEngine is updated
    return EpistemeEngine(
        gap_scanner=MockGapScanner(),
        bridge_builder=MockBridgeBuilder(),
        causal_validator=MockCausalValidator(),
        adversarial_reviewer=MockAdversarialReviewer(),
        protocol_designer=MockProtocolDesigner(),
        veritas_client=mock_veritas,
    )


def test_engine_lifecycle_logging(engine: EpistemeEngine, mock_veritas: MockVeritasClient) -> None:
    """Test that engine logs all lifecycle events."""
    results = engine.run("TargetX")
    assert len(results) == 1
    hypothesis_id = results[0].id

    # Filter traces for this hypothesis
    traces = [t for t in mock_veritas.traces if t["id"] == hypothesis_id]

    # We expect logs for:
    # 1. Causal Validation
    # 2. Adversarial Review
    # 3. Protocol Design (Completion)

    # Check Causal Validation Log
    validation_logs = [t for t in traces if "causal_validation_score" in t["data"]]
    assert len(validation_logs) == 1
    assert validation_logs[0]["data"]["causal_validation_score"] == 0.85

    # Check Review Log
    review_logs = [t for t in traces if "critiques_count" in t["data"]]
    assert len(review_logs) == 1

    # Check Completion Log
    completion_logs = [t for t in traces if "event" in t["data"] and t["data"]["event"] == "PROTOCOL_DESIGNED"]
    assert len(completion_logs) == 1
