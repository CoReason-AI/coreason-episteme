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
    """Test that engine logs all lifecycle events in a single consolidated trace."""
    results = engine.run("TargetX")
    assert len(results) == 1
    hypothesis_id = results[0].id

    # Filter traces for this hypothesis
    traces = [t for t in mock_veritas.traces if t["id"] == hypothesis_id]

    # We now expect exactly ONE trace per successful hypothesis generation
    assert len(traces) == 1
    trace_data = traces[0]["data"]

    # Verify Causal Validation data is present
    assert "causal_validation_score" in trace_data
    assert trace_data["causal_validation_score"] == 0.85

    # Verify Review data is present (even if empty list, the field should exist)
    assert "critiques" in trace_data

    # Verify Status
    assert trace_data["status"] == "ACCEPTED"

    # Verify Metadata
    assert trace_data["bridges_found_count"] > 0
