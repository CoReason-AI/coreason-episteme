# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from unittest.mock import patch

from coreason_episteme.models import ConfidenceLevel, GeneticTarget, Hypothesis, PICO
from coreason_episteme.server import EpistemeAsync, app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.2.1"}


def test_generate_hypothesis():
    mock_engine = MagicMock(spec=EpistemeAsync)
    mock_engine.run = AsyncMock()
    # We also need access to the inner engine if overriding retries
    mock_engine.engine = MagicMock()
    mock_engine.engine.gap_scanner = MagicMock()
    mock_engine.engine.bridge_builder = MagicMock()
    # ... set other attributes if needed, but if we don't send max_retries, it uses run directly

    with TestClient(app) as client:
        # Override the engine in the app state
        # Note: TestClient(app) runs lifespan. app.state is populated.
        # We overwrite it.
        app.state.engine = mock_engine

        hyp = Hypothesis(
            id="hyp-1",
            title="Test Hypothesis",
            knowledge_gap="Gap",
            proposed_mechanism="Mech",
            target_candidate=GeneticTarget(
                symbol="TEST", ensembl_id="ENSG01", druggability_score=0.9, novelty_score=0.8
            ),
            causal_validation_score=0.9,
            key_counterfactual="If X then Y",
            killer_experiment_pico=PICO(population="P", intervention="I", comparator="C", outcome="O"),
            evidence_chain=["PMID:123"],
            confidence=ConfidenceLevel.PROBABLE,
        )
        mock_engine.run.return_value = [hyp]

        response = client.post("/generate", json={"disease_id": "D123"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "hyp-1"

        mock_engine.run.assert_called_once()
        args, kwargs = mock_engine.run.call_args
        assert args[0] == "D123"
        assert kwargs["context"].user_id == "api-user"


def test_generate_hypothesis_with_retries():
    mock_engine = MagicMock(spec=EpistemeAsync)
    # mock_engine.run is not called on the base engine if retries are set,
    # because we create a new engine and run that.

    # We need to mock EpistemeEngineAsync in server.py to verify it's instantiated and run
    with patch("coreason_episteme.server.EpistemeEngineAsync") as MockEngineClass:
        mock_temp_engine = AsyncMock()
        MockEngineClass.return_value = mock_temp_engine
        mock_temp_engine.run.return_value = []

        with TestClient(app) as client:
            app.state.engine = mock_engine
            # Ensure base_engine is accessible
            mock_engine.engine = MagicMock()

            response = client.post("/generate", json={"disease_id": "D123", "max_retries": 5})
            assert response.status_code == 200

            # Check if EpistemeEngineAsync was initialized with max_retries=5
            MockEngineClass.assert_called_once()
            _, kwargs = MockEngineClass.call_args
            assert kwargs["max_retries"] == 5

            # Check if run was called on temp engine
            mock_temp_engine.run.assert_called_once()
