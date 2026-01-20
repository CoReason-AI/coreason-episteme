# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import Any, Dict, List, Optional

import pytest

from coreason_episteme.main import generate_hypothesis
from coreason_episteme.models import GeneticTarget, KnowledgeGap

# --- Stub Clients for Integration Test ---


class StubGraphNexusClient:
    def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        return []


class StubCodexClient:
    def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        return 0.0

    def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        return None


class StubSearchClient:
    def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        return []

    def verify_citation(self, interaction_claim: str) -> bool:
        return False

    def check_patent_infringement(self, target_candidate: GeneticTarget, mechanism: str) -> List[str]:
        return []

    def find_disconfirming_evidence(self, subject: str, object: str, action: str) -> List[str]:
        return []


class StubPrismClient:
    def check_druggability(self, target_id: str) -> float:
        return 0.0


class StubInferenceClient:
    def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        return 0.0

    def run_toxicology_screen(self, target_candidate: GeneticTarget) -> List[str]:
        return []

    def check_clinical_redundancy(self, mechanism: str, target_candidate: GeneticTarget) -> List[str]:
        return []


class StubVeritasClient:
    def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        pass


# --- Tests ---


def test_generate_hypothesis_missing_graph_client() -> None:
    """Test that runtime error is raised if GraphNexusClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: GraphNexusClient"):
        generate_hypothesis("TargetX", graph_client=None)


def test_generate_hypothesis_missing_codex_client() -> None:
    """Test that runtime error is raised if CodexClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: CodexClient"):
        generate_hypothesis("TargetX", graph_client=StubGraphNexusClient(), codex_client=None)


def test_generate_hypothesis_missing_search_client() -> None:
    """Test that runtime error is raised if SearchClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: SearchClient"):
        generate_hypothesis(
            "TargetX",
            graph_client=StubGraphNexusClient(),
            codex_client=StubCodexClient(),
            search_client=None,
        )


def test_generate_hypothesis_missing_prism_client() -> None:
    """Test that runtime error is raised if PrismClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: PrismClient"):
        generate_hypothesis(
            "TargetX",
            graph_client=StubGraphNexusClient(),
            codex_client=StubCodexClient(),
            search_client=StubSearchClient(),
            prism_client=None,
        )


def test_generate_hypothesis_missing_inference_client() -> None:
    """Test that runtime error is raised if InferenceClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: InferenceClient"):
        generate_hypothesis(
            "TargetX",
            graph_client=StubGraphNexusClient(),
            codex_client=StubCodexClient(),
            search_client=StubSearchClient(),
            prism_client=StubPrismClient(),
            inference_client=None,
        )


def test_generate_hypothesis_missing_veritas_client() -> None:
    """Test that runtime error is raised if VeritasClient is missing."""
    with pytest.raises(RuntimeError, match="Missing required external client: VeritasClient"):
        generate_hypothesis(
            "TargetX",
            graph_client=StubGraphNexusClient(),
            codex_client=StubCodexClient(),
            search_client=StubSearchClient(),
            prism_client=StubPrismClient(),
            inference_client=StubInferenceClient(),
            veritas_client=None,
        )


def test_generate_hypothesis_success() -> None:
    """Test that generate_hypothesis runs with mock clients."""
    # We use stubs that return empty lists/None, so the result should be empty list of hypotheses
    # But it proves the wiring works.
    results = generate_hypothesis(
        disease_id="TargetX",
        graph_client=StubGraphNexusClient(),
        codex_client=StubCodexClient(),
        search_client=StubSearchClient(),
        prism_client=StubPrismClient(),
        inference_client=StubInferenceClient(),
        veritas_client=StubVeritasClient(),
    )
    assert isinstance(results, list)
    assert len(results) == 0
