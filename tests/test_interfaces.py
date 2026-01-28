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

from coreason_identity.models import UserContext

from coreason_episteme.interfaces import (
    BridgeBuilder,
    CausalValidator,
    CodexClient,
    GapScanner,
    GraphNexusClient,
    InferenceClient,
    PrismClient,
    ProtocolDesigner,
    SearchClient,
    VeritasClient,
)
from coreason_episteme.models import BridgeResult, GeneticTarget, Hypothesis, KnowledgeGap


class MockGraphNexusClient:
    async def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    async def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        return []


class MockInferenceClient:
    async def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        return 0.5

    async def run_toxicology_screen(self, target_candidate: GeneticTarget) -> List[str]:
        return []

    async def check_clinical_redundancy(self, mechanism: str, target_candidate: GeneticTarget) -> List[str]:
        return []


class MockCodexClient:
    async def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        return 0.8

    async def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        return None


class MockPrismClient:
    async def check_druggability(self, target_id: str) -> float:
        return 0.9


class MockSearchClient:
    async def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        return []

    async def verify_citation(self, interaction_claim: str) -> bool:
        return True

    async def check_patent_infringement(self, target_candidate: GeneticTarget, mechanism: str) -> List[str]:
        return []

    async def find_disconfirming_evidence(self, subject: str, object: str, action: str) -> List[str]:
        return []


class MockVeritasClient:
    async def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        pass


class MockGapScanner:
    async def scan(self, target: str, context: UserContext) -> List[KnowledgeGap]:
        return []


class MockBridgeBuilder:
    async def generate_hypothesis(
        self, gap: KnowledgeGap, context: UserContext, excluded_targets: Optional[List[str]] = None
    ) -> BridgeResult:
        return BridgeResult(hypothesis=None, bridges_found_count=0, considered_candidates=[])


class MockCausalValidator:
    async def validate(self, hypothesis: Hypothesis, context: UserContext) -> Hypothesis:
        return hypothesis


class MockProtocolDesigner:
    async def design_experiment(self, hypothesis: Hypothesis) -> Hypothesis:
        return hypothesis


def test_interfaces_are_implementable() -> None:
    """
    Verifies that the mock classes correctly implement the protocols.
    Static type checking (mypy) will do the heavy lifting here,
    but we can also instantiate them to ensure runtime correctness.
    """
    _graph_nexus: GraphNexusClient = MockGraphNexusClient()
    _inference: InferenceClient = MockInferenceClient()
    _codex: CodexClient = MockCodexClient()
    _prism: PrismClient = MockPrismClient()
    _search: SearchClient = MockSearchClient()
    _veritas: VeritasClient = MockVeritasClient()

    _gap_scanner: GapScanner = MockGapScanner()
    _bridge_builder: BridgeBuilder = MockBridgeBuilder()
    _causal_validator: CausalValidator = MockCausalValidator()
    _protocol_designer: ProtocolDesigner = MockProtocolDesigner()

    assert True
