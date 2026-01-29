# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

"""
HTTP implementations of external service clients.
"""

from typing import Any, Dict, List, Optional

import httpx

from coreason_episteme.models import GeneticTarget, KnowledgeGap


class HttpGraphNexusClient:
    """HTTP client for coreason-graph-nexus."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = await self.client.post(f"{self.base_url}/find_disconnected_clusters", json=criteria)
        response.raise_for_status()
        return response.json()

    async def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        payload = {
            "source_cluster_id": source_cluster_id,
            "target_cluster_id": target_cluster_id,
        }
        response = await self.client.post(f"{self.base_url}/find_latent_bridges", json=payload)
        response.raise_for_status()
        return [GeneticTarget(**item) for item in response.json()]


class HttpCodexClient:
    """HTTP client for coreason-codex."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        payload = {"entity1": entity1, "entity2": entity2}
        response = await self.client.post(f"{self.base_url}/get_semantic_similarity", json=payload)
        response.raise_for_status()
        return float(response.json())

    async def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        payload = {"symbol": symbol}
        response = await self.client.post(f"{self.base_url}/validate_target", json=payload)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if data is None:
            return None
        return GeneticTarget(**data)


class HttpSearchClient:
    """HTTP client for coreason-search."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        payload = {"topic": topic}
        response = await self.client.post(f"{self.base_url}/find_literature_inconsistency", json=payload)
        response.raise_for_status()
        return [KnowledgeGap(**item) for item in response.json()]

    async def verify_citation(self, interaction_claim: str) -> bool:
        payload = {"interaction_claim": interaction_claim}
        response = await self.client.post(f"{self.base_url}/verify_citation", json=payload)
        response.raise_for_status()
        return bool(response.json())

    async def check_patent_infringement(self, target_candidate: GeneticTarget, mechanism: str) -> List[str]:
        payload = {
            "target_candidate": target_candidate.model_dump(),
            "mechanism": mechanism,
        }
        response = await self.client.post(f"{self.base_url}/check_patent_infringement", json=payload)
        response.raise_for_status()
        return list(response.json())

    async def find_disconfirming_evidence(self, subject: str, object: str, action: str) -> List[str]:
        payload = {"subject": subject, "object": object, "action": action}
        response = await self.client.post(f"{self.base_url}/find_disconfirming_evidence", json=payload)
        response.raise_for_status()
        return list(response.json())


class HttpPrismClient:
    """HTTP client for coreason-prism."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def check_druggability(self, target_id: str) -> float:
        payload = {"target_id": target_id}
        response = await self.client.post(f"{self.base_url}/check_druggability", json=payload)
        response.raise_for_status()
        return float(response.json())


class HttpInferenceClient:
    """HTTP client for coreason-inference."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        payload = {"mechanism": mechanism, "intervention_target": intervention_target}
        response = await self.client.post(f"{self.base_url}/run_counterfactual_simulation", json=payload)
        response.raise_for_status()
        return float(response.json())

    async def run_toxicology_screen(self, target_candidate: GeneticTarget) -> List[str]:
        payload = {"target_candidate": target_candidate.model_dump()}
        response = await self.client.post(f"{self.base_url}/run_toxicology_screen", json=payload)
        response.raise_for_status()
        return list(response.json())

    async def check_clinical_redundancy(self, mechanism: str, target_candidate: GeneticTarget) -> List[str]:
        payload = {
            "mechanism": mechanism,
            "target_candidate": target_candidate.model_dump(),
        }
        response = await self.client.post(f"{self.base_url}/check_clinical_redundancy", json=payload)
        response.raise_for_status()
        return list(response.json())


class HttpVeritasClient:
    """HTTP client for coreason-veritas."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url
        self.client = client

    async def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        payload = {"hypothesis_id": hypothesis_id, "trace_data": trace_data}
        # Fire and forget / background task might be better, but we await as per interface
        await self.client.post(f"{self.base_url}/log_trace", json=payload)
