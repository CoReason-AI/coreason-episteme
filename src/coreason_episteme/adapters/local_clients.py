from typing import Any, Dict, List, Optional

import networkx as nx

from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    InferenceClient,
    PrismClient,
    SearchClient,
    VeritasClient,
)
from coreason_episteme.models import GeneticTarget, KnowledgeGap


class LocalGraphNexusClient(GraphNexusClient):
    """Local implementation of GraphNexusClient using NetworkX."""

    def __init__(self, graph: Optional[nx.Graph] = None):
        self.graph = graph if graph is not None else nx.Graph()

    def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Finds disconnected subgraphs matching criteria.
        For this local implementation, we look for connected components
        that match the node attributes in 'criteria'.
        """
        clusters = []
        # Get connected components
        components = list(nx.connected_components(self.graph))

        # Filter components based on criteria (mock logic)
        # Assuming criteria might filter by node type or attributes present in the cluster
        target_type = criteria.get("type")

        for i, comp in enumerate(components):
            subgraph = self.graph.subgraph(comp)
            # Simple check: if criteria is provided, check if any node matches
            match = True
            if target_type:
                match = any(subgraph.nodes[n].get("type") == target_type for n in subgraph.nodes)

            if match:
                # Calculate simple centroid or representative
                clusters.append({"id": f"cluster_{i}", "nodes": list(comp), "size": len(comp)})
        return clusters

    def find_latent_bridges(self, source_cluster_id: str, target_cluster_id: str) -> List[GeneticTarget]:
        """
        Finds potential bridges between two clusters.
        In a real graph, this might be finding short paths or common neighbors
        if we consider a super-graph including weak links.

        For this local simulation, we return hardcoded 'latent' nodes
        that might have been added to the graph but marked as 'weak_signal'.
        """
        # In this simulation, we'll scan for nodes that have edges to both clusters
        # but are not strongly part of either (maybe strictly by connectivity).

        # Since we don't have the cluster definitions persistence, we assume
        # the user/system sets up the graph such that we can find them.

        # Mock behavior: Return a specific mock target if 'GeneX' is present in the graph
        # This is a simplification.

        bridges = []
        # scan all nodes not in source or target cluster?
        # For simplicity, we just return a mock list for now to satisfy the interface,
        # or check if specific nodes exist.

        # Let's say we look for nodes with attribute "is_bridge=True"
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("is_bridge"):
                bridges.append(
                    GeneticTarget(
                        symbol=node,
                        ensembl_id=attrs.get("ensembl_id", "ENSG00000000000"),
                        druggability_score=attrs.get("druggability", 0.5),
                        novelty_score=attrs.get("novelty", 0.8),
                    )
                )
        return bridges


class LocalSearchClient(SearchClient):
    """Local implementation of SearchClient."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}

    def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]:
        """Finds inconsistencies in literature regarding a topic."""
        # Return pre-loaded gaps if they match the topic
        results = []
        for gap_data in self.data.get("inconsistencies", []):
            if topic.lower() in gap_data["description"].lower():
                results.append(
                    KnowledgeGap(description=gap_data["description"], source_nodes=gap_data.get("source_nodes"))
                )
        return results

    def verify_citation(self, interaction_claim: str) -> bool:
        """Verifies if a claimed interaction is supported by literature."""
        # Simple keyword check against 'valid_claims' list in data
        valid_claims = self.data.get("valid_claims", [])
        return interaction_claim in valid_claims

    def check_disconfirming_evidence(self, interaction_claim: str) -> bool:
        """
        Checks for evidence that explicitly contradicts the claim.
        Returns True if disconfirming evidence is found.
        """
        # Look for the claim in a 'disproven_claims' list
        disproven_claims = self.data.get("disproven_claims", [])
        return interaction_claim in disproven_claims


class LocalCodexClient(CodexClient):
    """Local implementation of CodexClient."""

    def __init__(self, similarities: Optional[Dict[str, float]] = None):
        self.similarities = similarities or {}

    def get_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """Calculates semantic similarity between two entities."""
        key = f"{entity1}:{entity2}"
        reverse_key = f"{entity2}:{entity1}"
        return self.similarities.get(key, self.similarities.get(reverse_key, 0.0))

    def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        """Validates a genetic target and returns its details."""
        # Always valid in this mock unless symbol is "INVALID"
        if symbol == "INVALID":
            return None
        return GeneticTarget(
            symbol=symbol,
            ensembl_id=f"ENS_{symbol}",
            druggability_score=0.5,  # Default, can be overridden by Prism
            novelty_score=0.5,
        )


class LocalPrismClient(PrismClient):
    """Local implementation of PrismClient."""

    def __init__(self, scores: Optional[Dict[str, float]] = None):
        self.scores = scores or {}

    def check_druggability(self, target_id: str) -> float:
        """Returns a druggability score for the target."""
        return self.scores.get(target_id, 0.1)  # Default low score


class LocalInferenceClient(InferenceClient):
    """Local implementation of InferenceClient."""

    def __init__(self, results: Optional[Dict[str, float]] = None):
        self.results = results or {}

    def run_counterfactual_simulation(self, mechanism: str, intervention_target: str) -> float:
        """
        Runs a counterfactual simulation.
        Returns a plausibility score (0.0 - 1.0).
        """
        key = f"{intervention_target}->{mechanism}"
        # We allow partial matching for flexibility in testing
        if key in self.results:
            return self.results[key]
        return 0.0


class LocalVeritasClient(VeritasClient):
    """Local implementation of VeritasClient (The Log)."""

    def __init__(self) -> None:
        self.logs: List[Dict[str, Any]] = []

    def log_trace(self, hypothesis_id: str, trace_data: Dict[str, Any]) -> None:
        """Logs the hypothesis generation trace."""
        self.logs.append({"hypothesis_id": hypothesis_id, "trace_data": trace_data})
