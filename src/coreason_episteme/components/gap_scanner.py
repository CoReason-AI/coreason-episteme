# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from typing import List

from coreason_episteme.interfaces import (
    CodexClient,
    GraphNexusClient,
    SearchClient,
)
from coreason_episteme.models import KnowledgeGap
from coreason_episteme.utils.logger import logger


class GapScannerImpl:
    """
    Implementation of the GapScanner (The Void Detector).
    Identifies 'Negative Space' in the Knowledge Graph and Literature.
    """

    def __init__(
        self,
        graph_client: GraphNexusClient,
        search_client: SearchClient,
        codex_client: CodexClient,
        similarity_threshold: float = 0.7,
    ):
        self.graph_client = graph_client
        self.search_client = search_client
        self.codex_client = codex_client
        self.similarity_threshold = similarity_threshold

    def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Scans for knowledge gaps related to the target.

        Args:
            target: The disease or biological entity to scan for.

        Returns:
            A list of KnowledgeGap objects.
        """
        logger.info(f"Scanning for knowledge gaps for target: {target}")
        gaps: List[KnowledgeGap] = []

        # 1. Literature Discrepancy
        logger.debug(f"Scanning literature inconsistencies for {target}")
        lit_gaps = self.search_client.find_literature_inconsistency(target)
        gaps.extend(lit_gaps)
        logger.info(f"Found {len(lit_gaps)} literature gaps.")

        # 2. Cluster Analysis (Graph Gaps)
        # Find disconnected clusters related to the target
        logger.debug(f"Scanning disconnected clusters for {target}")
        clusters = self.graph_client.find_disconnected_clusters({"topic": target})

        # Compare clusters pairwise to find "High Similarity but Disconnected"
        for i, cluster_a in enumerate(clusters):
            for j, cluster_b in enumerate(clusters):
                if i >= j:
                    continue

                # Heuristic: Represent the cluster by its first node or ID for semantic check
                # In a real system, we'd use a cluster centroid or summary vector.
                # Assuming cluster dict has 'id' and possibly 'nodes'
                id_a = cluster_a.get("id", f"cluster_{i}")
                id_b = cluster_b.get("id", f"cluster_{j}")

                # Ideally, we check similarity between the concepts represented by the clusters.
                # Here we use the IDs or a representative node.
                # Let's assume the cluster ID is meaningful or we pick a node.
                # Using the target vs cluster might be more appropriate?
                # PRD says: "Identifies two densely connected clusters..."
                # (e.g., 'Protein Family A' and 'Disease Pathway B')
                # So we compare Cluster A and Cluster B.

                # If the cluster ID is just "cluster_0", we can't do semantic similarity.
                # We need the entity name. Let's assume 'nodes' list exists.
                nodes_a = cluster_a.get("nodes", [])
                nodes_b = cluster_b.get("nodes", [])

                if not nodes_a or not nodes_b:
                    continue

                # Use first node as proxy for cluster concept
                concept_a = str(nodes_a[0])
                concept_b = str(nodes_b[0])

                similarity = self.codex_client.get_semantic_similarity(concept_a, concept_b)

                if similarity >= self.similarity_threshold:
                    logger.info(f"Found structural gap between {concept_a} and {concept_b} (sim={similarity})")
                    description = (
                        f"Structural Gap: High semantic similarity ({similarity:.2f}) between "
                        f"{concept_a} (Cluster {id_a}) and {concept_b} (Cluster {id_b}) "
                        f"but no direct connection found."
                    )
                    gaps.append(KnowledgeGap(description=description, source_nodes=[concept_a, concept_b]))

        return gaps
