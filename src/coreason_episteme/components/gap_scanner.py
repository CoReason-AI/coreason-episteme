# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_episteme

from dataclasses import dataclass, field
from typing import List

from coreason_episteme.config import settings
from coreason_episteme.interfaces import CodexClient, GraphNexusClient, SearchClient
from coreason_episteme.models import KnowledgeGap, KnowledgeGapType
from coreason_episteme.utils.logger import logger


@dataclass
class GapScannerImpl:
    """
    Implementation of the GapScanner (The Void Detector).

    Identifies "Negative Space" in the Knowledge Graph by finding disconnected
    clusters that should be connected based on semantic similarity, or by
    finding explicit inconsistencies in the literature.

    Attributes:
        graph_client: Client for GraphNexus.
        codex_client: Client for Codex.
        search_client: Client for Search.
        similarity_threshold: Threshold for semantic similarity to consider a disconnect significant.
    """

    graph_client: GraphNexusClient
    codex_client: CodexClient
    search_client: SearchClient
    similarity_threshold: float = field(default_factory=lambda: settings.GAP_SCANNER_SIMILARITY_THRESHOLD)

    def scan(self, target: str) -> List[KnowledgeGap]:
        """
        Scans for knowledge gaps (Negative Space Analysis).

        Steps:
        1. Cluster Analysis: Finds disconnected subgraphs in GraphNexus.
           Checks semantic similarity via Codex.
        2. Literature Discrepancy: Queries SearchClient for inconsistencies.

        Args:
            target: The disease or biological entity to scan for.

        Returns:
            A list of KnowledgeGap objects representing identified gaps.
        """
        logger.info(f"Scanning for knowledge gaps related to {target}...")
        gaps: List[KnowledgeGap] = []

        # 1. Cluster Analysis
        logger.debug(f"Querying GraphNexus for disconnected clusters for {target}...")
        raw_clusters = self.graph_client.find_disconnected_clusters({"target": target})
        logger.debug(f"Found {len(raw_clusters)} potential disconnected cluster pairs.")

        for pair in raw_clusters:
            cluster_a_id = pair.get("cluster_a_id")
            cluster_b_id = pair.get("cluster_b_id")
            cluster_a_name = pair.get("cluster_a_name", "Unknown")
            cluster_b_name = pair.get("cluster_b_name", "Unknown")

            if cluster_a_id and cluster_b_id:
                similarity = self.codex_client.get_semantic_similarity(cluster_a_id, cluster_b_id)
                if similarity >= self.similarity_threshold:
                    logger.info(
                        f"Found disconnect with high similarity ({similarity}): {cluster_a_name} <-> {cluster_b_name}"
                    )
                    description = (
                        f"Cluster Disconnect: {cluster_a_name} and {cluster_b_name} "
                        f"are similar ({similarity}) but unconnected."
                    )
                    gaps.append(
                        KnowledgeGap(
                            description=description,
                            type=KnowledgeGapType.CLUSTER_DISCONNECT,
                            source_nodes=[cluster_a_id, cluster_b_id],
                        )
                    )

        # 2. Literature Discrepancy
        logger.debug(f"Searching for literature inconsistencies for {target}...")
        lit_gaps = self.search_client.find_literature_inconsistency(target)
        if lit_gaps:
            logger.info(f"Found {len(lit_gaps)} literature inconsistencies.")
            gaps.extend(lit_gaps)

        logger.info(f"Total gaps found for {target}: {len(gaps)}")
        return gaps
