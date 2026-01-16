from unittest.mock import MagicMock

import networkx as nx

from coreason_episteme.adapters.local_clients import (
    LocalCodexClient,
    LocalGraphNexusClient,
    LocalSearchClient,
)
from coreason_episteme.components.gap_scanner import GapScannerImpl


def test_gap_scanner_literature_inconsistency() -> None:
    # Setup
    search_data = {"inconsistencies": [{"description": "Gap 1 for TargetX", "source_nodes": ["P1"]}]}
    search_client = LocalSearchClient(search_data)
    graph_client = LocalGraphNexusClient()  # Empty graph
    codex_client = LocalCodexClient()

    scanner = GapScannerImpl(graph_client, search_client, codex_client)

    # Execute
    gaps = scanner.scan("TargetX")

    # Verify
    assert len(gaps) == 1
    assert gaps[0].description == "Gap 1 for TargetX"


def test_gap_scanner_cluster_gap() -> None:
    # Setup
    # 1. Create a graph with two disconnected components
    g = nx.Graph()
    # Cluster A
    g.add_node("ProteinA", type="protein")
    # Cluster B
    g.add_node("PathwayB", type="pathway")

    graph_client = LocalGraphNexusClient(g)

    # 2. Setup semantic similarity to be high
    codex_client = LocalCodexClient({"ProteinA:PathwayB": 0.9})

    search_client = LocalSearchClient()

    scanner = GapScannerImpl(graph_client, search_client, codex_client, similarity_threshold=0.8)

    # Execute
    # We pass "ProteinA" as target, logic finds clusters related to it.
    # LocalGraphNexusClient.find_disconnected_clusters returns all components in the simplified version.
    gaps = scanner.scan("ProteinA")

    # Verify
    # Should find a gap between ProteinA and PathwayB because similarity 0.9 > 0.8
    assert len(gaps) == 1
    assert "ProteinA" in gaps[0].description
    assert "PathwayB" in gaps[0].description
    assert gaps[0].source_nodes == ["ProteinA", "PathwayB"]


def test_gap_scanner_no_gap_low_similarity() -> None:
    # Setup
    g = nx.Graph()
    g.add_node("ProteinA")
    g.add_node("RandomC")

    graph_client = LocalGraphNexusClient(g)
    codex_client = LocalCodexClient({"ProteinA:RandomC": 0.1})
    search_client = LocalSearchClient()

    scanner = GapScannerImpl(graph_client, search_client, codex_client, similarity_threshold=0.8)

    gaps = scanner.scan("ProteinA")

    assert len(gaps) == 0


def test_gap_scanner_empty() -> None:
    scanner = GapScannerImpl(LocalGraphNexusClient(), LocalSearchClient(), LocalCodexClient())
    gaps = scanner.scan("Nothing")
    assert len(gaps) == 0


def test_gap_scanner_empty_cluster_handling() -> None:
    # Setup: One valid cluster, one empty cluster
    g = nx.Graph()
    g.add_node("A")
    # We mock the client to return an empty cluster structure
    # Since LocalGraphNexusClient computes clusters from the graph,
    # we can just add a node to the graph to get one cluster,
    # and we rely on the fact that if we had logic to produce empty clusters it would be handled.
    # To strictly test the 'if not nodes_a' line, we might need to mock find_disconnected_clusters directly.

    mock_graph_client = MagicMock()
    # Return 2 clusters, one with nodes, one without
    mock_graph_client.find_disconnected_clusters.return_value = [
        {"id": "c1", "nodes": ["A"]},
        {"id": "c2", "nodes": []},
    ]

    scanner = GapScannerImpl(mock_graph_client, LocalSearchClient(), LocalCodexClient())

    gaps = scanner.scan("Target")
    # Should not crash, should just skip comparison involving c2
    assert len(gaps) == 0
