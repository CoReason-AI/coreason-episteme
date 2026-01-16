import networkx as nx

from coreason_episteme.adapters.local_clients import (
    LocalCodexClient,
    LocalGraphNexusClient,
    LocalInferenceClient,
    LocalPrismClient,
    LocalSearchClient,
    LocalVeritasClient,
)


def test_local_graph_nexus_client() -> None:
    g = nx.Graph()
    g.add_node("A", type="protein")
    g.add_node("B", type="protein")
    g.add_edge("A", "B")

    g.add_node("C", type="disease")

    # Bridge node
    g.add_node("Bridge", is_bridge=True, ensembl_id="ENSG_BRIDGE")

    client = LocalGraphNexusClient(g)

    # Test find_disconnected_clusters
    clusters = client.find_disconnected_clusters({"type": "protein"})
    # Should find the A-B cluster
    assert len(clusters) >= 1
    # Check if cluster contains A and B
    cluster_nodes = [set(c["nodes"]) for c in clusters]
    assert {"A", "B"} in cluster_nodes or any({"A", "B"}.issubset(s) for s in cluster_nodes)

    # Test find_latent_bridges
    bridges = client.find_latent_bridges("cluster_0", "cluster_1")
    assert len(bridges) == 1
    assert bridges[0].symbol == "Bridge"


def test_local_search_client() -> None:
    data = {
        "inconsistencies": [{"description": "Drug X inhibits Pathway Y but...", "source_nodes": ["Drug X"]}],
        "valid_claims": ["GeneA interacts with GeneB"],
    }
    client = LocalSearchClient(data)

    gaps = client.find_literature_inconsistency("Drug X")
    assert len(gaps) == 1
    assert gaps[0].description == "Drug X inhibits Pathway Y but..."

    assert client.verify_citation("GeneA interacts with GeneB") is True
    assert client.verify_citation("GeneA interacts with GeneC") is False


def test_local_codex_client() -> None:
    sims = {"A:B": 0.8}
    client = LocalCodexClient(sims)

    assert client.get_semantic_similarity("A", "B") == 0.8
    assert client.get_semantic_similarity("B", "A") == 0.8
    assert client.get_semantic_similarity("A", "C") == 0.0

    target = client.validate_target("GeneA")
    assert target is not None
    assert target.symbol == "GeneA"
    assert client.validate_target("INVALID") is None


def test_local_prism_client() -> None:
    scores = {"GeneA": 0.9}
    client = LocalPrismClient(scores)

    assert client.check_druggability("GeneA") == 0.9
    assert client.check_druggability("GeneB") == 0.1


def test_local_inference_client() -> None:
    results = {"TargetA->MechB": 0.75}
    client = LocalInferenceClient(results)

    assert client.run_counterfactual_simulation("MechB", "TargetA") == 0.75
    assert client.run_counterfactual_simulation("MechC", "TargetA") == 0.0


def test_local_veritas_client() -> None:
    client = LocalVeritasClient()
    client.log_trace("hyp123", {"data": "test"})
    assert len(client.logs) == 1
    assert client.logs[0]["hypothesis_id"] == "hyp123"
    assert client.logs[0]["trace_data"] == {"data": "test"}
