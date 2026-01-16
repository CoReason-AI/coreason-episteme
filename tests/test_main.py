import pytest

from coreason_episteme.main import EpistemeEngine


def test_episteme_engine_e2e() -> None:
    """
    End-to-End test for EpistemeEngine.
    We populate the local clients with data that guarantees a result.
    """
    engine = EpistemeEngine()

    # 1. Setup Graph with Disconnected Clusters (Gap)
    g = engine.graph_client.graph
    # Cluster A: Target Disease Context
    g.add_node("DiseaseX", type="disease")
    g.add_node("GeneA", type="gene")
    g.add_edge("DiseaseX", "GeneA")

    # Cluster B: Some Pathway
    g.add_node("PathwayY", type="pathway")
    g.add_node("GeneB", type="gene")
    g.add_edge("PathwayY", "GeneB")

    # Add a latent bridge node
    g.add_node("BridgeGene", is_bridge=True, ensembl_id="ENSG_BRIDGE", druggability=0.8)

    # 2. Setup Semantic Similarity to trigger Gap detection
    # GapScanner checks similarity between cluster representatives.
    # In our simplified logic, it picks the first node.
    # Since set iteration is arbitrary, we must ensure any combination matches
    # or at least the likely ones are covered.
    engine.codex_client.similarities = {
        "DiseaseX:PathwayY": 0.9,
        "DiseaseX:GeneB": 0.9,
        "GeneA:PathwayY": 0.9,
        "GeneA:GeneB": 0.9,
    }

    # 3. Setup Bridge Builder requirements
    # BridgeBuilder looks for bridges between source nodes of the gap.
    # Gap source nodes will be [DiseaseX, PathwayY] (or similar).
    # LocalGraphNexusClient.find_latent_bridges logic needs to find 'BridgeGene'.
    # Our mock implementation of find_latent_bridges just returns nodes with `is_bridge=True`.

    # 4. Setup Prism (Druggability)
    engine.prism_client.scores = {"ENSG_BRIDGE": 0.9}

    # 5. Setup Search (Verification)
    # BridgeBuilder checks citation for "Source interacts with Bridge and Bridge affects Target"
    # Because GapScanner might pick any node as representative, we need to allow for all possibilities
    engine.search_client.data["valid_claims"] = [
        "DiseaseX interacts with BridgeGene and BridgeGene affects PathwayY",
        "DiseaseX interacts with BridgeGene and BridgeGene affects GeneB",
        "GeneA interacts with BridgeGene and BridgeGene affects PathwayY",
        "GeneA interacts with BridgeGene and BridgeGene affects GeneB",
    ]

    # 6. Setup Inference (Validation)
    # Mechanism: "Regulation of PathwayY via BridgeGene..."
    # Again, source node varies.
    engine.inference_client.results = {
        "BridgeGene->Regulation of PathwayY via BridgeGene (bridging from DiseaseX).": 0.85,
        "BridgeGene->Regulation of PathwayY via BridgeGene (bridging from GeneA).": 0.85,
        "BridgeGene->Regulation of GeneB via BridgeGene (bridging from DiseaseX).": 0.85,
        "BridgeGene->Regulation of GeneB via BridgeGene (bridging from GeneA).": 0.85,
    }

    # Execute
    hypotheses = engine.generate_hypothesis("DiseaseX")

    # Verify
    assert len(hypotheses) == 1
    h = hypotheses[0]
    assert h.target_candidate.symbol == "BridgeGene"
    assert h.causal_validation_score == 0.85
    assert h.killer_experiment_pico["Population"] is not None


def test_episteme_engine_no_gaps() -> None:
    engine = EpistemeEngine()
    hypotheses = engine.generate_hypothesis("PerfectlyKnownDisease")
    assert len(hypotheses) == 0


def test_episteme_engine_low_score_filter() -> None:
    engine = EpistemeEngine()

    # Setup Gap
    g = engine.graph_client.graph
    g.add_node("A")
    g.add_node("B")
    engine.codex_client.similarities = {"A:B": 0.9}

    # Setup Bridge
    g.add_node("Bridge", is_bridge=True, ensembl_id="ENS", druggability=0.9)
    engine.prism_client.scores = {"ENS": 0.9}
    engine.search_client.data["valid_claims"] = ["A interacts with Bridge and Bridge affects B"]

    # Setup Low Score
    key = "Bridge->Regulation of B via Bridge (bridging from A)."
    engine.inference_client.results = {key: 0.2}

    hypotheses = engine.generate_hypothesis("A")
    assert len(hypotheses) == 0


def test_episteme_engine_invalid_config() -> None:
    with pytest.raises(NotImplementedError):
        EpistemeEngine(use_local_defaults=False)


def test_episteme_engine_bridge_failure() -> None:
    """Test where gap is found but bridge building fails (e.g. no druggable targets)."""
    engine = EpistemeEngine()

    # 1. Setup Gap
    g = engine.graph_client.graph
    g.add_node("A")
    g.add_node("B")
    engine.codex_client.similarities = {"A:B": 0.9}

    # 2. No Bridges added to graph

    hypotheses = engine.generate_hypothesis("A")
    assert len(hypotheses) == 0


def test_episteme_engine_confidence_levels() -> None:
    """Test that confidence level is set correctly based on score."""
    engine = EpistemeEngine()
    # Setup Gap & Bridge
    g = engine.graph_client.graph
    g.add_node("A")
    g.add_node("B")
    g.add_node("Bridge", is_bridge=True, ensembl_id="ENS", druggability=0.9)
    engine.codex_client.similarities = {"A:B": 0.9}
    engine.prism_client.scores = {"ENS": 0.9}
    engine.search_client.data["valid_claims"] = ["A interacts with Bridge and Bridge affects B"]

    # Case 1: > 0.8 -> PROBABLE
    engine.inference_client.results = {"Bridge->Regulation of B via Bridge (bridging from A).": 0.85}
    hyps = engine.generate_hypothesis("A")
    assert hyps[0].confidence == "PROBABLE"

    # Case 2: > 0.5 but <= 0.8 -> PLAUSIBLE
    engine.inference_client.results = {"Bridge->Regulation of B via Bridge (bridging from A).": 0.6}
    hyps = engine.generate_hypothesis("A")
    assert hyps[0].confidence == "PLAUSIBLE"
