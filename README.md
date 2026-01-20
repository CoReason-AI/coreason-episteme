# coreason-episteme

**Theorist / Hypothesis Engine**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_episteme/blob/main/LICENSE)
[![CI/Status](https://github.com/CoReason-AI/coreason_episteme/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_episteme/actions/workflows/ci-cd.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-product__requirements-blue)](docs/product_requirements.md)

## Overview

**coreason-episteme** is the engine of **Scientific Intuition** within the CoReason platform. It acts as the "Theorist" that automates the "Eureka" moment by systematically scanning for "White Space" in the Knowledge Graph. It synthesizes weak signals from disparate papers into coherent, testable hypotheses (e.g., identifying a novel genetic target for a disease based on pathway adjacency).

Crucially, it implements a **"Null Hypothesis First"** architecture. Every generated hypothesis is treated as false until it survives a rigorous adversarial review process involving causal counterfactuals and patent whitespace analysis.

## Core Philosophy

"Identify the Void. Bridge the Gap. Propose the Novel."

1.  **Gap Scanning (Negative Space Analysis):** Identifies disconnected clusters in the Knowledge Graph that share high semantic similarity but lack connections.
2.  **Latent Bridging (The Leap):** Finds "Latent Bridges"—genes or proteins that are structurally or functionally connected to the conflict but rarely mentioned in the context of the specific disease.
3.  **Causal Simulation (The Test):** Runs a counterfactual simulation using `coreason-inference` to validate the hypothesis.
4.  **Adversarial Review (The Council):** Convenes a virtual "Review Board" (The Toxicologist, The Clinician, The IP Strategist) to attack the hypothesis for safety risks, clinical redundancy, or patent infringement.

## Installation

```bash
pip install coreason-episteme
```

## Features

*   **Gap Scanner:** Detects inconsistencies or missing links in the Knowledge Graph and literature.
*   **Bridge Builder:** Formulates novel hypotheses by exploring the "Neighborhood" of conflicting entities and proposing valid genetic targets.
*   **Causal Validator:** Validates proposed mechanisms using counterfactual simulations (A -> B -> C).
*   **Protocol Designer:** Designs the "Killer Experiment" (PICO) to prove or disprove the hypothesis in a wet lab.
*   **Adversarial Reviewer:** Critiques hypotheses using multi-perspective strategies (Toxicology, IP, Clinical, Scientific Skeptic).

For detailed requirements, see [Product Requirements](docs/product_requirements.md).

## Usage

```python
from typing import Any, Dict, List, Optional
from coreason_episteme.main import generate_hypothesis
from coreason_episteme.models import GeneticTarget, KnowledgeGap

# 1. Define Mock Clients (Simulate external services)
class MockClient:
    def find_disconnected_clusters(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{"cluster_a_id": "C1", "cluster_b_id": "C2", "cluster_a_name": "A", "cluster_b_name": "B"}]

    def find_latent_bridges(self, source: str, target: str) -> List[GeneticTarget]:
        return [GeneticTarget(symbol="GENE_X", ensembl_id="ENSG001", druggability_score=0.9, novelty_score=0.8)]

    def get_semantic_similarity(self, e1: str, e2: str) -> float: return 0.95
    def validate_target(self, symbol: str) -> Optional[GeneticTarget]:
        return GeneticTarget(symbol=symbol, ensembl_id="ENSG001", druggability_score=0.9, novelty_score=0.8)

    def find_literature_inconsistency(self, topic: str) -> List[KnowledgeGap]: return []
    def verify_citation(self, claim: str) -> bool: return True
    def check_patent_infringement(self, target: GeneticTarget, mech: str) -> List[str]: return []
    def find_disconfirming_evidence(self, sub: str, obj: str, act: str) -> List[str]: return []

    def check_druggability(self, target_id: str) -> float: return 0.9

    def run_counterfactual_simulation(self, mech: str, target: str) -> float: return 0.85
    def run_toxicology_screen(self, target: GeneticTarget) -> List[str]: return []
    def check_clinical_redundancy(self, mech: str, target: GeneticTarget) -> List[str]: return []

    def log_trace(self, h_id: str, trace: Dict[str, Any]) -> None: print(f"Logged trace for {h_id}")

# 2. Instantiate Clients
mock_client = MockClient()

# 3. Run the Engine
try:
    hypotheses = generate_hypothesis(
        disease_id="DOID:12345",
        graph_client=mock_client,      # type: ignore
        codex_client=mock_client,      # type: ignore
        search_client=mock_client,     # type: ignore
        prism_client=mock_client,      # type: ignore
        inference_client=mock_client,  # type: ignore
        veritas_client=mock_client     # type: ignore
    )

    for h in hypotheses:
        print(f"Hypothesis: {h.title}")
        print(f"Mechanism: {h.proposed_mechanism}")
        print(f"Confidence: {h.confidence}")

except Exception as e:
    print(f"Error generating hypotheses: {e}")
```

## License

This project is licensed under the **Prosperity Public License 3.0**.
