# coreason-episteme

**Theorist / Hypothesis Engine**

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_episteme/blob/main/LICENSE)
[![CI/CD](https://github.com/CoReason-AI/coreason_episteme/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_episteme/actions/workflows/ci-cd.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/CoReason-AI/coreason_episteme)

## Overview

**coreason-episteme** is the engine of **Scientific Intuition** within the CoReason platform. It acts as the "Theorist" that automates the "Eureka" moment by systematically scanning for "White Space" in the Knowledge Graph. It synthesizes weak signals from disparate papers into coherent, testable hypotheses (e.g., identifying a novel genetic target for a disease based on pathway adjacency).

Crucially, it implements a **"Null Hypothesis First"** architecture. Every generated hypothesis is treated as false until it survives a rigorous adversarial review process involving causal counterfactuals and patent whitespace analysis.

## Core Philosophy

"Identify the Void. Bridge the Gap. Propose the Novel."

1.  **Gap Scanning (Negative Space Analysis):** Identifies disconnected clusters in the Knowledge Graph that share high semantic similarity but lack connections.
2.  **Latent Bridging (The Leap):** Finds "Latent Bridges"—genes or proteins that are structurally or functionally connected to the conflict but rarely mentioned in the context of the specific disease.
3.  **Causal Simulation (The Test):** Runs a counterfactual simulation using `coreason-inference` to validatethe hypothesis.
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

## Usage

```python
import logging
from typing import List, Optional

# Import coreason-episteme entry point and models
from coreason_episteme.main import generate_hypothesis
from coreason_episteme.models import Hypothesis

# Import or Mock external clients (Protocol interfaces)
from coreason_episteme.interfaces import (
    GraphNexusClient, CodexClient, SearchClient,
    PrismClient, InferenceClient, VeritasClient
)

# NOTE: In a real application, you would instantiate concrete clients
# that connect to the respective microservices.
class MockGraphClient:
    def find_disconnected_clusters(self, criteria): return []
    def find_latent_bridges(self, source, target): return []

# ... Instantiate other clients similarly ...

def main():
    # Instantiate clients (replace Mocks with real implementations)
    graph_client = MockGraphClient() # type: ignore
    codex_client = ... # Instantiate CodexClient
    search_client = ... # Instantiate SearchClient
    prism_client = ... # Instantiate PrismClient
    inference_client = ... # Instantiate InferenceClient
    veritas_client = ... # Instantiate VeritasClient

    disease_id = "DOID:12345" # Example Disease ID

    try:
        hypotheses: List[Hypothesis] = generate_hypothesis(
            disease_id=disease_id,
            graph_client=graph_client,
            codex_client=codex_client, # type: ignore
            search_client=search_client, # type: ignore
            prism_client=prism_client, # type: ignore
            inference_client=inference_client, # type: ignore
            veritas_client=veritas_client # type: ignore
        )

        for h in hypotheses:
            print(f"Hypothesis: {h.title}")
            print(f"Mechanism: {h.proposed_mechanism}")
            print(f"Confidence: {h.confidence}")

    except Exception as e:
        print(f"Error generating hypotheses: {e}")

if __name__ == "__main__":
    main()
```

## License

This project is licensed under the **Prosperity Public License 3.0**.
See [LICENSE](LICENSE) for more details.
