# The Architecture and Utility of coreason-episteme

### 1. The Philosophy (The Why)

Scientific discovery has historically been a game of serendipity—a researcher connecting two seemingly unrelated facts in a moment of brilliance. However, the sheer volume of modern biological literature makes this "manual" intuition increasingly inefficient. Critical connections lie dormant in the "white space" between published papers, invisible to human review.

**coreason-episteme** codifies this "Eureka" moment. It is not designed to confirm what we already know, but to systematically hunt for what is missing. By treating the absence of connections in a Knowledge Graph as a "Negative Space" signal, it identifies logical gaps where a biological link *should* exist but hasn't been documented.

The system adopts a "Strict Null Hypothesis" philosophy: every generated idea is presumed false until proven otherwise. It doesn't just suggest a drug target; it actively tries to kill its own ideas through simulated adversarial review—acting as the Skeptic, the Toxicologist, and the IP Lawyer before a human ever sees the proposal.

### 2. Under the Hood (The Dependencies & logic)

The package operates as a high-level orchestration engine, relying on a robust stack of internal and external services:

*   **`pydantic` & `pydantic-settings`:** These form the rigid backbone of the system. In scientific computing, ambiguity is fatal. Pydantic ensures that every `Hypothesis`, `GeneticTarget`, and `ClinicalTrial` adheres to a strict schema, preventing "hallucinated" data structures from propagating through the pipeline.
*   **Protocol Interfaces (`interfaces.py`):** The system uses dependency injection to interface with its "senses":
    *   **The Map (`coreason-graph-nexus`):** For traversing the knowledge graph.
    *   **The Lab (`coreason-inference`):** For running causal counterfactuals (e.g., "If we inhibit X, does Y change?").
    *   **The Scout (`coreason-search`):** For validating claims against raw literature.
*   **The Refinement Loop (`EpistemeEngine`):** The core logic isn't a linear pipeline; it's a loop. If the `AdversarialReviewer` finds a fatal flaw (e.g., liver toxicity), the engine doesn't just fail—it refines. It adds the rejected target to an exclusion list and re-runs the `BridgeBuilder` to find an alternative mechanism, mimicking the iterative process of a real research scientist.

### 3. In Practice (The How)

In the happy path, `coreason-episteme` takes a disease ID and orchestrates the entire discovery process, from finding a gap to designing the "Killer Experiment."

```python
from coreason_episteme.main import generate_hypothesis
from coreason_episteme.models import Hypothesis

# In a real deployment, these clients would be fully initialized services.
# The engine uses dependency injection to wire them together.
hypotheses = generate_hypothesis(
    disease_id="MONDO:0005148",  # Type 2 Diabetes
    graph_client=production_graph_client,
    codex_client=production_codex_client,
    search_client=production_search_client,
    prism_client=production_prism_client,
    inference_client=production_inference_client,
    veritas_client=production_veritas_client,
)

for hypothesis in hypotheses:
    print(f"Title: {hypothesis.title}")
    print(f"Gap: {hypothesis.knowledge_gap}")
    print(f"Proposed Mechanism: {hypothesis.proposed_mechanism}")

    # The system automatically generates the PICO formatted experiment
    pico = hypothesis.killer_experiment_pico
    print(f"Experiment: Treat {pico['population']} with {pico['intervention']}...")
```

The system also provides granular control via the `EpistemeEngine` class for advanced workflows, allowing researchers to inspect the "trace" of why specific hypotheses were rejected or refined.

```python
from coreason_episteme.engine import EpistemeEngine
from coreason_episteme.components.gap_scanner import GapScannerImpl
from coreason_episteme.models import HypothesisTrace

# Accessing the engine directly allows for custom retry logic or component swapping
engine = EpistemeEngine(
    gap_scanner=GapScannerImpl(...),
    # ... other components
    veritas_client=audit_logger,
)

results = engine.run("MONDO:0004975")  # Alzheimer's Disease

# Each result is a survivor of the "Adversarial Review"
# The 'veritas_client' has already logged the full decision tree (trace)
# for every rejected attempt during this run.
```
