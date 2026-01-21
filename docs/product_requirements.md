# Product Requirements Document: coreason-episteme

**Domain:** Automated Scientific Discovery, Causal Hypothesis Generation & Literature Synthesis
**Architectural Role:** The "Theorist" / The Hypothesis Engine
**Core Philosophy:** "Identify the Void. Bridge the Gap. Propose the Novel."
**Dependencies:** coreason-search (Scout), coreason-graph-nexus (Traversal), coreason-inference (Validation), coreason-codex (Ontology)

## 1. Executive Summary

coreason-episteme is the engine of **Scientific Intuition** within the CoReason platform. While coreason-inference mathematically validates if A causes B, coreason-episteme is responsible for *guessing* that A might be related to B in the first place.

It automates the "Eureka" moment by systematically scanning for **"White Space"** in the Knowledge Graph—areas where connections *should* exist but are missing from the literature. It synthesizes weak signals from disparate papers into coherent, testable hypotheses (e.g., identifying a novel genetic target for a disease based on pathway adjacency).

Crucially, it implements a **"Null Hypothesis First"** architecture. Every generated hypothesis is treated as false until it survives a rigorous adversarial review process involving causal counterfactuals and patent whitespace analysis.

## 2. Functional Philosophy

The agent must implement the **Scan-Bridge-Simulate-Critique Loop**:

1.  **Gap Scanning (Negative Space Analysis):** The system does not just read what *is* known; it looks for what is *missing*. It identifies two densely connected clusters in the Knowledge Graph (e.g., "Protein Family A" and "Disease Pathway B") that have zero edges between them, despite structural similarity suggesting a link.
2.  **Latent Bridging (The Leap):** It uses coreason-graph-nexus to find "Latent Bridges"—genes or proteins that are structurally or functionally connected to the conflict but rarely mentioned in the context of the specific disease.
3.  **Causal Simulation (The Test):** It calls coreason-inference to run a counterfactual simulation: *"If this hypothesis were true, and we inhibited Gene X, what would the downstream biomarkers look like?"*
4.  **Adversarial Review (The Council):** It convenes a virtual "Review Board" (The Toxicologist, The Clinician, The IP Strategist) to attack the hypothesis for safety risks, clinical redundancy, or patent infringement.

## 3. Core Functional Requirements (Component Level)

### 3.1 The Gap Scanner (The Void Detector)

**Concept:** Identifies "Negative Space" in the Knowledge Graph.

*   **Mechanism:**
    *   **Cluster Analysis:** Finds disconnected subgraphs in coreason-graph-nexus that share high semantic similarity vectors (via coreason-codex).
    *   **Literature Discrepancy:** Uses coreason-search to find "The Inconsistency"—two reliable sources that disagree, or a clinical phenomenon that current biological models fail to explain (e.g., *"Drug X inhibits Pathway Y, yet patients still show symptoms of Z"*).
*   **Output:** A KnowledgeGap object defining the specific inconsistency or missing link.

### 3.2 The Bridge Builder (Hypothesis Formulator)

**Concept:** Generates the narrative linking the gap.

*   **Action:**
    *   **Multi-Hop Reasoning:** Explores the "Neighborhood" of the conflicting entities.
    *   **Synthesis:** Proposes a **Novel Genetic Target** or mechanism that resolves the inconsistency.
    *   *Example:* "Protein A might regulate Pathway B via the obscure Kinase Z, which explains the side effect seen in the 1980 paper."
*   **Constraint:** Calls coreason-prism to ensure the proposed target is "Druggable" (structurally viable for binding), filtering out biologically interesting but pharmaceutically useless targets.

### 3.3 The Causal Validator (The Simulator)

**Concept:** Tests the hypothesis against known causal models.

*   **Input:** The proposed mechanism (A $\to$ B $\to$ C).
*   **Mechanism:** Calls coreason-inference to construct a Directed Cyclic Graph (DCG).
*   **Simulation:** Runs a **Counterfactual**: *"If we inhibit Target Gene [A], does it causally interrupt the disease pathway [C] without triggering known toxicity loops?"*
*   **Scoring:** Assigns a `Causal_Plausibility_Score` (0.0 - 1.0). Hypotheses scoring below threshold are discarded before human review.

### 3.4 The Protocol Designer (The Experimentalist)

**Concept:** Defines how to prove the hypothesis in the real world.

*   **Output:** Designs the *exact* **"Killer Experiment"** (Wet Lab) required to prove or disprove the hypothesis.
*   **Structure:** Defines the **PICO** (Population, Intervention, Comparator, Outcome) structure for the validation experiment.
*   **Value:** Moves the output from "Abstract Theory" to "Actionable R&D Plan."

## 4. Integration Requirements

*   **coreason-graph-nexus (The Map):**
    *   episteme queries nexus to find the "Latent Bridges" and disconnected clusters.
*   **coreason-inference (The Lab):**
    *   episteme sends candidate mechanisms to inference for "In-Silico Stress Testing" via Neural ODEs.
*   **coreason-codex (The Dictionary):**
    *   Ensures all proposed targets and pathways map to valid OMOP/Ensembl IDs.
*   **coreason-veritas (The Log):**
    *   Logs the "Hypothesis Generation Trace"—recording exactly which papers and graph nodes led to the breakthrough.

## 5. User Stories (Behavioral Expectations)

### Story A: The "Mechanism of Action" Discovery

*   **Context:** A drug works clinically, but the mechanism is unknown ("Black Box Efficacy").
*   **Action:** episteme scans the protein interaction network around the drug's known ligands.
*   **Bridge:** Identifies a weak affinity for "Receptor X," which is expressed in the target tissue.
*   **Simulation:** inference confirms that inhibiting Receptor X would theoretically produce the observed clinical results.
*   **Result:** Proposes "Receptor X" as the true MOA target.

### Story B: The "Resistance Breaker"

*   **Context:** Patients are developing resistance to a standard cancer therapy.
*   **Gap Scan:** episteme identifies a bypass pathway that is upregulated in resistant cells but not targeted by the current drug.
*   **Hypothesis:** Proposes a "Combination Therapy" targeting the bypass pathway.
*   **Critique:** The "Toxicologist" persona flags that this combination might cause liver failure.
*   **Refinement:** episteme adjusts the hypothesis to target a downstream effector that avoids the liver toxicity loop.

## 6. Data Schema

### HypothesisManifest

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class ConfidenceLevel(str, Enum):
    SPECULATIVE = "SPECULATIVE"
    PLAUSIBLE = "PLAUSIBLE"
    PROBABLE = "PROBABLE"

class GeneticTarget(BaseModel):
    symbol: str
    ensembl_id: str
    druggability_score: float  # From coreason-prism
    novelty_score: float

class Hypothesis(BaseModel):
    id: str
    title: str
    knowledge_gap: str  # The problem statement
    proposed_mechanism: str  # The biological pathway
    target_candidate: GeneticTarget

    # Validation
    causal_validation_score: float
    key_counterfactual: str

    # Experimental Design
    killer_experiment_pico: dict

    # Provenance
    evidence_chain: List[str]  # Links to source papers/nodes
    confidence: ConfidenceLevel
```

## 7. Implementation Directives for the Coding Agent

1.  **Graph Traversal:** Use networkx for initial hops, but delegate complex pathfinding to coreason-graph-nexus (Neo4j) to handle scale.
2.  **Strict Null Hypothesis:** The default state of any hypothesis is False. The code must explicitly seek *disconfirming* evidence (e.g., "Search for papers that say Gene X is NOT involved in Disease Y").
3.  **Hallucination Check:** Before returning a hypothesis, the system must run a "Citation Verification" loop using coreason-search. If the referenced biological interaction does not exist in the retrieved text, the hypothesis is flagged as INVALID.
4.  **Interface:** Expose a simple API: `episteme.generate_hypothesis(disease_id: str) -> List[Hypothesis]`.

This PRD positions coreason-episteme not just as a search tool, but as a **generative reasoning engine** that bridges the gap between static knowledge and novel discovery.
