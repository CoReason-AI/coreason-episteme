# Usage Guide

`coreason-episteme` can be used in two primary modes: as a standalone **Python Library** or as a **Microservice**.

## 1. Python Library Usage

You can import `coreason-episteme` directly into your Python code to perform hypothesis generation tasks. This is useful for scripts, notebooks, or integrating into other Python applications.

### Basic Example

```python
from typing import Any, Dict, List, Optional
from coreason_episteme.main import generate_hypothesis
from coreason_episteme.models import GeneticTarget, KnowledgeGap

# 1. Define Mock Clients (Simulate external services)
# In production, replace these with actual client implementations or use the provided HttpClients.
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

## 2. Microservice Usage

`coreason-episteme` exposes a RESTful API using FastAPI, allowing it to function as the "Theorist" service within a larger microservices architecture.

### Running the Server

You can run the server using `uvicorn`:

```bash
uvicorn coreason_episteme.server:app --host 0.0.0.0 --port 8000
```

### Docker Usage

The project includes a `Dockerfile` for easy deployment.

**Build the image:**
```bash
docker build -t coreason-episteme:latest .
```

**Run the container:**
```bash
docker run -p 8000:8000 \
  -e GRAPH_NEXUS_URL="http://graph-nexus:8000" \
  -e CODEX_URL="http://codex:8000" \
  -e SEARCH_URL="http://search:8000" \
  -e PRISM_URL="http://prism:8000" \
  -e INFERENCE_URL="http://inference:8000" \
  -e VERITAS_URL="http://veritas:8000" \
  coreason-episteme:latest
```

### API Endpoints

#### Health Check

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.3.0"
}
```

#### Generate Hypothesis

**Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "disease_id": "DOID:12345",
    "max_retries": 5,
    "user_id": "researcher-1"
  }'
```

**Response:**
```json
[
  {
    "id": "...",
    "title": "Hypothesis Title",
    "knowledge_gap": "...",
    "proposed_mechanism": "...",
    "target_candidate": {
      "symbol": "GENE_X",
      "ensembl_id": "ENSG...",
      "druggability_score": 0.9,
      "novelty_score": 0.8
    },
    "causal_validation_score": 0.85,
    "key_counterfactual": "...",
    "killer_experiment_pico": {
      "population": "...",
      "intervention": "...",
      "comparator": "...",
      "outcome": "..."
    },
    "evidence_chain": ["PMID:123"],
    "confidence": "PROBABLE",
    "critiques": []
  }
]
```
